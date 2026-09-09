# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DamageTriageFormer inference workflow for the DTF sub-model (Option B).

Ported to the same HASTE workflow contract ``embed_buildings.py`` uses:

* Driven by a single ``--config`` JSON file (written by
  ``DTFInferencePreprocessor`` / ``DTFInferencePostprocessor``,
  ``core/processors/damage_triage.py``).
* Reads the post-event mosaic COG and the cached building-footprints
  GeoPackage (both downloaded into the task working dir by the runner).
* Unlike the per-building-crop MOSAIKS embedding workflow, DamageTriageFormer
  (DINOv3 ViT-L/16 backbone + mask-pooled instance heads, see
  ``Damage-TriageFormer/src/model/net.py``) consumes a full 1024x1024 tile
  plus a batch of per-building binary instance masks at tile resolution — the
  model downsamples masks to its feature-map stride internally
  (``SeverityHead.forward``, ``Damage-TriageFormer/src/model/heads.py``).
  So this workflow tiles the mosaic into non-overlapping ``IM_SIZE`` windows,
  and for each tile rasterizes one binary mask per intersecting footprint.
* A building that straddles a tile boundary is scored once per tile it
  appears in; we keep the highest-confidence prediction across tiles.
* Writes outputs to ``outputs/``: a per-building predictions GeoPackage
  (layer ``"predictions"``), a manifest, and two map-ready rasters —
  ``*_predictions.tif`` (single-band, per-pixel ``dtf_class + 1``, 0 =
  nodata) and ``*_visualizer.tif`` (RGBA, footprints painted by damage
  class) — mirroring the embedding sub-model's ``output2visualizer.py`` /
  ``run_workflow.py`` COG pipeline so ``GetVisualizerResults`` and the
  existing Visualizer map (``predictedDamageLayer`` / ``predictionsLayer``)
  work for DTF models unchanged. Row-order/id convention matches
  ``embed_buildings.py`` and ``PutBuildingPredictions``: exactly one row per
  input footprint, ``id = 0..N-1`` in the footprints file's native order —
  ``GetValidationReport`` / ``GetAssessmentReport`` join by that row index.

CRITICAL — row-order invariant: buildings with no overlap to any tile (or
zero valid mask pixels) keep their ``id`` slot with ``dtf_class = -1``
("Unclassified") rather than being dropped or reordered.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.windows
import torch
import torch.nn.functional as F
from hastegeo.core.utils.logs import Logger as HasteLogger
from rasterio.features import geometry_mask
from shapely.geometry import box

def _find_local_checkpoint():
    """Look for pre-downloaded best_model.pth and config.json in local directories."""
    env_ckpt = os.getenv("DTF_CHECKPOINT_PATH")
    env_cfg = os.getenv("DTF_CHECKPOINT_CONFIG_PATH")
    if env_ckpt and env_cfg and os.path.exists(env_ckpt) and os.path.exists(env_cfg):
        return env_ckpt, env_cfg

    candidate_dirs = []

    cwd = Path.cwd()
    candidate_dirs.append(cwd / "models" / "damage_triage")
    for parent in cwd.parents:
        candidate_dirs.append(parent / "models" / "damage_triage")

    file_dir = Path(__file__).resolve().parent
    for parent in file_dir.parents:
        candidate_dirs.append(parent / "models" / "damage_triage")
        candidate_dirs.append(parent / "Damage assesment" / "damage-assesment-ml" / "models" / "damage_triage")

    candidate_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    candidate_dirs.append(Path("/opt/dtf-checkpoint"))
    candidate_dirs.append(Path("/app/models/damage_triage"))

    for d in candidate_dirs:
        if not d.exists():
            continue
        ckpt = d / "best_model.pth"
        cfg = d / "config.json"
        if not cfg.exists():
            cfg = d / "dtf_config.json"
        if ckpt.exists() and cfg.exists():
            return str(ckpt), str(cfg)

    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.exists():
        found_ckpts = list(hf_hub.glob("**/best_model.pth"))
        found_cfgs = list(hf_hub.glob("**/config.json"))
        if found_ckpts and found_cfgs:
            return str(found_ckpts[0]), str(found_cfgs[0])

    return None, None

WORKDIR = os.getenv("WORKDIR", ".")
LOG_DIR = os.path.join(WORKDIR, "logs")
LOG_FILE = "damage_triage_verbose.log"
os.makedirs(LOG_DIR, exist_ok=True)

logger = HasteLogger.get_logger(
    "damage_triage_infer", log_dir=LOG_DIR, log_file=LOG_FILE
)
logger.info("Executing %s", __file__)

# Must match Damage-TriageFormer/src/config.py CATEGORIES exactly (checkpoint
# class-index order). Kept as a local constant so this workflow doesn't need
# the training repo's config module importable at runtime — only its
# `src.model.net.Model` class (see build_dtf_model below).
DTF_CATEGORIES = [
    "Undamaged",
    "Partial Roof Damage",
    "Total Roof Damage",
    "Partial Structural Collapse",
    "Total Structural Collapse",
]
DTF_IM_SIZE = 1024
UNCLASSIFIED = -1

# Where the training docker image bakes the cloned Damage-TriageFormer repo
# (its `src` package must be importable to reconstruct the model + load the
# checkpoint). Override for local dev runs outside the image.
DTF_REPO_ROOT = os.getenv("DTF_REPO_ROOT", "/opt/damage-triage-former")

# RGBA palette for the map visualizer raster, indexed by (dtf_class + 1) —
# index 0 is nodata/unclassified (transparent). Same 5-color damage ramp
# (white -> dark red) as the embedding sub-model's output2visualizer.py
# IDX_TO_COLOR, so both sub-models render with a visually consistent scale
# on the Visualizer map.
DTF_IDX_TO_COLOR = np.array(
    [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [252, 190, 165, 255],
        [251, 112, 80, 255],
        [211, 32, 32, 255],
        [103, 0, 13, 255],
    ],
    dtype=np.uint8,
)


def build_dtf_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load Damage-TriageFormer's Model class and checkpoint weights.

    IMPORTANT: the released checkpoint (Ymx1025/DamageTriageFormer-model,
    ``best_model.pth`` + ``config.json``) was trained with
    ``use_gated_head=True, use_sfp=True, sfp_stride=8, no_multiscale=True,
    no_attention_pool=True, use_aux_severity=True`` — NOT the plain 5-way
    ``SeverityHead`` (confirmed by inspecting the shipped ``config.json``'s
    training args). The gated head is a binary damaged/undamaged **gate**
    plus a conditional 4-way **leaf** classifier over the damage
    sub-types — see ``decode_gated_predictions`` for how this becomes a
    5-class prediction. Loading with the wrong head shape would silently
    drop most of the damage-head weights under ``strict=False``.

    Mirrors main.py's checkpoint-loading branch (Damage-TriageFormer/main.py):
    a saved ``last_checkpoint.pth`` is a dict with a ``model_state_dict`` key;
    a bare ``best_model.pth`` (what the HF repo ships) is a raw state_dict.
    """
    if DTF_REPO_ROOT not in sys.path:
        sys.path.insert(0, DTF_REPO_ROOT)
    from src.model.net import Model as DTFModel  # noqa: E402 (see sys.path above)

    with open(config_path) as f:
        train_cfg = json.load(f)

    model = DTFModel(
        im_size=DTF_IM_SIZE,
        num_classes=len(DTF_CATEGORIES),
        use_hierarchical=bool(train_cfg.get("use_hierarchical", False)),
        use_gated_head=bool(train_cfg.get("use_gated_head", False)),
        use_aux_severity=bool(train_cfg.get("use_aux_severity", False)),
        use_sfp=bool(train_cfg.get("use_sfp", False)),
        sfp_stride=int(train_cfg.get("sfp_stride", 4)),
        use_multiscale=not train_cfg.get("no_multiscale", False),
        use_attention=not train_cfg.get("no_attention_pool", False),
        drop_path_rate=float(train_cfg.get("drop_path", 0.0)),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            f"Checkpoint load: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys "
            f"(architecture derived from {config_path})"
        )
    model.to(device)
    model.eval()
    return model, model.use_gated_head


def compute_tile_windows(raster_width: int, raster_height: int, tile_size: int):
    """Non-overlapping tile windows covering the raster, row-major order."""
    windows = []
    for row_off in range(0, raster_height, tile_size):
        h = min(tile_size, raster_height - row_off)
        for col_off in range(0, raster_width, tile_size):
            w = min(tile_size, raster_width - col_off)
            windows.append(
                rasterio.windows.Window(col_off, row_off, w, h)
            )
    return windows


def rasterize_instance_masks(
    footprints: gpd.GeoDataFrame,
    tile_window: rasterio.windows.Window,
    tile_transform: rasterio.Affine,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize every footprint that intersects this tile into its own
    binary mask at (tile_size, tile_size), zero-padded to full tile size
    (edge tiles read fewer source pixels than ``tile_size``).

    Returns ``(row_indices, masks)`` where ``masks`` has shape
    ``(N_intersecting, tile_size, tile_size)``.
    """
    tile_bounds_geom = box(
        *rasterio.windows.bounds(tile_window, tile_transform)
    )
    hits = footprints[footprints.geometry.intersects(tile_bounds_geom)]
    if hits.empty:
        return np.empty(0, dtype=np.int64), np.empty(
            (0, tile_size, tile_size), dtype=np.float32
        )

    row_indices = hits.index.to_numpy()
    masks = np.zeros((len(hits), tile_size, tile_size), dtype=np.float32)
    for i, geom in enumerate(hits.geometry):
        masks[i] = geometry_mask(
            [geom],
            out_shape=(tile_size, tile_size),
            transform=tile_transform,
            all_touched=True,
            invert=True,
        ).astype(np.float32)
    return row_indices, masks


def decode_gated_predictions(damage_out) -> np.ndarray:
    """Turn a GatedDamageHead output for one image into a (N, 5) prob matrix.

    ``damage_out`` is ``(gate_list, leaf_list)`` or, when the checkpoint was
    trained with ``use_aux_severity`` (true for the released checkpoint),
    ``(gate_list, leaf_list, aux_list)`` — see
    ``Damage-TriageFormer/src/model/gated_head.py:GatedDamageHead.forward``.
    Each element is a per-image list; this decodes image index 0 (this
    workflow always runs one tile per forward call).

    The gate is a binary damaged/undamaged classifier (``sigmoid``); the
    leaf is a 4-way classifier over damage sub-types, conditional on
    ``damaged``. Decomposed into the same 5-class probability space as
    ``DTF_CATEGORIES`` (sums to 1 per row): ``P(0) = 1 - P(damaged)``,
    ``P(i+1) = P(damaged) * P(leaf=i)`` for leaf class i in 0..3.
    """
    gate_list, leaf_list = damage_out[0], damage_out[1]
    gate_logits = gate_list[0]  # (N,)
    leaf_logits = leaf_list[0]  # (N, 4)
    n = gate_logits.shape[0]
    if n == 0:
        return np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)

    p_damaged = torch.sigmoid(gate_logits).unsqueeze(-1)  # (N, 1)
    leaf_probs = F.softmax(leaf_logits, dim=-1)  # (N, 4)
    probs = torch.cat([1.0 - p_damaged, p_damaged * leaf_probs], dim=-1)  # (N, 5)
    return probs.cpu().numpy()


def decode_severity_predictions(damage_out) -> np.ndarray:
    """Turn a plain SeverityHead output for one image into a (N, 5) prob matrix.

    ``damage_out`` is ``[sev_logits_list]`` (see ``Model.forward``'s
    non-gated, non-hierarchical branch); ``sev_logits_list[0]`` is ``(N, 5)``.
    Only used if a future checkpoint is trained without the gated head.
    """
    logits = damage_out[0][0]
    if logits.shape[0] == 0:
        return np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)
    return F.softmax(logits, dim=-1).cpu().numpy()


def read_tile(raster, window: rasterio.windows.Window, tile_size: int) -> np.ndarray:
    """Read an RGB tile, zero-padding edge tiles up to (3, tile_size, tile_size)."""
    data = raster.read([1, 2, 3], window=window)
    _, h, w = data.shape
    if h == tile_size and w == tile_size:
        return data
    padded = np.zeros((3, tile_size, tile_size), dtype=data.dtype)
    padded[:, :h, :w] = data
    return padded


def write_visualizer_rasters(
    work: gpd.GeoDataFrame,
    best_class: np.ndarray,
    raster_profile: dict,
    output_prefix: str,
) -> tuple[str, str]:
    """Paint per-building ``dtf_class`` onto the mosaic's pixel grid and
    write the two map-ready rasters the Visualizer consumes: a single-band
    ``*_predictions.tif`` (raw ``dtf_class + 1``, 0 = nodata/unclassified —
    served as the hidden ``predictionsLayer`` via a TiTiler colormap query
    param) and an RGBA ``*_visualizer.tif`` COG (footprints pre-colored by
    ``DTF_IDX_TO_COLOR`` — the always-visible ``predictedDamageLayer``).

    Mirrors the embedding sub-model's ``output2visualizer.py`` +
    ``run_workflow.py`` gdal_translate COG step so both sub-models produce
    map layers ``GetVisualizerResults`` can serve identically.

    ``work`` must already be in the raster's CRS (see ``run_damage_triage``,
    which reprojects footprints into ``raster.crs`` before scoring).
    """
    height, width = raster_profile["height"], raster_profile["width"]
    transform = raster_profile["transform"]

    # dtf_class in [-1 (unclassified), 0..4] -> raster index in [0..5].
    shape_vals = [
        (geom, int(cls) + 1)
        for geom, cls in zip(work.geometry, best_class)
        if geom is not None and not geom.is_empty
    ]
    class_idx = rasterio.features.rasterize(
        shape_vals,
        out_shape=(height, width),
        transform=transform,
        fill=0,
    ).astype(np.uint8)

    predictions_fn = f"{output_prefix}_predictions.tif"
    pred_profile = raster_profile.copy()
    pred_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="lzw",
        predictor=2,
        nodata=0,
        blockxsize=512,
        blockysize=512,
        tiled=True,
        interleave="pixel",
    )
    with rasterio.open(predictions_fn, "w", **pred_profile) as f:
        f.write(class_idx, 1)

    colors = DTF_IDX_TO_COLOR[class_idx].transpose(2, 0, 1)
    temp_vis_fn = f"{output_prefix}_temp_vis.tif"
    vis_profile = raster_profile.copy()
    vis_profile.update(count=4, nodata=0)
    with rasterio.open(temp_vis_fn, "w", **vis_profile) as f:
        f.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]
        f.write(colors)

    visualizer_fn = f"{output_prefix}_visualizer.tif"
    subprocess.run(
        ["gdal_translate", "-of", "COG", temp_vis_fn, visualizer_fn],
        check=True,
    )
    os.remove(temp_vis_fn)

    return predictions_fn, visualizer_fn


def run_damage_triage(
    image_path: str,
    footprints_path: str,
    checkpoint_path: str,
    checkpoint_config_path: str,
    *,
    tile_size: int = DTF_IM_SIZE,
    visualizer_output_prefix: str = None,
) -> tuple[gpd.GeoDataFrame, tuple]:
    """Run DamageTriageFormer over every tile and return a row-aligned
    GeoDataFrame with exactly one row per input footprint (see module
    docstring for the row-order invariant)."""
    log_progress(f"Reading footprints from {footprints_path}")
    footprints = gpd.read_file(footprints_path)
    if footprints.empty:
        raise ValueError("Footprints file is empty.")
    if footprints.crs is None:
        raise ValueError("Footprints have no CRS.")
    footprints = footprints.reset_index(drop=True)
    n_total = len(footprints)
    log_progress(f"{n_total} footprints loaded (CRS: {footprints.crs})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Loading DamageTriageFormer checkpoint on {device}...")
    model, use_gated = build_dtf_model(
        checkpoint_path, checkpoint_config_path, device
    )
    log_progress(
        f"Model architecture: {'gated head' if use_gated else 'severity head'}"
    )

    raster = rasterio.open(image_path)
    if footprints.crs != raster.crs:
        log_progress(f"Reprojecting footprints {footprints.crs} -> {raster.crs}")
        work = footprints.copy()
        work["geometry"] = footprints.to_crs(raster.crs).geometry
    else:
        work = footprints

    windows = compute_tile_windows(raster.width, raster.height, tile_size)
    log_progress(
        f"Image: {raster.width}x{raster.height}, tiling into "
        f"{len(windows)} {tile_size}x{tile_size} windows"
    )

    best_class = np.full(n_total, UNCLASSIFIED, dtype=np.int64)
    best_conf = np.zeros(n_total, dtype=np.float32)
    best_probs = np.zeros((n_total, len(DTF_CATEGORIES)), dtype=np.float32)
    seen = np.zeros(n_total, dtype=bool)

    for t_idx, window in enumerate(windows):
        tile_transform = rasterio.windows.transform(window, raster.transform)
        row_indices, masks = rasterize_instance_masks(
            work, window, tile_transform, tile_size
        )
        if len(row_indices) == 0:
            continue

        tile = read_tile(raster, window, tile_size).astype(np.float32) / 255.0
        tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        with torch.inference_mode():
            try:
                _, damage_out = model(tile_t, [masks_t])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning(
                    f"CUDA OOM on tile {t_idx}; retrying on CPU"
                )
                model.to("cpu")
                _, damage_out = model(tile_t.cpu(), [masks_t.cpu()])
                model.to(device)
            probs = (
                decode_gated_predictions(damage_out)
                if use_gated
                else decode_severity_predictions(damage_out)
            )

        classes = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        for k, row_idx in enumerate(row_indices):
            seen[row_idx] = True
            if confs[k] > best_conf[row_idx]:
                best_conf[row_idx] = confs[k]
                best_class[row_idx] = int(classes[k])
                best_probs[row_idx] = probs[k]

        if device.type == "cuda":
            torch.cuda.empty_cache()
        log_progress(f"Scored tile {t_idx + 1}/{len(windows)}")

    raster_profile = raster.profile.copy()
    raster.close()

    n_unclassified = int((~seen).sum())
    log_progress(
        f"Classified {n_total - n_unclassified}/{n_total} buildings "
        f"({n_unclassified} outside raster extent kept as Unclassified)"
    )

    out = footprints.copy()
    out.insert(0, "id", range(n_total))
    out["dtf_class"] = best_class
    out["dtf_class_name"] = [
        DTF_CATEGORIES[c] if c != UNCLASSIFIED else "Unclassified"
        for c in best_class
    ]
    out["dtf_confidence"] = best_conf
    for i, name in enumerate(DTF_CATEGORIES):
        out[f"dtf_prob_{i}"] = best_probs[:, i]
    # Backward-compatible columns so the existing Validation/Assessment
    # reports (which read `damaged`/`damage_pct_0m`/`unknown_pct`, written by
    # PutBuildingPredictions for the embedding sub-model) keep working
    # unchanged for a DTF model row.
    out["damaged"] = (best_class > 0).astype(int)
    out["damage_pct_0m"] = out["damaged"].astype(float)
    out["unknown_pct"] = [
        0.0 if s else 1.0 for s in seen
    ]
    try:
        out["area"] = footprints.geometry.to_crs(epsg=6933).area
    except Exception:
        out["area"] = None
    out = gpd.GeoDataFrame(out, geometry=footprints.geometry.name, crs=footprints.crs)

    raster_fns = None
    if visualizer_output_prefix:
        log_progress("Rendering map visualizer layers")
        raster_fns = write_visualizer_rasters(
            work, best_class, raster_profile, visualizer_output_prefix
        )

    return out.to_crs(epsg=4326), raster_fns


def log_progress(message):
    """Append a friendly progress line consumed by DTFInferencePostprocessor."""
    logger.info(message)
    log_file = os.path.join(LOG_DIR, "damage_triage_friendly.log")
    with open(log_file, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}|{message}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config JSON file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("No config file found at location %s", args.config)
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)
    if not config:
        logger.error("Config file is empty")
        sys.exit(1)

    output_dir = os.path.join(WORKDIR, config.get("output_dir", "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    files = config.get("files", {})
    checkpoint_cache = config.get("checkpoint_cache", {})
    pipeline = config.get("pipeline", {})
    image_path = files.get("imagery")
    footprints_path = files.get("footprints")
    checkpoint_path = (
        checkpoint_cache.get("local_checkpoint")
        or files.get("checkpoint")
        or os.getenv("DTF_CHECKPOINT_PATH", "/opt/dtf-checkpoint/best_model.pth")
    )
    # The training-run config.json shipped alongside best_model.pth on
    # Hugging Face — NOT hastegeo's job config (args.config). Required to
    # reconstruct the checkpoint's actual head architecture (gated vs
    # plain severity) before loading weights; see build_dtf_model.
    checkpoint_config_path = (
        checkpoint_cache.get("local_config")
        or files.get("checkpoint_config")
        or os.getenv("DTF_CHECKPOINT_CONFIG_PATH", "/opt/dtf-checkpoint/config.json")
    )
    gpkg_name = files.get(
        "predictions_gpkg", f"dtf_predictions_{config.get('model_id')}.gpkg"
    )

    try:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagery not found: {image_path}")
        if not footprints_path or not os.path.exists(footprints_path):
            raise FileNotFoundError(f"Footprints not found: {footprints_path}")
        if not os.path.exists(checkpoint_path) or not os.path.exists(checkpoint_config_path):
            local_ckpt, local_cfg = _find_local_checkpoint()
            if local_ckpt and local_cfg:
                checkpoint_path, checkpoint_config_path = local_ckpt, local_cfg
                logger.info(f"Resolved existing local DamageTriageFormer checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(checkpoint_config_path):
            raise FileNotFoundError(
                f"Checkpoint config not found: {checkpoint_config_path}"
            )

        log_progress("Running DamageTriageFormer inference")
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        visualizer_prefix = os.path.join(output_dir, image_stem)
        gdf, raster_fns = run_damage_triage(
            image_path,
            footprints_path,
            checkpoint_path,
            checkpoint_config_path,
            tile_size=int(pipeline.get("tile_size", DTF_IM_SIZE)),
            visualizer_output_prefix=visualizer_prefix,
        )

        gpkg_path = os.path.join(output_dir, os.path.basename(gpkg_name))
        log_progress(f"Writing {len(gdf)} buildings -> {gpkg_path}")
        gdf.to_file(gpkg_path, layer="predictions", driver="GPKG")

        manifest = {
            "predictions_gpkg_filename": os.path.basename(gpkg_path),
            "num_buildings": int(len(gdf)),
            "num_classified": int((gdf["dtf_class"] != UNCLASSIFIED).sum()),
            "categories": DTF_CATEGORIES,
        }
        if raster_fns:
            predictions_tif_fn, visualizer_tif_fn = raster_fns
            manifest["predictions_tif_filename"] = os.path.basename(
                predictions_tif_fn
            )
            manifest["visualizer_tif_filename"] = os.path.basename(
                visualizer_tif_fn
            )
        log_progress("Finalizing outputs")
        with open(
            os.path.join(output_dir, "damage_triage_manifest.json"), "w"
        ) as f:
            json.dump(manifest, f, indent=4)
        logger.info("DamageTriageFormer inference completed successfully.")
    except Exception as e:
        logger.error("Error during DamageTriageFormer inference", exc_info=True)
        log_progress(f"Error during DamageTriageFormer inference: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during main execution: {e}", exc_info=True)
        sys.exit(1)
