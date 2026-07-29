# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DamageTriageFormer inference workflow, scoped to an analyst-drawn AOI
or an explicit building selection — the per-building-crop path from
``docs/OptionB_AOI_Integration_Plan.md`` (plan.pdf Modules 5/6, Part 01),
as opposed to ``damage_triage_infer.py``'s whole-mosaic tile-batch path.

Driven by a single ``--config`` JSON file (written by
``DTFAoiInferencePostprocessor._execute_inference``,
``core/processors/damage_triage_aoi.py``), which carries:

* the post-event mosaic COG path,
* the PostGIS-resolved building shortlist (``[{building_id, geometry}]``,
  already scoped by ``postgis_query.buildings_in_aoi`` — this workflow
  never sees buildings outside that shortlist, and never re-queries
  PostGIS itself — it has no DB connectivity),
* the checkpoint paths.

Unlike ``damage_triage_infer.py`` (one forward pass per 1024x1024 tile,
batched over every intersecting footprint's mask), this workflow does one
forward pass **per building**: ``crop_service.crop_building`` extracts a
padded window around just that building, and a single binary instance
mask (that building's own geometry only) is rasterized against it — the
literal "Job 1: B101 / Job 2: B102 / ..." framing from plan.pdf Module 6,
made tractable by loading the DINOv3 checkpoint once for the whole run
rather than once per building (see ``run_damage_triage_aoi``).

Reuses — does NOT reimplement — ``damage_triage_infer.py``'s
checkpoint-loading and head-decoding logic (``build_dtf_model``,
``decode_gated_predictions``, ``decode_severity_predictions``,
``DTF_CATEGORIES``, ``DTF_IM_SIZE``, ``DTF_IDX_TO_COLOR``): that file is
the single source of truth for "how do you load this exact checkpoint and
turn its output into 5 class probabilities" (see its own docstring on the
gated-head architecture mismatch trap) — this workflow only supplies a
different outer loop (per-building crop vs. per-tile batch).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import rasterio  # type: ignore
import torch

from hastegeo.core.utils.crop_service import (
    crop_building,
    crop_building_mask,
    write_png,
)
from hastegeo.core.utils.logs import Logger as HasteLogger
from hastegeo.workflows.damage_triage_infer import (
    DTF_CATEGORIES,
    DTF_IDX_TO_COLOR,
    DTF_IM_SIZE,
    build_dtf_model,
    decode_gated_predictions,
    decode_severity_predictions,
)

WORKDIR = os.getenv("WORKDIR", ".")
LOG_DIR = os.path.join(WORKDIR, "logs")
LOG_FILE = "damage_triage_aoi_verbose.log"
os.makedirs(LOG_DIR, exist_ok=True)

logger = HasteLogger.get_logger(
    "damage_triage_aoi_infer", log_dir=LOG_DIR, log_file=LOG_FILE
)
logger.info("Executing %s", __file__)

DEFAULT_PAD_PX = 32


def crop_class_mask_to_png(class_idx: int, crop_size: int) -> np.ndarray:
    """Solid-color (3, H, W) PNG payload for one building's predicted class.

    Uses the same ``DTF_IDX_TO_COLOR`` palette as the whole-layer
    workflow's ``write_visualizer_rasters`` (index = ``dtf_class + 1``, 0
    is nodata) so a building's per-crop mask thumbnail and the whole-layer
    map's damage-class colors read as the same visual scale.
    """
    rgba = DTF_IDX_TO_COLOR[class_idx + 1]
    mask = np.zeros((3, crop_size, crop_size), dtype=np.uint8)
    for c in range(3):
        mask[c, :, :] = rgba[c]
    return mask


def run_damage_triage_aoi(
    image_path: str,
    buildings: list,
    checkpoint_path: str,
    checkpoint_config_path: str,
    output_dir: str,
    *,
    pad_px: int = DEFAULT_PAD_PX,
    crop_size: int = DTF_IM_SIZE,
) -> list:
    """Run DamageTriageFormer once per building in ``buildings``.

    Args:
        image_path: local path to the post-event mosaic COG.
        buildings: ``[{"building_id": str, "geometry": <GeoJSON, EPSG:4326>}, ...]``
            — the PostGIS shortlist from
            ``hastegeo.core.utils.postgis_query.buildings_in_aoi``.
        checkpoint_path / checkpoint_config_path: DamageTriageFormer
            checkpoint + its training-run config.json (see
            ``damage_triage_infer.py::build_dtf_model``).
        output_dir: where ``building_<id>.png`` / ``mask_<id>.png`` are
            written.

    Returns:
        ``[{building_id, class, class_name, confidence, probs,
        crop_filename, mask_filename}, ...]``, in the same order as
        ``buildings``.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_total = len(buildings)
    log_progress(f"{n_total} buildings in scope for this AOI run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Loading DamageTriageFormer checkpoint on {device}...")
    # Loaded ONCE for the whole run, not once per building — reloading a
    # ~500MB ViT-L checkpoint per building would make even a modest AOI
    # (thousands of buildings) impractical. See "Biggest pitfalls" #4 in
    # docs/OptionB_AOI_Integration_Plan.md.
    model, use_gated = build_dtf_model(
        checkpoint_path, checkpoint_config_path, device
    )
    log_progress(
        f"Model architecture: {'gated head' if use_gated else 'severity head'}"
    )

    results = []
    with rasterio.open(image_path) as raster:
        for i, building in enumerate(buildings):
            building_id = building["building_id"]
            geom_4326 = building["geometry"]

            crop, crop_transform, geom_raster_crs = crop_building(
                raster, geom_4326, pad_px=pad_px, crop_size=crop_size
            )
            mask = crop_building_mask(
                geom_raster_crs, crop_transform, crop_size
            )

            crop_t = (
                torch.from_numpy(crop.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(device)
            )
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

            with torch.inference_mode():
                try:
                    _, damage_out = model(crop_t, [mask_t])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"CUDA OOM on building {building_id}; retrying on CPU"
                    )
                    model.to("cpu")
                    _, damage_out = model(crop_t.cpu(), [mask_t.cpu()])
                    model.to(device)
                probs = (
                    decode_gated_predictions(damage_out)
                    if use_gated
                    else decode_severity_predictions(damage_out)
                )

            if probs.shape[0] == 0:
                logger.warning(
                    f"Building {building_id} produced zero valid mask "
                    "pixels; skipping (no crop/mask written)."
                )
                results.append(
                    {
                        "building_id": building_id,
                        "class": -1,
                        "class_name": "Unclassified",
                        "confidence": 0.0,
                        "probs": [0.0] * len(DTF_CATEGORIES),
                        "crop_url": "",
                        "mask_url": "",
                    }
                )
                continue

            class_idx = int(probs[0].argmax())
            confidence = float(probs[0].max())

            crop_filename = f"building_{building_id}.png"
            mask_filename = f"mask_{building_id}.png"
            write_png(os.path.join(output_dir, crop_filename), crop)
            write_png(
                os.path.join(output_dir, mask_filename),
                crop_class_mask_to_png(class_idx, crop_size),
            )

            results.append(
                {
                    "building_id": building_id,
                    "class": class_idx,
                    "class_name": DTF_CATEGORIES[class_idx],
                    "confidence": confidence,
                    "probs": [float(p) for p in probs[0]],
                    "crop_url": crop_filename,
                    "mask_url": mask_filename,
                }
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                log_progress(f"Scored building {i + 1}/{n_total}")

    log_progress(f"Classified {len(results)}/{n_total} buildings")
    return results


def write_predictions_gpkg(buildings: list, results: list, output_path: str) -> int:
    """Write an AOI-scoped predictions GeoPackage — only the buildings this
    run touched, not the whole layer.

    Built entirely from data already in this container (``buildings``'
    geometries, as resolved by ``postgis_query.buildings_in_aoi`` and
    passed through the job config, plus this run's ``results``) — this
    workflow has no PostGIS connectivity of its own (see module
    docstring), so unlike a live re-export it reflects predictions at
    scoring time only, exactly mirroring how the whole-layer workflow
    (``damage_triage_infer.py``) builds its own predictions GeoPackage
    self-contained within the training container rather than round-
    tripping through a database.
    """
    import geopandas as gpd  # type: ignore
    from shapely.geometry import shape as shapely_shape  # type: ignore

    results_by_id = {r["building_id"]: r for r in results}
    rows = []
    for building in buildings:
        building_id = building["building_id"]
        result = results_by_id.get(building_id)
        if result is None:
            continue
        rows.append(
            {
                "id": building_id,
                "dtf_class": result["class"],
                "dtf_class_name": result["class_name"],
                "dtf_confidence": result["confidence"],
                "crop_url": result.get("crop_url", ""),
                "mask_url": result.get("mask_url", ""),
                "geometry": shapely_shape(building["geometry"]),
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(output_path, layer="predictions", driver="GPKG")
    return len(gdf)


def log_progress(message):
    """Append a friendly progress line, mirroring damage_triage_infer.py's
    contract with DTFAoiInferencePostprocessor's friendly-log polling."""
    logger.info(message)
    log_file = os.path.join(LOG_DIR, "damage_triage_aoi_friendly.log")
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
    buildings = config.get("buildings", [])
    model_id = config.get("model_id", "unknown")
    gpkg_name = files.get(
        "predictions_gpkg", f"dtf_aoi_predictions_{model_id}.gpkg"
    )
    checkpoint_path = (
        checkpoint_cache.get("local_checkpoint")
        or files.get("checkpoint")
        or os.getenv("DTF_CHECKPOINT_PATH", "/opt/dtf-checkpoint/best_model.pth")
    )
    checkpoint_config_path = (
        checkpoint_cache.get("local_config")
        or files.get("checkpoint_config")
        or os.getenv("DTF_CHECKPOINT_CONFIG_PATH", "/opt/dtf-checkpoint/config.json")
    )

    try:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagery not found: {image_path}")
        if not buildings:
            raise ValueError(
                "Config has no buildings — the PostGIS shortlist for this "
                "AOI/selection was empty."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(checkpoint_config_path):
            raise FileNotFoundError(
                f"Checkpoint config not found: {checkpoint_config_path}"
            )

        log_progress("Running AOI-scoped DamageTriageFormer inference")
        results = run_damage_triage_aoi(
            image_path,
            buildings,
            checkpoint_path,
            checkpoint_config_path,
            output_dir,
            pad_px=int(pipeline.get("pad_px", DEFAULT_PAD_PX)),
            crop_size=int(pipeline.get("crop_size", DTF_IM_SIZE)),
        )

        gpkg_path = os.path.join(output_dir, os.path.basename(gpkg_name))
        log_progress(f"Writing {len(results)} buildings -> {gpkg_path}")
        write_predictions_gpkg(buildings, results, gpkg_path)

        manifest = {
            "num_buildings": len(buildings),
            "num_classified": sum(1 for r in results if r["class"] != -1),
            "categories": DTF_CATEGORIES,
            "buildings": results,
            "predictions_gpkg_filename": os.path.basename(gpkg_path),
        }
        log_progress("Finalizing outputs")
        with open(
            os.path.join(output_dir, "damage_triage_aoi_manifest.json"), "w"
        ) as f:
            json.dump(manifest, f, indent=4)
        logger.info(
            "AOI-scoped DamageTriageFormer inference completed successfully."
        )
    except Exception as e:
        logger.error(
            "Error during AOI-scoped DamageTriageFormer inference",
            exc_info=True,
        )
        log_progress(f"Error during AOI-scoped DamageTriageFormer inference: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during main execution: {e}", exc_info=True)
        sys.exit(1)
