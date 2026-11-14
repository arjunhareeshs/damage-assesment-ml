# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DamageTriageFormer batched region inference runner.

Runs inside the haste-training container for large-AOI assessments.
Processes a batch of 1024x1024 processing units on a single source GeoTIFF:
- No database connections (isolated runner contract).
- Never reads whole rasters into memory (windowed reads only).
- Enforces strict model weight verification and digest pinning.
- Computes valid_pixel_fraction per building mask; discards nodata buildings.
- Handles mask chunking and GPU OOM fallbacks.
- Writes outputs/evidence.jsonl and outputs/region_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import torch
from shapely.geometry import shape
from shapely.wkt import loads as wkt_loads

from hastegeo.core.utils.crop_service import _reproject_geometry
from hastegeo.core.utils.dtf_preprocess import PREPROCESSING_VERSION
from hastegeo.core.utils.logs import Logger as HasteLogger
from hastegeo.workflows.damage_triage_infer import (
    DTF_IM_SIZE,
    _find_local_checkpoint,
    MIN_CROP_FOOTPRINT_PX,
    build_dtf_model,
    compute_model_digest,
    crop_box_from_mask,
    read_tile,
    resolve_crops_per_forward,
    resolve_device,
    score_crops,
    score_window,
)

WORKDIR = os.getenv("WORKDIR", ".")
LOG_DIR = os.path.join(WORKDIR, "logs")
LOG_FILE = "damage_triage_region_infer.log"
os.makedirs(LOG_DIR, exist_ok=True)

logger = HasteLogger.get_logger(
    "damage_triage_region_infer", log_dir=LOG_DIR, log_file=LOG_FILE
)

# Used only when the batch config carries no min_valid_pixel_fraction
# (HASTE_REGION_MIN_VALID_FRACTION's documented default).
DEFAULT_MIN_VALID_FRACTION = 0.5


def log_progress(message: str) -> None:
    """Write timestamped progress update."""
    log_file = os.path.join(LOG_DIR, "region_infer_progress.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}|{message}\n")


def parse_geometry(geom_val: Any):
    """Parse geometry from Shapely, GeoJSON dict, or WKT string."""
    if hasattr(geom_val, "__geo_interface__"):
        return geom_val
    if isinstance(geom_val, dict):
        return shape(geom_val)
    if isinstance(geom_val, str):
        geom_val = geom_val.strip()
        if geom_val.startswith("{"):
            return shape(json.loads(geom_val))
        return wkt_loads(geom_val)
    raise ValueError(f"Unrecognized geometry format: {type(geom_val)}")


def run_batch_inference(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute inference for all processing units in this batch.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Dict[str, Any]: Region manifest dictionary.
    """
    output_dir = os.path.join(WORKDIR, config.get("output_dir", "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    batch_id = config.get("batch_id", "")
    assessment_id = config.get("assessment_id", "")
    imagery_id = config.get("imagery_id", "")
    # The runner downloads the TIF to <WORKDIR>/<files.imagery>.
    image_path = config.get("image_path") or os.path.join(
        WORKDIR, config.get("files", {}).get("imagery", "")
    )
    units = config.get("units", [])
    normalization = config.get("normalization") or "imagenet"
    masks_per_forward = config.get("masks_per_forward")
    min_valid_fraction = config.get("min_valid_pixel_fraction")
    if min_valid_fraction is None:
        min_valid_fraction = DEFAULT_MIN_VALID_FRACTION
    min_valid_fraction = float(min_valid_fraction)

    checkpoint_path = config.get("checkpoint_path") or os.getenv("DTF_CHECKPOINT_PATH", "")
    config_path = config.get("checkpoint_config_path") or os.getenv(
        "DTF_CHECKPOINT_CONFIG_PATH", ""
    )
    if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
        checkpoint_path, config_path = _find_local_checkpoint()
    expected_digest = config.get("expected_model_digest")

    if not checkpoint_path or not config_path:
        raise FileNotFoundError(
            "DamageTriageFormer checkpoint not found: set DTF_CHECKPOINT_PATH / "
            "DTF_CHECKPOINT_CONFIG_PATH or run download_models.py"
        )

    # Version pinning check
    actual_digest = compute_model_digest(checkpoint_path)
    if expected_digest and actual_digest != expected_digest:
        raise ValueError(
            f"Model digest mismatch! Expected {expected_digest}, got {actual_digest}"
        )

    device = resolve_device()
    logger.info("Loading DTF model on %s with strict region validation", device)
    model, use_gated = build_dtf_model(
        checkpoint_path,
        config_path,
        device,
        strict_region_check=True,
        allow_crop_mode=True,
    )
    crop_mode = getattr(model, "crop_mode", False) is True
    crops_per_forward = None
    if crop_mode:
        # Sized from the memory that is free right now; masks_per_forward is
        # the upper bound. A crop-mode checkpoint is trained on ImageNet-
        # normalised input, whatever the config asks for.
        crops_per_forward = resolve_crops_per_forward(device, masks_per_forward)
        if normalization != "imagenet":
            logger.warning(
                "normalization=%r ignored: the crop-mode checkpoint needs imagenet",
                normalization,
            )
    # Units prepared ahead on CPU threads while the GPU scores the current one.
    prefetch = max(1, int(os.getenv("HASTE_REGION_PREFETCH_UNITS", "2")))
    log_progress(
        f"Scoring on {device}: "
        + (
            f"crop mode, {crops_per_forward} crops per forward"
            if crop_mode
            else f"mask mode, {masks_per_forward or 'all'} masks per forward"
        )
        + f", {prefetch} unit(s) prefetched"
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source raster not found at: {image_path}")

    evidence_file = os.path.join(output_dir, "evidence.jsonl")
    unit_summaries = []
    total_evidence_count = 0
    completed_units = 0
    failed_units = 0

    local = threading.local()
    handles = []

    def _raster():
        # One dataset handle per worker thread: rasterio handles are not
        # safe to share across threads.
        ds = getattr(local, "ds", None)
        if ds is None:
            ds = local.ds = rasterio.open(image_path)
            handles.append(ds)
        return ds

    def _prepare_unit(u):
        """CPU-side work for one unit: windowed read, valid-pixel gate, and
        per-building masks (mask mode) or crop boxes (crop mode)."""
        raster = _raster()
        col_off = int(u["col_off"])
        row_off = int(u["row_off"])
        w = int(u["width"])
        h = int(u["height"])
        unit_id = u["unit_id"]

        win = rasterio.windows.Window(col_off, row_off, w, h)
        # Windowed raster read only (no full-image load!)
        tile = read_tile(raster, win, tile_size=DTF_IM_SIZE)
        tile_transform = rasterio.windows.transform(win, raster.transform)

        # Valid pixels: inside the real (unpadded) window, and not nodata in
        # all three bands. Without a declared nodata value, all-zero RGB
        # (black collar at imagery edges) counts as nodata - the same rule
        # utils/aoi.py uses for the valid-area mask.
        nodata_val = raster.nodata
        invalid_value = nodata_val if nodata_val is not None else 0
        valid_tile_mask = ~np.all(tile == invalid_value, axis=0)
        valid_tile_mask[h:, :] = False
        valid_tile_mask[:, w:] = False

        items = []  # masks (mask mode) or boxes (crop mode)
        bldg_meta = []
        for b in u.get("buildings", []):
            b_id = b["building_id"]
            cov_frac = float(b.get("coverage_fraction", 1.0))
            # Footprints are stored in EPSG:4326; rasterize in the raster's
            # own CRS (e.g. UTM for NAIP).
            geom = _reproject_geometry(parse_geometry(b["geometry"]), raster.crs)

            # Binary instance mask; all_touched matches the whole-layer and
            # AOI workflows' rasterization.
            mask_bool = rasterio.features.geometry_mask(
                [geom],
                out_shape=(DTF_IM_SIZE, DTF_IM_SIZE),
                transform=tile_transform,
                all_touched=True,
                invert=True,
            )
            mask_sum = int(mask_bool.sum())
            if mask_sum == 0 or (crop_mode and mask_sum < MIN_CROP_FOOTPRINT_PX):
                continue

            valid_frac = float((mask_bool & valid_tile_mask).sum() / mask_sum)
            if valid_frac < min_valid_fraction:
                logger.info(
                    "Building %s in unit %s has valid fraction %.2f < %.2f; skipping evidence",
                    b_id,
                    unit_id,
                    valid_frac,
                    min_valid_fraction,
                )
                continue

            if crop_mode:
                box = crop_box_from_mask(mask_bool, w, h)
                if box is None:
                    continue
                items.append(box)
            else:
                items.append(mask_bool.astype(np.float32))
            bldg_meta.append({
                "building_id": b_id,
                "coverage_fraction": cov_frac,
                "valid_pixel_fraction": valid_frac,
            })
        return tile, items, bldg_meta

    try:
        with ThreadPoolExecutor(max_workers=prefetch) as pool, open(
            evidence_file, "w", encoding="utf-8"
        ) as ev_out:
            pending = deque()
            unit_iter = iter(enumerate(units))

            def _fill():
                while len(pending) < prefetch:
                    nxt = next(unit_iter, None)
                    if nxt is None:
                        return
                    idx, unit = nxt
                    future = (
                        pool.submit(_prepare_unit, unit)
                        if unit.get("buildings")
                        else None
                    )
                    pending.append((idx, unit, future))

            _fill()
            while pending:
                u_idx, u, future = pending.popleft()
                _fill()  # keep the next units preparing while this one scores
                unit_id = u["unit_id"]
                u_summary: Dict[str, Any] = {
                    "unit_id": unit_id,
                    "status": "pending",
                    "buildings_scored": 0,
                    "error": None,
                }

                if future is None:
                    u_summary["status"] = "completed"
                    unit_summaries.append(u_summary)
                    completed_units += 1
                    continue

                try:
                    tile, items, bldg_meta = future.result()
                    if not items:
                        u_summary["status"] = "completed"
                        unit_summaries.append(u_summary)
                        completed_units += 1
                        continue

                    # Model scoring with batching and OOM fallback
                    if crop_mode:
                        probs = score_crops(
                            model, tile, items, device, crops_per_forward
                        )
                    else:
                        probs = score_window(
                            model=model,
                            use_gated=use_gated,
                            tile=tile,
                            masks=np.stack(items, axis=0),
                            device=device,
                            normalization=normalization,
                            masks_per_forward=masks_per_forward,
                        )

                    classes = probs.argmax(axis=1)
                    confs = probs.max(axis=1)

                    # Write evidence lines
                    for k, meta in enumerate(bldg_meta):
                        ev_record = {
                            "unit_id": unit_id,
                            "building_id": meta["building_id"],
                            "probs": probs[k].tolist(),
                            "class": int(classes[k]),
                            "confidence": float(confs[k]),
                            "coverage_fraction": meta["coverage_fraction"],
                            "valid_pixel_fraction": meta["valid_pixel_fraction"],
                        }
                        ev_out.write(json.dumps(ev_record) + "\n")
                        total_evidence_count += 1

                    u_summary["status"] = "completed"
                    u_summary["buildings_scored"] = len(bldg_meta)
                    completed_units += 1

                except Exception as exc:
                    logger.error("Error processing unit %s: %s", unit_id, exc, exc_info=True)
                    u_summary["status"] = "failed"
                    u_summary["error"] = str(exc)
                    failed_units += 1

                unit_summaries.append(u_summary)
                log_progress(f"Processed unit {u_idx + 1}/{len(units)}: {u_summary['status']}")
    finally:
        for ds in handles:
            ds.close()

    manifest = {
        "batch_id": batch_id,
        "assessment_id": assessment_id,
        "imagery_id": imagery_id,
        "model_digest": actual_digest,
        "preprocessing_version": PREPROCESSING_VERSION,
        "total_units": len(units),
        "completed_units": completed_units,
        "failed_units": failed_units,
        "evidence_count": total_evidence_count,
        "units": unit_summaries,
    }

    manifest_path = os.path.join(output_dir, "region_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Batch inference complete: %d evidence rows written", total_evidence_count)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to batch config JSON"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("Batch config file not found at: %s", args.config)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        run_batch_inference(config)
    except Exception as e:
        logger.error("Batch inference failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
