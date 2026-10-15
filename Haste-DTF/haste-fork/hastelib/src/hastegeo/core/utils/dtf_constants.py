# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DamageTriageFormer constants and helpers that need no deep-learning stack.

``hastegeo.workflows.damage_triage_infer`` imports torch and geopandas at
module load, and only the ``haste-training`` image has torch installed. The
Azure Functions images (``hastefuncapi``, ``hastefuncqueues``) do not. Code
that runs in those images (the large-AOI orchestrator, window selection,
aggregation, the HTTP API) imports from here instead.
``damage_triage_infer`` re-exports every name below, so existing imports of
these names from that module keep working.
"""

from __future__ import annotations

import hashlib
import os

import rasterio.windows

# Must match Damage-TriageFormer/src/config.py CATEGORIES exactly
# (checkpoint class-index order).
DTF_CATEGORIES = [
    "Undamaged",
    "Partial Roof Damage",
    "Total Roof Damage",
    "Partial Structural Collapse",
    "Total Structural Collapse",
]
DTF_IM_SIZE = 1024
UNCLASSIFIED = -1

# Hugging Face repo the released checkpoint comes from. Overridable so a
# retrained checkpoint can be recorded against its own source.
DTF_HF_MODEL_ID = os.getenv("DTF_HF_MODEL_ID", "Ymx1025/DamageTriageFormer-model")

# CRS conventions shared by hg_buildings, hg_imagery and hg_assessments.
STORAGE_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"


def compute_tile_windows(raster_width: int, raster_height: int, tile_size: int):
    """Non-overlapping tile windows covering the raster, row-major order."""
    windows = []
    for row_off in range(0, raster_height, tile_size):
        h = min(tile_size, raster_height - row_off)
        for col_off in range(0, raster_width, tile_size):
            w = min(tile_size, raster_width - col_off)
            windows.append(rasterio.windows.Window(col_off, row_off, w, h))
    return windows


def compute_model_digest(checkpoint_path: str) -> str:
    """sha256 of a checkpoint file, read in chunks."""
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
