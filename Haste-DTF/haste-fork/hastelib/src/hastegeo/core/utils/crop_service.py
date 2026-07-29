# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Per-building windowed crop extraction from a COG.

This is plan.pdf's Crop Extraction Service, and its own warning made
literal: *"It does NOT send Polygon to the model. It uses the polygon to
extract pixels."* ``crop_building`` is exactly that boundary — a
building's geometry is consumed here only to pick a pixel window (via
``rasterio.windows``, which only pulls the needed blocks of a
Cloud-Optimized GeoTIFF, never the whole raster); everything downstream
(``hastegeo.workflows.damage_triage_aoi_infer``) works with pixel arrays
plus one binary instance mask, never the vector geometry itself.

Runs only inside the training Docker image (see
``docker/training/env/env.yml`` — rasterio/geopandas/shapely are training-
image dependencies, not ``hastelib``'s own ``pyproject.toml`` deps, matching
how ``hastegeo.workflows.damage_triage_infer`` is already structured: it
lives in the ``hastegeo`` package tree for import convenience but only
executes inside the training container).
"""

from __future__ import annotations

import numpy as np
import rasterio  # type: ignore
import rasterio.windows  # type: ignore
from rasterio.features import geometry_mask  # type: ignore
from shapely.geometry import shape as shapely_shape  # type: ignore
from shapely.geometry.base import BaseGeometry


def _reproject_geometry(geom: BaseGeometry, dst_crs) -> BaseGeometry:
    """Reproject a shapely geometry from EPSG:4326 into ``dst_crs``."""
    import pyproj  # type: ignore
    from shapely.ops import transform as shapely_transform  # type: ignore

    if str(dst_crs).upper() in ("EPSG:4326", "4326"):
        return geom
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", dst_crs, always_xy=True
    ).transform
    return shapely_transform(project, geom)


def crop_building(
    raster: rasterio.io.DatasetReader,
    building_geom_4326: dict,
    pad_px: int = 32,
    crop_size: int = None,
) -> tuple[np.ndarray, "rasterio.Affine", BaseGeometry]:
    """Read a windowed RGB crop around one building's footprint.

    Args:
        raster: an already-open ``rasterio`` dataset for the post-event mosaic.
        building_geom_4326: GeoJSON geometry dict, EPSG:4326 — as returned
            by ``hastegeo.core.utils.postgis_query.buildings_in_aoi``.
        pad_px: context margin (in raster pixels) added on each side of the
            building's bounding box before reading. A bare zero-margin crop
            starves the model of the surrounding context DamageTriageFormer
            was trained on (1024x1024 tiles) — tune empirically against a
            held-out AOI before trusting AOI-path confidence scores as
            comparable to the whole-layer tile-batch path's scores. See
            ``docs/OptionB_AOI_Integration_Plan.md`` "Biggest pitfalls" #2.
        crop_size: if given, the returned array is zero-padded (or, for a
            building whose padded bbox exceeds ``crop_size``, truncated —
            acceptable for the common case of buildings much smaller than a
            1024x1024 tile) to exactly ``(3, crop_size, crop_size)``,
            matching the model's tile-resolution input contract
            (``DTF_IM_SIZE`` in ``damage_triage_infer.py``). If ``None``,
            the natural padded-bbox window size is returned unchanged.

    Returns:
        ``(crop_array (3, H, W), crop_transform, building_geom_raster_crs)``
        — the geometry is returned already reprojected into ``raster.crs``
        so the caller can rasterize this building's own instance mask
        against ``crop_transform`` without reprojecting twice.
    """
    geom_4326 = shapely_shape(building_geom_4326)
    geom_raster_crs = _reproject_geometry(geom_4326, raster.crs)

    minx, miny, maxx, maxy = geom_raster_crs.bounds
    row_start, col_start = raster.index(minx, maxy)
    row_stop, col_stop = raster.index(maxx, miny)

    row_start = max(row_start - pad_px, 0)
    col_start = max(col_start - pad_px, 0)
    row_stop = min(row_stop + pad_px, raster.height)
    col_stop = min(col_stop + pad_px, raster.width)

    height = max(row_stop - row_start, 1)
    width = max(col_stop - col_start, 1)
    if crop_size is not None:
        height = min(height, crop_size)
        width = min(width, crop_size)

    window = rasterio.windows.Window(col_start, row_start, width, height)
    data = raster.read([1, 2, 3], window=window)
    crop_transform = rasterio.windows.transform(window, raster.transform)

    if crop_size is not None:
        _, h, w = data.shape
        padded = np.zeros((3, crop_size, crop_size), dtype=data.dtype)
        padded[:, :h, :w] = data[:, :crop_size, :crop_size]
        data = padded

    return data, crop_transform, geom_raster_crs


def crop_building_mask(
    geom_raster_crs: BaseGeometry,
    crop_transform: "rasterio.Affine",
    crop_size: int,
) -> np.ndarray:
    """Binary instance mask for exactly one building, at crop resolution.

    Unlike the whole-layer workflow (``damage_triage_infer.py``'s
    ``rasterize_instance_masks``, which rasterizes every intersecting
    footprint into a shared tile), this rasterizes a single geometry —
    each per-building crop carries its own single-building mask.
    """
    return geometry_mask(
        [geom_raster_crs],
        out_shape=(crop_size, crop_size),
        transform=crop_transform,
        all_touched=True,
        invert=True,
    ).astype(np.float32)


def write_png(path: str, rgb_array: np.ndarray) -> str:
    """Write a ``(3, H, W)`` uint8 array as a PNG via GDAL's PNG driver.

    Uses ``rasterio``/GDAL (already a training-image dependency) rather
    than adding a Pillow/opencv dependency — ``docker/training/env/env.yml``
    does not list either.
    """
    _, h, w = rgb_array.shape
    with rasterio.open(
        path, "w", driver="PNG", height=h, width=w, count=3, dtype="uint8"
    ) as dst:
        dst.write(rgb_array.astype(np.uint8))
    return path
