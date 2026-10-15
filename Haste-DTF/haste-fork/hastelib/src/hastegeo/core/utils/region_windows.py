# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Window selection and batch grouping for large-AOI damage assessments.

Pure geospatial functions for:
1. Computing coverage fraction of building footprints inside tile windows.
2. Selecting grid and building-centred processing units for an AOI.
3. Grouping units by TIF into bounded runner batches.

Does not depend on PyTorch, GPU runtime, or database connections, so it can
run inside the Azure Functions images (see ``dtf_constants``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pyproj
import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.windows
from shapely import STRtree
from shapely.geometry import box, shape
from shapely.ops import transform

from .dtf_constants import DTF_IM_SIZE, STORAGE_CRS, compute_tile_windows

logger = logging.getLogger(__name__)

# A building counts as fully inside a window above this coverage; guards
# against floating-point area noise on reprojected geometries.
FULL_COVERAGE = 0.9999


def building_coverage(
    building_geom: Any,
    window_geom: Any,
) -> float:
    """Calculate the fraction of a building footprint contained within a window.

    Args:
        building_geom: Shapely geometry of the building footprint.
        window_geom: Shapely geometry of the tile window.

    Returns:
        float: Fraction in range [0.0, 1.0].
    """
    if building_geom is None or building_geom.is_empty:
        return 0.0
    if window_geom is None or window_geom.is_empty:
        return 0.0
    if not building_geom.intersects(window_geom):
        return 0.0

    b_area = building_geom.area
    if b_area <= 0:
        return 0.0

    inter = building_geom.intersection(window_geom)
    coverage = inter.area / b_area
    return float(min(1.0, max(0.0, coverage)))


def _transformers(raster_crs: Any):
    """(to_storage, to_raster) transform functions; identity when the
    raster is already in the storage CRS or no CRS is given."""
    if raster_crs is None:
        return None, None
    crs_obj = rasterio.crs.CRS.from_user_input(raster_crs)
    if crs_obj.to_epsg() == 4326:
        return None, None
    to_storage = pyproj.Transformer.from_crs(crs_obj, STORAGE_CRS, always_xy=True)
    to_raster = pyproj.Transformer.from_crs(STORAGE_CRS, crs_obj, always_xy=True)
    return to_storage, to_raster


def select_units(
    imagery_id: str,
    raster_width: int,
    raster_height: int,
    raster_transform: rasterio.Affine,
    aoi_geom: Any,
    eligible_buildings: List[Dict[str, Any]],
    tile_size: int = DTF_IM_SIZE,
    raster_crs: Any = None,
    assessment_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Select compute units for one source GeoTIFF covering the AOI.

    - Keep a grid window only if it intersects the AOI AND contains >= 1
      eligible building.
    - Track each building's best coverage across units.
    - If a building's best coverage is below 1.0 and its bounding box fits in
      ``tile_size`` px, add one building_centred unit for it.

    Args:
        imagery_id: Unique identifier of the source GeoTIFF.
        raster_width: Raster width in pixels.
        raster_height: Raster height in pixels.
        raster_transform: The raster's own affine transform (pixel -> raster
            CRS coordinates), as stored in ``hg_imagery.transform``.
        aoi_geom: Shapely geometry of the AOI in EPSG:4326.
        eligible_buildings: ``[{building_id, geometry}]`` with geometry in
            EPSG:4326 (shapely or GeoJSON dict). Callers with very large
            inputs should pre-filter to buildings intersecting this TIF.
        tile_size: Model input size in pixels.
        raster_crs: CRS of the raster. ``None`` means EPSG:4326.
        assessment_id: Optional assessment identifier for namespacing unit IDs.

    Returns:
        List[Dict[str, Any]]: Selected processing unit records. ``bounds``
        is a shapely geometry in EPSG:4326; ``buildings`` lists
        ``{building_id, coverage_fraction}``.
    """
    if aoi_geom is None or aoi_geom.is_empty:
        return []
    if not eligible_buildings:
        return []

    to_storage, to_raster = _transformers(raster_crs)

    def _to_storage(geom):
        return geom if to_storage is None else transform(to_storage.transform, geom)

    def _to_raster(geom):
        return geom if to_raster is None else transform(to_raster.transform, geom)

    r_bounds = rasterio.transform.array_bounds(
        raster_height, raster_width, raster_transform
    )
    r_footprint = _to_storage(box(*r_bounds))
    if not r_footprint.intersects(aoi_geom):
        return []

    # Buildings that intersect both this raster's footprint and the AOI.
    active_ids: List[str] = []
    active_geoms: List[Any] = []
    for b in eligible_buildings:
        b_geom = b["geometry"]
        if isinstance(b_geom, dict):
            b_geom = shape(b_geom)
        if b_geom.intersects(r_footprint) and b_geom.intersects(aoi_geom):
            active_ids.append(str(b["building_id"]))
            active_geoms.append(b_geom)

    if not active_ids:
        return []

    tree = STRtree(active_geoms)
    prefix = f"{assessment_id}_" if assessment_id else ""
    units: List[Dict[str, Any]] = []
    best_coverage: Dict[str, float] = {b_id: 0.0 for b_id in active_ids}

    # 1. Regular grid windows
    for win in compute_tile_windows(raster_width, raster_height, tile_size):
        col_off, row_off = int(win.col_off), int(win.row_off)
        win_geom = _to_storage(
            box(*rasterio.windows.bounds(win, raster_transform))
        )
        if not win_geom.intersects(aoi_geom):
            continue

        unit_buildings = []
        for idx in sorted(tree.query(win_geom, predicate="intersects")):
            b_id = active_ids[idx]
            cov = building_coverage(active_geoms[idx], win_geom)
            if cov > 0:
                unit_buildings.append(
                    {"building_id": b_id, "coverage_fraction": cov}
                )
                best_coverage[b_id] = max(best_coverage[b_id], cov)

        if not unit_buildings:
            continue

        units.append({
            "unit_id": f"unit_{prefix}{imagery_id}_{col_off}_{row_off}",
            "imagery_id": imagery_id,
            "col_off": col_off,
            "row_off": row_off,
            "width": int(win.width),
            "height": int(win.height),
            "bounds": win_geom,
            "kind": "grid",
            "buildings": unit_buildings,
        })

    # 2. Building-centred units for buildings no grid window holds whole
    inv_transform = ~raster_transform
    occupied = {(u["col_off"], u["row_off"]) for u in units}
    for b_id, b_geom in zip(active_ids, active_geoms):
        if best_coverage[b_id] >= FULL_COVERAGE:
            continue

        b_px = transform(
            lambda x, y: inv_transform * (x, y), _to_raster(b_geom)
        )
        min_c, min_r, max_c, max_r = b_px.bounds
        if (max_c - min_c) > tile_size or (max_r - min_r) > tile_size:
            logger.info(
                "Building %s is larger than one %d px window; scored from "
                "its best grid window only", b_id, tile_size,
            )
            continue

        c_off = int(round((min_c + max_c) / 2.0 - tile_size / 2.0))
        r_off = int(round((min_r + max_r) / 2.0 - tile_size / 2.0))
        c_off = max(0, min(c_off, max(0, raster_width - tile_size)))
        r_off = max(0, min(r_off, max(0, raster_height - tile_size)))
        if (c_off, r_off) in occupied:
            continue
        w = min(tile_size, raster_width - c_off)
        h = min(tile_size, raster_height - r_off)

        centred = rasterio.windows.Window(c_off, r_off, w, h)
        c_geom = _to_storage(
            box(*rasterio.windows.bounds(centred, raster_transform))
        )
        cov = building_coverage(b_geom, c_geom)
        if cov <= best_coverage[b_id]:
            continue

        units.append({
            "unit_id": f"unit_{prefix}{imagery_id}_b_{b_id}",
            "imagery_id": imagery_id,
            "col_off": c_off,
            "row_off": r_off,
            "width": w,
            "height": h,
            "bounds": c_geom,
            "kind": "building_centred",
            "buildings": [{"building_id": b_id, "coverage_fraction": cov}],
        })
        occupied.add((c_off, r_off))
        best_coverage[b_id] = cov

    return units


def group_batches(
    units: List[Dict[str, Any]],
    windows_per_batch: int,
    assessment_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Group units of the same TIF into batches of at most windows_per_batch each.

    Args:
        units: Processing units returned by select_units.
        windows_per_batch: Maximum number of windows in a single runner task.
        assessment_id: Optional assessment identifier for batch ID namespacing.

    Returns:
        List[Dict[str, Any]]: Batches with assigned batch_id and linked units.
    """
    if windows_per_batch <= 0:
        raise ValueError(f"windows_per_batch must be positive, got {windows_per_batch}")

    by_imagery: Dict[str, List[Dict[str, Any]]] = {}
    for u in units:
        by_imagery.setdefault(u["imagery_id"], []).append(u)

    batches: List[Dict[str, Any]] = []
    prefix = f"{assessment_id}_" if assessment_id else ""

    for img_id, img_units in by_imagery.items():
        for i in range(0, len(img_units), windows_per_batch):
            chunk = img_units[i : i + windows_per_batch]
            batch_id = f"batch_{prefix}{img_id}_{(i // windows_per_batch) + 1}"

            for u in chunk:
                u["batch_id"] = batch_id

            batches.append({
                "batch_id": batch_id,
                "assessment_id": assessment_id,
                "imagery_id": img_id,
                "status": "pending",
                "attempts": 0,
                "unit_ids": [u["unit_id"] for u in chunk],
                "units": chunk,
            })

    return batches
