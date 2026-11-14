# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Read-only spatial analysis of an AOI before any inference is queued.

Works from catalogue metadata only (``hg_imagery`` footprints and
geotransforms), so no GeoTIFF is opened or read. Reports imagery coverage,
fully/partially covered tiles, the uncovered area, the expected number of
processing windows and an overall readiness state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pyproj
import rasterio
import rasterio.crs
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

from .dtf_constants import AREA_CRS, DTF_IM_SIZE, STORAGE_CRS

# Readiness states reported to the UI.
READY = "ready"
INCOMPLETE_IMAGERY = "incomplete_imagery"
NO_BUILDINGS = "no_buildings"
NO_IMAGERY = "no_imagery"

# Coverage above this counts as complete (guards float noise from unions).
FULL_COVERAGE_PCT = 99.99

_TO_AREA = pyproj.Transformer.from_crs(STORAGE_CRS, AREA_CRS, always_xy=True)


def _area_km2(geom: Any) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return transform(_TO_AREA.transform, geom).area / 1_000_000.0


def estimate_windows(
    aoi_geom: Any,
    imagery: Dict[str, Any],
    tile_size: int = DTF_IM_SIZE,
) -> int:
    """Upper bound on grid windows of one raster that overlap the AOI.

    Maps the AOI bounding box into the raster's pixel grid and counts the
    tile-aligned windows it spans, clipped to the raster. O(1) per raster;
    the planner later keeps only windows that actually hold buildings.
    """
    tf = imagery.get("transform")
    width, height = imagery.get("width"), imagery.get("height")
    if not tf or not width or not height:
        return 0

    affine = rasterio.Affine(*tf[:6])
    crs = imagery.get("crs")
    bounds = aoi_geom.bounds
    if crs:
        crs_obj = rasterio.crs.CRS.from_user_input(crs)
        if crs_obj.to_epsg() != 4326:
            to_raster = pyproj.Transformer.from_crs(
                STORAGE_CRS, crs_obj, always_xy=True
            )
            bounds = transform(to_raster.transform, box(*bounds)).bounds

    inv = ~affine
    corners = [
        inv * (bounds[0], bounds[1]),
        inv * (bounds[0], bounds[3]),
        inv * (bounds[2], bounds[1]),
        inv * (bounds[2], bounds[3]),
    ]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    c0 = max(0, int(math.floor(min(cols))))
    c1 = min(width, int(math.ceil(max(cols))))
    r0 = max(0, int(math.floor(min(rows))))
    r1 = min(height, int(math.ceil(max(rows))))
    if c1 <= c0 or r1 <= r0:
        return 0

    n_cols = (c1 - 1) // tile_size - c0 // tile_size + 1
    n_rows = (r1 - 1) // tile_size - r0 // tile_size + 1
    return n_cols * n_rows


def analyze_aoi(
    aoi_geom: Any,
    area_km2: float,
    imagery: List[Dict[str, Any]],
    building_count: int,
    max_units: Optional[int] = None,
    tile_size: int = DTF_IM_SIZE,
) -> Dict[str, Any]:
    """Summarise coverage and workload for a validated AOI.

    Args:
        aoi_geom: Validated shapely AOI in EPSG:4326.
        area_km2: AOI area, already computed by ``validate_aoi``.
        imagery: ``imagery_for_aoi`` rows (footprint GeoJSON in EPSG:4326).
        building_count: Buildings intersecting the AOI.
        max_units: Optional cap the planner will enforce.
        tile_size: Model window size in pixels.
    """
    footprints = [shape(i["footprint"]).buffer(0) for i in imagery]
    tiles = []
    for img, fp in zip(imagery, footprints):
        tiles.append(
            {
                "imageryId": img["imagery_id"],
                "coverage": "full" if aoi_geom.within(fp) else "partial",
                "fileUri": img.get("file_uri"),
                "crs": img.get("crs"),
                "pixelSize": [img.get("pixel_size_x"), img.get("pixel_size_y")],
                "acquired": img.get("source_version"),
            }
        )

    covered = unary_union(footprints).intersection(aoi_geom) if footprints else None
    covered_km2 = _area_km2(covered) if covered is not None else 0.0
    coverage_pct = (
        min(100.0, 100.0 * covered_km2 / area_km2) if area_km2 > 0 else 0.0
    )
    missing = aoi_geom.difference(covered) if covered is not None else aoi_geom
    missing_km2 = max(0.0, area_km2 - covered_km2)

    est_units = sum(estimate_windows(aoi_geom, i, tile_size) for i in imagery)

    if not imagery:
        state = NO_IMAGERY
    elif building_count == 0:
        state = NO_BUILDINGS
    elif coverage_pct < FULL_COVERAGE_PCT:
        state = INCOMPLETE_IMAGERY
    else:
        state = READY

    over_cap = max_units is not None and est_units > max_units
    return {
        "areaKm2": round(area_km2, 4),
        "imageryCount": len(imagery),
        "fullyCoveredTiles": sum(1 for t in tiles if t["coverage"] == "full"),
        "partiallyCoveredTiles": sum(1 for t in tiles if t["coverage"] == "partial"),
        "tiles": tiles,
        "coveragePercent": round(coverage_pct, 2),
        "coveredAreaKm2": round(covered_km2, 4),
        "missingAreaKm2": round(missing_km2, 4),
        "missingGeometry": (
            None
            if missing is None or missing.is_empty or missing_km2 <= 0
            else missing.__geo_interface__
        ),
        "buildingCount": building_count,
        "estimatedUnits": est_units,
        "exceedsMaxUnits": bool(over_cap),
        "maxUnits": max_units,
        "state": state,
        # Submission is allowed when something can be assessed; incomplete
        # coverage is surfaced, never silently treated as "undamaged".
        "canSubmit": state in (READY, INCOMPLETE_IMAGERY) and not over_cap,
    }
