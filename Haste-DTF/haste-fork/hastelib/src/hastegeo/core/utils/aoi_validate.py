# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validation and area calculation for analyst-drawn AOI polygons.

Strictly validates input geometries in EPSG:4326 and calculates true equal-area
surface area in km² using EPSG:6933. Rejects invalid or unsupported geometries
with distinct, informative error messages without attempting silent auto-repair.
"""

from typing import Any, Dict, Tuple, Union
import pyproj
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform

STORAGE_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"

# Transformer from WGS84 (lon, lat) to cylindrical equal-area (EPSG:6933)
_TRANSFORMER_4326_TO_6933 = pyproj.Transformer.from_crs(
    STORAGE_CRS, AREA_CRS, always_xy=True
)


def validate_aoi(
    geojson: Union[Dict[str, Any], Polygon, MultiPolygon],
    max_area_km2: float | None = None,
) -> Tuple[Union[Polygon, MultiPolygon], float]:
    """Validate an AOI GeoJSON geometry and calculate its area in km².

    Args:
        geojson: A GeoJSON geometry dict, Feature dict, or Shapely Polygon/MultiPolygon.
        max_area_km2: Optional maximum allowed area in km².

    Returns:
        Tuple[Union[Polygon, MultiPolygon], float]: Validated Shapely geometry in EPSG:4326
            and calculated area in km².

    Raises:
        ValueError: If geometry is empty, invalid, out-of-bounds, antimeridian-crossing,
            not a Polygon/MultiPolygon, or exceeds max_area_km2.
    """
    if geojson is None:
        raise ValueError("AOI geometry cannot be None or empty.")

    # Convert to Shapely geometry if dict
    if isinstance(geojson, dict):
        if "geometry" in geojson and isinstance(geojson["geometry"], dict):
            geom_dict = geojson["geometry"]
        elif "features" in geojson and isinstance(geojson["features"], list):
            if not geojson["features"]:
                raise ValueError("GeoJSON FeatureCollection contains no features.")
            geom_dict = geojson["features"][0].get("geometry", {})
        else:
            geom_dict = geojson

        try:
            geom = shape(geom_dict)
        except Exception as e:
            raise ValueError(f"Failed to parse GeoJSON geometry: {e}") from e
    elif isinstance(geojson, (Polygon, MultiPolygon)):
        geom = geojson
    elif hasattr(geojson, "geom_type"):
        raise ValueError(
            f"Unsupported geometry type: {geojson.geom_type}. "
            "AOI must be a Polygon or MultiPolygon."
        )
    else:
        raise ValueError(
            f"Unsupported input type for AOI: {type(geojson).__name__}. "
            "Must be GeoJSON dict or Shapely Polygon/MultiPolygon."
        )

    # 1. Type validation
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError(
            f"Unsupported geometry type: {geom.geom_type}. "
            "AOI must be a Polygon or MultiPolygon."
        )

    # 2. Emptiness check
    if geom.is_empty:
        raise ValueError("AOI geometry is empty.")

    # 3. Validity check (no self-intersection, etc. - DO NOT auto-repair)
    if not geom.is_valid:
        raise ValueError(
            "AOI geometry is topologically invalid (e.g. self-intersecting). "
            "Please provide a clean polygon."
        )

    # 4. Coordinate boundary checks (EPSG:4326 lat/lon ranges)
    minx, miny, maxx, maxy = geom.bounds
    if minx < -180.0 or maxx > 180.0 or miny < -90.0 or maxy > 90.0:
        raise ValueError(
            f"AOI coordinates out of WGS84 range: bounds ({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}). "
            "Longitude must be between -180 and 180, and latitude between -90 and 90."
        )

    # 5. Antimeridian crossing check
    # Check if any consecutive coordinate points in exterior rings cross the 180 degree boundary
    polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for p in polygons:
        coords = list(p.exterior.coords)
        for i in range(len(coords) - 1):
            lon1, _ = coords[i][:2]
            lon2, _ = coords[i + 1][:2]
            if abs(lon1 - lon2) > 180.0:
                raise ValueError(
                    "AOI crosses the 180th meridian (antimeridian). "
                    "Please split the AOI into separate polygons across the antimeridian."
                )

    # 6. Area calculation in equal-area projection (EPSG:6933)
    try:
        projected_geom = transform(_TRANSFORMER_4326_TO_6933.transform, geom)
        area_m2 = projected_geom.area
        area_km2 = area_m2 / 1_000_000.0
    except Exception as e:
        raise ValueError(
            f"Failed to calculate equal-area surface area for AOI: {e}"
        ) from e

    if area_km2 <= 0.0:
        raise ValueError("AOI geometry has zero or degenerate surface area.")

    # 7. Maximum area limit check
    if max_area_km2 is not None and area_km2 > max_area_km2:
        raise ValueError(
            f"AOI area ({area_km2:.2f} km²) exceeds the maximum allowable limit of {max_area_km2:.2f} km²."
        )

    return geom, area_km2
