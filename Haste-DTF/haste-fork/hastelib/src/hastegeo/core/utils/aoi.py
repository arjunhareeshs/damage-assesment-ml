# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Helpers for deriving an area-of-interest polygon from a GeoTIFF.

Used by the imageryprep workflow to produce the bounding polygon that the
Overture Maps building-footprint download is filtered by. Originally lived
as the standalone script ``docker/training/code/extract_data_mask_from_geotiff.py``
in the inference workflow; lifted here so the imageryprep workflow can
generate footprints once per image layer.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
import shapely.geometry.base

from .gdal_security import harden_gdal

# Harden GDAL drivers before any rasterio read (GDAL CVE compensating
# control — docs/known-vulnerabilities.md Root Cause C).
harden_gdal()

logger = logging.getLogger(__name__)


def extract_aoi_polygon(
    cog_path: str,
    *,
    simplify_tolerance_meters: float = 10.0,
    min_region_fraction: float = 0.01,
) -> shapely.geometry.base.BaseGeometry:
    """Extract the valid-data area of a GeoTIFF as a polygon, in EPSG:4326.

    The valid-data mask is built from pixels that are non-zero in any band
    (matching the original ``extract_data_mask_from_geotiff.py`` semantics).

    Every disconnected valid region is kept, so a layer built from imagery
    tiles that don't touch (two NAIP tiles kilometres apart, say) covers all
    of them. Only specks smaller than ``min_region_fraction`` of the largest
    region are dropped as noise. This used to keep just the largest region,
    which silently threw away every other tile -- and with it its buildings.

    Each region's exterior ring is simplified by approximately
    ``simplify_tolerance_meters`` meters and its interior holes are dropped.

    Args:
        cog_path: Path to a readable GeoTIFF (typically the post-event mosaic).
        simplify_tolerance_meters: Douglas-Peucker tolerance applied to the
            exterior rings before reprojection. Defaults to 10 m.
        min_region_fraction: Regions smaller than this fraction of the
            largest region's area are dropped as noise. Defaults to 1%.

    Returns:
        A ``Polygon`` (one region) or ``MultiPolygon`` (several), in
        EPSG:4326.

    Raises:
        ValueError: If no valid-data region can be derived.
    """
    import shapely.ops

    with rasterio.open(cog_path) as f:
        data = f.read()
        transform = f.transform
        crs = f.crs.to_string()

    num_channels, _, _ = data.shape
    mask = ((data == 0).sum(axis=0) != num_channels).astype(np.uint8)

    regions = [
        shapely.geometry.shape(geom)
        for geom, val in rasterio.features.shapes(mask, transform=transform)
        if val == 1
    ]
    if not regions:
        raise ValueError(
            f"No valid-data regions found in {cog_path} (mask is all zero)."
        )

    largest = max(r.area for r in regions)
    kept = [r for r in regions if r.area >= largest * min_region_fraction]
    if len(regions) > 1:
        logger.info(
            "Found %d valid-data regions in %s; keeping %d (dropped %d "
            "specks under %.1f%% of the largest region).",
            len(regions),
            cog_path,
            len(kept),
            len(regions) - len(kept),
            min_region_fraction * 100,
        )

    tolerance = (
        simplify_tolerance_meters / 111_139.0  # 1 degree lat ~ 111,139 m
        if crs == "EPSG:4326"
        else simplify_tolerance_meters
    )
    # Drop holes -- we only want each region's outer footprint.
    outlines = [
        shapely.geometry.Polygon(list(r.exterior.coords)).simplify(tolerance)
        for r in kept
    ]
    merged = shapely.ops.unary_union(outlines)

    if crs == "EPSG:4326":
        return merged
    import fiona.transform

    geom_4326 = fiona.transform.transform_geom(
        crs, "EPSG:4326", shapely.geometry.mapping(merged)
    )
    return shapely.geometry.shape(geom_4326)


def aoi_bbox_from_cog(
    cog_path: str,
) -> Tuple[float, float, float, float]:
    """Return the EPSG:4326 bounding box of the AOI polygon as (xmin, ymin, xmax, ymax)."""
    polygon = extract_aoi_polygon(cog_path)
    return polygon.bounds


def save_polygon_as_geojson(
    polygon: shapely.geometry.base.BaseGeometry,
    output_path: str,
    *,
    properties: Optional[dict] = None,
) -> str:
    """Write a shapely geometry to disk as a single-feature GeoJSON.

    The output is a FeatureCollection with one Feature whose geometry is
    ``polygon`` (assumed already in EPSG:4326) and whose properties are
    ``properties`` (defaults to ``{}``). Used by the imageryprep workflow
    to persist the AOI polygon so the UI can offer it as a downloadable
    "valid area mask" artifact.

    Args:
        polygon: A shapely geometry in EPSG:4326.
        output_path: Where to write the .geojson file.
        properties: Optional dict of feature properties. Defaults to ``{}``.

    Returns:
        ``output_path`` for caller convenience.
    """
    feature = {
        "type": "Feature",
        "geometry": shapely.geometry.mapping(polygon),
        "properties": properties or {},
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    with open(output_path, "w") as f:
        json.dump(fc, f)
    return output_path
