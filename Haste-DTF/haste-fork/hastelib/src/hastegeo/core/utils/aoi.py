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
import os
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


# Longest side, in pixels, of the mask the valid area is traced from. The
# mosaic is read decimated to this size, so memory stays bounded (about
# max_dim^2 bytes per band) whatever the mosaic's size. 8192 px keeps the
# outline within one source pixel for mosaics up to 8192 px and within a few
# metres for statewide mosaics, well under the 10 m simplification.
DEFAULT_MASK_MAX_DIM = int(os.getenv("HASTE_AOI_MASK_MAX_DIM", "8192"))
# Interior no-data gaps at least this large (m^2) are kept as holes, so
# buildings inside a gap are not treated as covered by imagery. Smaller
# holes are speckle (dark roofs, shadows) and are filled.
DEFAULT_MIN_HOLE_AREA_M2 = 2500.0
_M_PER_DEG = 111_320.0


def valid_data_mask(src, max_dim: int = DEFAULT_MASK_MAX_DIM):
    """Boolean valid-data mask of an open raster, read decimated so its
    longest side is at most ``max_dim``. Returns ``(mask, transform)``.

    A pixel is valid unless it is the declared nodata value, transparent
    (alpha 0), or -- for a file that declares neither -- 0 in every band
    (the black collar convention ``mosaic_imagery`` also uses).
    """
    from rasterio.enums import ColorInterp, Resampling

    scale = max(1.0, max(src.width, src.height) / float(max_dim))
    out_h = max(1, int(round(src.height / scale)))
    out_w = max(1, int(round(src.width / scale)))
    has_alpha = any(ci == ColorInterp.alpha for ci in src.colorinterp)
    if src.nodata is not None or has_alpha:
        mask = src.dataset_mask(
            out_shape=(out_h, out_w), resampling=Resampling.nearest
        ) > 0
    else:
        mask = np.zeros((out_h, out_w), dtype=bool)
        for band in range(1, src.count + 1):  # one band at a time
            mask |= src.read(
                band, out_shape=(out_h, out_w), resampling=Resampling.nearest
            ) != 0
    transform = src.transform * src.transform.scale(
        src.width / out_w, src.height / out_h
    )
    return mask, transform


def _metres_to_crs_units(crs, metres: float, lat: float) -> Tuple[float, float]:
    """(length, area-per-m^2) factors: a length in metres expressed in CRS
    units, and the CRS-unit area of one m^2 (geographic CRS: at ``lat``)."""
    if crs is not None and crs.is_geographic:
        deg = metres / _M_PER_DEG
        m2 = 1.0 / (_M_PER_DEG * _M_PER_DEG * max(0.01, np.cos(np.radians(lat))))
        return deg, m2
    unit = 1.0
    try:
        unit = float(crs.linear_units_factor[1]) if crs is not None else 1.0
    except Exception:  # pragma: no cover - CRS without linear units
        unit = 1.0
    return metres / unit, 1.0 / (unit * unit)


def extract_aoi_polygon(
    cog_path: str,
    *,
    simplify_tolerance_meters: float = 10.0,
    min_region_fraction: float = 0.01,
    min_hole_area_m2: float = DEFAULT_MIN_HOLE_AREA_M2,
    max_dim: int = DEFAULT_MASK_MAX_DIM,
) -> shapely.geometry.base.BaseGeometry:
    """Extract the valid-data area of a GeoTIFF as a polygon, in EPSG:4326.

    * Memory is bounded: the raster is read decimated (``valid_data_mask``),
      one band at a time, never in full.
    * Validity honours the raster's nodata value and alpha band; only a
      file that declares neither falls back to "non-zero in any band".
    * Every disconnected region is kept, except specks smaller than
      ``min_region_fraction`` of the largest region.
    * Interior no-data gaps of at least ``min_hole_area_m2`` stay as holes,
      so buildings inside them are not treated as imaged; smaller holes
      (dark roofs, shadows read as 0) are filled.
    * Outlines are simplified by about ``simplify_tolerance_meters``.

    Raises:
        ValueError: If no valid-data region can be derived.
    """
    import shapely.ops
    from pyproj import Transformer

    with rasterio.open(cog_path) as f:
        mask, transform = valid_data_mask(f, max_dim)
        crs = f.crs
        centre_lat = 0.0
        if crs is not None:
            b = f.bounds
            if crs.is_geographic:
                centre_lat = (b.bottom + b.top) / 2.0
            else:
                lon_lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                centre_lat = lon_lat.transform(
                    (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
                )[1]

    regions = [
        shapely.geometry.shape(geom)
        for geom, val in rasterio.features.shapes(
            mask.astype(np.uint8), mask=mask, transform=transform
        )
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
            len(regions), cog_path, len(kept), len(regions) - len(kept),
            min_region_fraction * 100,
        )

    tolerance, area_per_m2 = _metres_to_crs_units(
        crs, simplify_tolerance_meters, centre_lat
    )
    min_hole = min_hole_area_m2 * area_per_m2
    outlines = []
    n_holes = 0
    for r in kept:
        holes = [
            ring.coords
            for ring in r.interiors
            if shapely.geometry.Polygon(ring).area >= min_hole
        ]
        n_holes += len(holes)
        outlines.append(
            shapely.geometry.Polygon(r.exterior.coords, holes).simplify(
                tolerance, preserve_topology=True
            )
        )
    if n_holes:
        logger.info("Kept %d interior no-data gap(s) as holes", n_holes)
    merged = shapely.ops.unary_union(outlines)

    if crs is None or crs.to_epsg() == 4326:
        return merged
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    # Densify first so reprojected edges follow the true boundary.
    return shapely.ops.transform(to_wgs84, merged.segmentize(tolerance * 4))


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
