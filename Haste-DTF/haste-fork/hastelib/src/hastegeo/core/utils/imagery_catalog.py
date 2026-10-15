# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Multi-TIF imagery catalogue (``hg_imagery``) for large-AOI assessments.

Two steps, deliberately separated because they run in different places:

* ``describe_raster`` reads a GeoTIFF's header (never its pixels) and
  returns the metadata the catalogue needs. It runs wherever the file is
  local — inside the imageryprep task container, which then writes the
  result into ``imagery_manifest.json``.
* ``register_imagery_record`` inserts that metadata into PostGIS. It runs
  in the queue worker (``ImageryPostprocessor``), which has database
  credentials and knows the uploaded COG's storage URL. The task container
  has neither.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import pyproj
import rasterio
from shapely.geometry import box, mapping
from shapely.ops import transform

from .dtf_constants import STORAGE_CRS
from .postgis_connect import get_connection

logger = logging.getLogger(__name__)


def describe_raster(local_path: str) -> Dict[str, Any]:
    """Header-only description of a GeoTIFF for the imagery catalogue.

    Returns:
        ``{crs, transform: [a, b, c, d, e, f], width, height, pixel_size_x,
        pixel_size_y, band_count, dtype, nodata, footprint}`` where
        ``footprint`` is the raster's extent as a GeoJSON Polygon in
        EPSG:4326.
    """
    with rasterio.open(local_path) as src:
        bounds = src.bounds
        crs = src.crs
        affine = src.transform
        meta = {
            "crs": crs.to_string() if crs else STORAGE_CRS,
            "transform": [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f],
            "width": src.width,
            "height": src.height,
            "pixel_size_x": abs(affine.a),
            "pixel_size_y": abs(affine.e),
            "band_count": src.count,
            "dtype": src.dtypes[0] if src.dtypes else "uint8",
            "nodata": src.nodata,
        }

    poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    if crs and crs.to_epsg() != 4326:
        to_storage = pyproj.Transformer.from_crs(crs, STORAGE_CRS, always_xy=True)
        # densify so a reprojected edge follows the true (curved) boundary
        poly = transform(to_storage.transform, poly.segmentize(
            max(bounds.right - bounds.left, bounds.top - bounds.bottom) / 16
        ))
    meta["footprint"] = mapping(poly)
    return meta


def imagery_id_for(image_layer_id: str, file_uri: str) -> str:
    """Stable catalogue id: same layer + same file always maps to one row."""
    digest = hashlib.sha1(f"{image_layer_id}|{file_uri}".encode("utf-8")).hexdigest()
    return f"img_{digest[:20]}"


def register_imagery_record(
    image_layer_id: str,
    project_id: str,
    file_uri: str,
    meta: Dict[str, Any],
    conn=None,
    event_id: Optional[str] = None,
    role: str = "post",
    source_version: str = "1.0",
) -> Dict[str, Any]:
    """Insert or update one catalogue row from ``describe_raster`` output."""
    imagery_id = imagery_id_for(image_layer_id, file_uri)

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO hg_imagery (
                    imagery_id, project_id, image_layer_id, file_uri,
                    footprint, crs, transform, width, height,
                    pixel_size_x, pixel_size_y, band_count, dtype, nodata,
                    event_id, role, source_version, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, NOW()
                )
                ON CONFLICT (image_layer_id, file_uri) DO UPDATE
                SET footprint = EXCLUDED.footprint,
                    crs = EXCLUDED.crs,
                    transform = EXCLUDED.transform,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    pixel_size_x = EXCLUDED.pixel_size_x,
                    pixel_size_y = EXCLUDED.pixel_size_y,
                    band_count = EXCLUDED.band_count,
                    dtype = EXCLUDED.dtype,
                    nodata = EXCLUDED.nodata,
                    event_id = EXCLUDED.event_id,
                    role = EXCLUDED.role,
                    source_version = EXCLUDED.source_version
                RETURNING imagery_id
                """,
                (
                    imagery_id,
                    project_id,
                    image_layer_id,
                    file_uri,
                    json.dumps(meta["footprint"]),
                    meta["crs"],
                    list(meta["transform"]),
                    meta["width"],
                    meta["height"],
                    meta["pixel_size_x"],
                    meta["pixel_size_y"],
                    meta["band_count"],
                    meta["dtype"],
                    meta["nodata"],
                    event_id,
                    role,
                    source_version,
                ),
            )
            saved_id = cursor.fetchone()[0]
        connection.commit()
    finally:
        if owns_conn:
            connection.close()

    return {
        "imagery_id": saved_id,
        "project_id": project_id,
        "image_layer_id": image_layer_id,
        "file_uri": file_uri,
        "event_id": event_id,
        "role": role,
        "source_version": source_version,
        **meta,
    }


def register_imagery(
    image_layer: Any,
    file_uri: str,
    local_path: str,
    conn=None,
    project_id: Optional[str] = None,
    event_id: Optional[str] = None,
    role: str = "post",
    source_version: str = "1.0",
) -> Dict[str, Any]:
    """Describe a local GeoTIFF and register it in one call.

    Convenience for callers that have both the file and a DB connection.
    ``image_layer`` is an ImageLayer (or anything with ``imageLayerId`` /
    ``projectId``) or a plain image layer id string.
    """
    image_layer_id = getattr(image_layer, "imageLayerId", str(image_layer))
    eff_project_id = project_id or getattr(image_layer, "projectId", "")
    return register_imagery_record(
        image_layer_id=image_layer_id,
        project_id=eff_project_id,
        file_uri=file_uri,
        meta=describe_raster(local_path),
        conn=conn,
        event_id=event_id,
        role=role,
        source_version=source_version,
    )


def imagery_for_aoi(
    image_layer_id: str,
    aoi_geojson: Dict[str, Any],
    conn=None,
) -> List[Dict[str, Any]]:
    """All catalogued TIFs of an image layer whose footprint intersects the AOI.

    ``ST_Intersects`` uses the GIST index on ``footprint`` for the bounding-
    box shortlist, then tests exact geometry.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    imagery_id, project_id, image_layer_id, file_uri,
                    ST_AsGeoJSON(footprint), crs, width, height,
                    pixel_size_x, pixel_size_y, band_count, dtype, nodata,
                    role, source_version, transform
                FROM hg_imagery
                WHERE image_layer_id = %s
                  AND ST_Intersects(
                      footprint,
                      ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                  )
                ORDER BY imagery_id
                """,
                (image_layer_id, json.dumps(aoi_geojson)),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    return [
        {
            "imagery_id": r[0],
            "project_id": r[1],
            "image_layer_id": r[2],
            "file_uri": r[3],
            "footprint": json.loads(r[4]),
            "crs": r[5],
            "width": r[6],
            "height": r[7],
            "pixel_size_x": float(r[8]),
            "pixel_size_y": float(r[9]),
            "band_count": r[10],
            "dtype": r[11],
            "nodata": r[12],
            "role": r[13],
            "source_version": r[14],
            "transform": list(r[15]) if r[15] is not None else None,
        }
        for r in rows
    ]
