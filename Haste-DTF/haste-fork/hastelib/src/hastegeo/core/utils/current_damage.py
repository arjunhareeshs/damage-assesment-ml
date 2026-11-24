# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Current building-level damage output.

``hg_buildings`` is the authoritative per-building "current" record: the
original footprint polygon plus the latest successful damage class. Every
consumer (HASTE UI, the Texas Mapbox app, the GeoParquet export) reads from
it, so there is one source of truth and no duplicate current rows.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .dtf_constants import DTF_CATEGORIES, DTF_CLASS_COLORS, DTF_UNKNOWN_COLOR
from .postgis_connect import get_connection

logger = logging.getLogger(__name__)

# Only these results may become the current output; failed, uncertain or
# insufficient-data results never overwrite a good earlier prediction.
PUBLISHABLE_STATUS = "completed"


_SCHEMA_READY = False


def ensure_schema(connection) -> None:
    """Add the 0003 columns if the migration was not applied.

    Existing Docker volumes never re-run init scripts, so without this a
    deployment that skipped 0003 would fail every publish. Idempotent.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS last_assessment_id TEXT"
        )
        cursor.execute(
            "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS result_status TEXT"
        )
        cursor.execute("DROP INDEX IF EXISTS hg_buildings_damaged_idx")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS hg_buildings_damaged_gix "
            "ON hg_buildings USING GIST (geometry) WHERE damage_class IS NOT NULL"
        )
    connection.commit()
    _SCHEMA_READY = True


def damage_color(damage_class: Optional[int]) -> str:
    if damage_class is None or not 0 <= int(damage_class) < len(DTF_CLASS_COLORS):
        return DTF_UNKNOWN_COLOR
    return DTF_CLASS_COLORS[int(damage_class)]


def publish_current_results(assessment_id: str, conn=None) -> int:
    """Copy this assessment's successful results onto ``hg_buildings``.

    One set-based UPDATE in a single transaction: all rows or none. The
    geometry column is never written. A row is replaced only when this
    result is ``completed`` and not older than the one already stored.

    Returns the number of buildings updated.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        ensure_schema(connection)
        from .assessment_store import ensure_review_schema

        ensure_review_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_buildings b
                SET damage_class       = d.damage_class,
                    damage_class_name  = d.damage_class_name,
                    confidence         = d.confidence,
                    result_status      = d.result_status,
                    review_status      = d.review_status,
                    last_assessment_id = d.assessment_id,
                    updated_at         = NOW()
                FROM hg_damage_assessments d
                WHERE d.assessment_id = %s
                  AND d.result_status = %s
                  AND d.damage_class IS NOT NULL
                  AND d.review_status <> 'rejected'
                  AND b.image_layer_id = d.image_layer_id
                  AND b.building_id = d.building_id
                  AND (b.updated_at IS NULL OR b.updated_at <= d.updated_at)
                """,
                (assessment_id, PUBLISHABLE_STATUS),
            )
            updated = cursor.rowcount
        connection.commit()
        logger.info(
            "Published %d current building results from assessment %s",
            updated, assessment_id,
        )
        return updated
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_conn:
            connection.close()


def _feature(row) -> Dict[str, Any]:
    building_id, geom, cls, name, conf, status, aid, updated = row
    return {
        "type": "Feature",
        "id": building_id,
        "geometry": json.loads(geom),
        "properties": {
            "building_id": building_id,
            "damage_class": cls,
            "damage_class_name": name
            or (DTF_CATEGORIES[cls] if cls is not None and 0 <= cls < 5 else None),
            "damage_color": damage_color(cls),
            "confidence": conf,
            "result_status": status,
            "assessment_id": aid,
            "updated_at": updated.isoformat() if updated else None,
        },
    }


def _query_rows(image_layer_id, bbox, limit, after_building_id, conn, geom_digits):
    if image_layer_id is None and not bbox:
        raise ValueError("bbox is required when imageLayerId is not given.")

    where = ["damage_class IS NOT NULL", "review_status <> 'rejected'"]
    params: List[Any] = []
    if image_layer_id is not None:
        where.append("image_layer_id = %s")
        params.append(image_layer_id)
    if bbox:
        where.append("geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)")
        params += list(bbox)
    if after_building_id:
        where.append("building_id > %s")
        params.append(after_building_id)
    params.append(limit)

    sql = f"""
        SELECT DISTINCT ON (building_id) building_id,
               ST_AsGeoJSON(geometry, {int(geom_digits)}), damage_class,
               damage_class_name, confidence, result_status,
               last_assessment_id, updated_at
        FROM hg_buildings
        WHERE {" AND ".join(where)}
        ORDER BY building_id, updated_at DESC NULLS LAST
        LIMIT %s
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()


def get_current_geojson(
    image_layer_id: Optional[str] = None,
    bbox: Optional[tuple] = None,
    limit: int = 50000,
    after_building_id: Optional[str] = None,
    conn=None,
) -> Dict[str, Any]:
    """Current damaged-building polygons as a GeoJSON dict.

    ``bbox`` is (min_lon, min_lat, max_lon, max_lat). Page with
    ``after_building_id`` = the ``nextCursor`` of the previous page.

    Without ``image_layer_id`` the query spans every layer (a statewide
    map) and ``bbox`` is required. A footprint loaded into several layers
    is returned once, with its most recent result.
    """
    rows = _query_rows(image_layer_id, bbox, limit, after_building_id, conn, 7)
    features = [_feature(r) for r in rows]
    return {
        "type": "FeatureCollection",
        "features": features,
        "nextCursor": features[-1]["id"] if len(features) == limit else None,
    }


def get_current_geojson_text(
    image_layer_id: Optional[str] = None,
    bbox: Optional[tuple] = None,
    limit: int = 50000,
    after_building_id: Optional[str] = None,
    conn=None,
) -> str:
    """Same payload as ``get_current_geojson`` but pre-serialised.

    PostGIS already emits each geometry as JSON text, so it is spliced in
    as-is instead of being parsed and re-encoded (the dominant cost for
    large pages). 6 decimals (~0.1 m) matches the source footprints.
    """
    rows = _query_rows(image_layer_id, bbox, limit, after_building_id, conn, 6)
    dumps = json.dumps
    parts = []
    for building_id, geom, cls, name, conf, status, aid, updated in rows:
        props = dumps(
            {
                "building_id": building_id,
                "damage_class": cls,
                "damage_class_name": name
                or (DTF_CATEGORIES[cls] if cls is not None and 0 <= cls < 5 else None),
                "damage_color": damage_color(cls),
                "confidence": conf,
                "result_status": status,
                "assessment_id": aid,
                "updated_at": updated.isoformat() if updated else None,
            },
            separators=(",", ":"),
        )
        parts.append(
            f'{{"type":"Feature","id":{dumps(building_id)},"geometry":{geom},"properties":{props}}}'
        )
    cursor = dumps(rows[-1][0]) if len(rows) == limit else "null"
    return (
        '{"type":"FeatureCollection","features":['
        + ",".join(parts)
        + f'],"nextCursor":{cursor}}}'
    )


def export_geoparquet(image_layer_id: str, path: str, conn=None) -> int:
    """Write the current output to a GeoParquet file (WKB geometry).

    A derived export of ``hg_buildings``; PostgreSQL stays authoritative and
    the file is rewritten atomically on each export. Returns rows written.
    """
    import pyarrow as pa

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT building_id, ST_AsBinary(geometry), damage_class,
                       damage_class_name, confidence, result_status,
                       last_assessment_id, updated_at
                FROM hg_buildings
                WHERE image_layer_id = %s AND damage_class IS NOT NULL
                  AND review_status <> 'rejected'
                ORDER BY building_id
                """,
                (image_layer_id,),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    from .geoparquet import write_geoparquet

    return write_geoparquet(
        path,
        {
            "building_id": [r[0] for r in rows],
            "damage_class": pa.array([r[2] for r in rows], pa.int32()),
            "damage_class_name": [r[3] for r in rows],
            "damage_color": [damage_color(r[2]) for r in rows],
            "confidence": pa.array([r[4] for r in rows], pa.float64()),
            "result_status": [r[5] for r in rows],
            "assessment_id": [r[6] for r in rows],
            "updated_at": pa.array([r[7] for r in rows], pa.timestamp("us", tz="UTC")),
            "image_layer_id": [image_layer_id] * len(rows),
        },
        [bytes(r[1]) for r in rows],
    )


def export_dir() -> Optional[str]:
    """Where automatic exports go; unset disables them."""
    return os.getenv("HASTE_CURRENT_DAMAGE_EXPORT_DIR") or None


def export_after_publish(image_layer_id: str, conn=None) -> Optional[str]:
    """Refresh the layer's GeoParquet if an export directory is configured.
    Never raises: an export problem must not fail the assessment."""
    base = export_dir()
    if not base:
        return None
    path = os.path.join(base, f"{image_layer_id}_current_damage.parquet")
    try:
        export_geoparquet(image_layer_id, path, conn=conn)
        return path
    except Exception:
        logger.exception("GeoParquet export failed for layer %s", image_layer_id)
        return None
