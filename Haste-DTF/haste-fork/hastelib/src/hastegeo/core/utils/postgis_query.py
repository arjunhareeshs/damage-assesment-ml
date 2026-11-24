# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Live PostGIS queries against ``hg_buildings``.

``buildings_in_aoi`` / ``count_buildings_in_aoi`` / ``building_ids_in_aoi``
resolve which buildings an analyst-drawn polygon or explicit id selection
covers (``ST_Intersects``, never the whole layer). Damage results are written
by the large-AOI assessment (``assessment_store`` / ``current_damage``), not
here; this module only reads and never touches the ``geometry`` column.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .postgis_connect import get_connection

logger = logging.getLogger(__name__)


def buildings_in_aoi(
    image_layer_id: str,
    aoi_geojson: Optional[dict] = None,
    building_ids: Optional[list] = None,
    conn=None,
) -> list[dict]:
    """Resolve which buildings are in scope for an AOI-scoped DTF run.

    ``building_ids`` alone selects exactly those ids (an explicit
    click-selection, "Method 2 style" per plan.pdf); ``aoi_geojson`` alone
    (a GeoJSON Polygon/MultiPolygon in EPSG:4326, drawn on the map) selects
    every footprint that intersects it. When both are supplied the AOI still
    applies: only the selected ids that intersect it are returned. At least
    one of the two must be supplied.

    Returns:
        ``[{"building_id": str, "geometry": <GeoJSON dict, EPSG:4326>}, ...]``
    """
    if not aoi_geojson and not building_ids:
        raise ValueError(
            "buildings_in_aoi requires aoi_geojson or building_ids."
        )

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            if building_ids and aoi_geojson:
                cursor.execute(
                    """
                    SELECT building_id, ST_AsGeoJSON(geometry)
                    FROM hg_buildings
                    WHERE image_layer_id = %s AND building_id = ANY(%s)
                      AND ST_Intersects(
                          geometry,
                          ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                      )
                    """,
                    (image_layer_id, list(building_ids), json.dumps(aoi_geojson)),
                )
            elif building_ids:
                cursor.execute(
                    """
                    SELECT building_id, ST_AsGeoJSON(geometry)
                    FROM hg_buildings
                    WHERE image_layer_id = %s AND building_id = ANY(%s)
                    """,
                    (image_layer_id, list(building_ids)),
                )
            else:
                cursor.execute(
                    """
                    SELECT building_id, ST_AsGeoJSON(geometry)
                    FROM hg_buildings
                    WHERE image_layer_id = %s
                      AND ST_Intersects(
                          geometry,
                          ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                      )
                    """,
                    (image_layer_id, json.dumps(aoi_geojson)),
                )
            rows = cursor.fetchall()
        return [
            {"building_id": row[0], "geometry": json.loads(row[1])}
            for row in rows
        ]
    finally:
        if owns_conn:
            connection.close()


def count_buildings_in_aoi(
    image_layer_id: str,
    aoi_geojson: Optional[dict] = None,
    building_ids: Optional[list] = None,
    conn=None,
) -> int:
    """Same predicate as ``buildings_in_aoi``, but ``SELECT count(*)`` only.

    Backs a pre-submit "this AOI contains N buildings" preview so an
    analyst can be warned before queuing a very large per-building-crop
    run (see ``docs/OptionB_AOI_Integration_Plan.md`` "Biggest pitfalls" #5).
    """
    if not aoi_geojson and not building_ids:
        raise ValueError(
            "count_buildings_in_aoi requires aoi_geojson or building_ids."
        )

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            if building_ids and aoi_geojson:
                cursor.execute(
                    """
                    SELECT count(*) FROM hg_buildings
                    WHERE image_layer_id = %s AND building_id = ANY(%s)
                      AND ST_Intersects(
                          geometry,
                          ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                      )
                    """,
                    (image_layer_id, list(building_ids), json.dumps(aoi_geojson)),
                )
            elif building_ids:
                cursor.execute(
                    """
                    SELECT count(*) FROM hg_buildings
                    WHERE image_layer_id = %s AND building_id = ANY(%s)
                    """,
                    (image_layer_id, list(building_ids)),
                )
            else:
                cursor.execute(
                    """
                    SELECT count(*) FROM hg_buildings
                    WHERE image_layer_id = %s
                      AND ST_Intersects(
                          geometry,
                          ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                      )
                    """,
                    (image_layer_id, json.dumps(aoi_geojson)),
                )
            return int(cursor.fetchone()[0])
    finally:
        if owns_conn:
            connection.close()


# Upper bound on footprints sent to the labeling map in one response. Far
# above a typical image layer (hundreds to a few thousand buildings); beyond
# it the map shows the first N and says so, rather than the browser stalling.
MAX_MAP_BUILDINGS = 20000


def layer_buildings_feature_collection(
    image_layer_id: str,
    bbox: Optional[tuple] = None,
    limit: int = MAX_MAP_BUILDINGS,
    conn=None,
) -> dict:
    """The layer's building footprints as a GeoJSON FeatureCollection.

    Read from ``hg_buildings`` -- the same rows a DamageTriageFormer run
    resolves its buildings from -- so what the analyst sees and clicks on the
    map is exactly what a run covers, with the same ids. Unlike the old
    random-sample endpoint, nothing is left out unless the layer exceeds
    ``limit``; then the first ``limit`` (by id, so stable between calls) are
    returned with ``truncated: true``.

    Args:
        image_layer_id: Layer whose buildings to return.
        bbox: Optional ``(minx, miny, maxx, maxy)`` in EPSG:4326; only
            footprints whose bounding box overlaps it are returned.
        limit: Maximum number of features.

    Returns:
        ``{"type": "FeatureCollection", "features": [...], "total": int,
        "truncated": bool}``. Each feature has ``properties.id``.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            params: list = [image_layer_id]
            bbox_sql = ""
            if bbox is not None:
                bbox_sql = (
                    " AND geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)"
                )
                params.extend(float(v) for v in bbox)
            params.append(int(limit))
            # count(*) OVER () is evaluated before LIMIT, so it is the total
            # number of matching buildings, not just the returned ones.
            cursor.execute(
                f"""
                SELECT building_id, ST_AsGeoJSON(geometry), count(*) OVER ()
                FROM hg_buildings
                WHERE image_layer_id = %s{bbox_sql}
                ORDER BY building_id
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    features = [
        {
            "type": "Feature",
            "geometry": json.loads(geometry),
            "properties": {"id": building_id},
        }
        for building_id, geometry, _ in rows
    ]
    total = int(rows[0][2]) if rows else 0
    return {
        "type": "FeatureCollection",
        "features": features,
        "total": total,
        "truncated": total > len(features),
    }


def building_ids_in_aoi(
    image_layer_id: str,
    aoi_geojson: dict,
    limit: int = 100000,
    conn=None,
) -> list[str]:
    """Ids of the layer's buildings that intersect ``aoi_geojson``.

    Same ``ST_Intersects`` predicate as :func:`buildings_in_aoi`, so the
    buildings highlighted on the map are exactly the ones an AOI run would
    assess.
    """
    if not aoi_geojson:
        raise ValueError("building_ids_in_aoi requires aoi_geojson.")
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT building_id FROM hg_buildings
                WHERE image_layer_id = %s
                  AND ST_Intersects(
                      geometry,
                      ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                  )
                LIMIT %s
                """,
                (image_layer_id, json.dumps(aoi_geojson), int(limit)),
            )
            return [row[0] for row in cursor.fetchall()]
    finally:
        if owns_conn:
            connection.close()


def get_classified_buildings_geojson(
    image_layer_id: str,
    bbox: Optional[tuple] = None,
    limit: int = 5000,
    conn=None,
) -> dict:
    """Buildings of a layer with a published damage result, as GeoJSON.

    Backs the map's per-building vector layer
    (``ui/src/Components/Visualizer/Visualizer.jsx``): click a polygon to see
    its class, confidence and Accept / Reject controls.

    Bounded: ``bbox`` (minx, miny, maxx, maxy, EPSG:4326) restricts it to the
    viewport and at most ``limit`` features are returned, most damaged first
    (then by id, so pages are stable). ``truncated`` says whether more exist.
    Properties: ``id``, ``dtf_class``, ``dtf_class_name``, ``dtf_confidence``,
    ``review_status``, ``assessment_id``, ``result_out_of_date``. Rejected
    rows are included so the map can show them as rejected.
    """
    where = ["image_layer_id = %s", "damage_class IS NOT NULL"]
    params: list = [image_layer_id]
    if bbox is not None:
        where.append("geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)")
        params.extend(float(v) for v in bbox)
    params.append(int(limit) + 1)
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT building_id, ST_AsGeoJSON(geometry, 7), damage_class,
                       damage_class_name, confidence, review_status,
                       last_assessment_id,
                       imagery_updated_at IS NOT NULL
                           AND (updated_at IS NULL OR updated_at < imagery_updated_at)
                FROM hg_buildings
                WHERE {" AND ".join(where)}
                ORDER BY damage_class DESC, building_id
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    truncated = len(rows) > limit
    rows = rows[:limit]
    return {
        "type": "FeatureCollection",
        "truncated": truncated,
        "features": [
            {
                "type": "Feature",
                "geometry": json.loads(r[1]),
                "properties": {
                    "id": r[0],
                    "dtf_class": r[2],
                    "dtf_class_name": r[3],
                    "dtf_confidence": r[4],
                    "review_status": r[5],
                    "assessment_id": r[6],
                    "result_out_of_date": bool(r[7]),
                },
            }
            for r in rows
        ],
    }
