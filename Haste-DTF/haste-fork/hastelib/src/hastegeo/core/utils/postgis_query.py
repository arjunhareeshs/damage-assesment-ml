# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Live PostGIS queries and per-building writes against ``hg_buildings``.

This is the "AOI is a spatial filter" step of the AOI-scoped
DamageTriageFormer path (``modelType="dtf-aoi"``,
``docs/OptionB_AOI_Integration_Plan.md`` Module 6.3/6.6): given an
analyst-drawn polygon or an explicit building-id selection,
``buildings_in_aoi`` resolves exactly which buildings are in scope via
``ST_Intersects`` — never the whole layer — and
``update_building_predictions`` writes DamageTriageFormer's results back
one row at a time, without ever touching the ``geometry`` column (the
polygon itself never changes, only its attributes do).
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


_RUN_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS hg_dtf_run_results (
    model_id          TEXT NOT NULL,
    building_id       TEXT NOT NULL,
    image_layer_id    TEXT NOT NULL,
    project_id        TEXT,
    damage_class      INTEGER,
    damage_class_name TEXT,
    confidence        DOUBLE PRECISION,
    probabilities     JSONB,
    crop_url          TEXT,
    mask_url          TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, building_id)
);
CREATE INDEX IF NOT EXISTS hg_dtf_run_results_building_idx
    ON hg_dtf_run_results (image_layer_id, building_id);
"""
_RUN_RESULTS_READY = False


def ensure_run_results_table(connection) -> None:
    """Create ``hg_dtf_run_results`` if missing (same DDL as migration 0005).

    Migrations only run when the database is first created, so a database
    that predates 0005 gets the table here, once per process.
    """
    global _RUN_RESULTS_READY
    if _RUN_RESULTS_READY:
        return
    with connection.cursor() as cursor:
        cursor.execute(_RUN_RESULTS_DDL)
    connection.commit()
    _RUN_RESULTS_READY = True


def record_run_results(
    image_layer_id: str,
    model_id: str,
    rows: list[dict],
    project_id: Optional[str] = None,
    conn=None,
) -> int:
    """Store one DamageTriageFormer run's per-building results.

    The history behind ``hg_buildings`` (which only keeps the current
    result): one row per (run, building). Re-recording a run replaces its
    rows. ``rows`` items: ``building_id``, ``class``, ``class_name``,
    ``confidence``, and optionally ``probabilities`` (list or dict),
    ``crop_url``, ``mask_url``. Returns the number of rows written.
    """
    if not rows:
        return 0
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        ensure_run_results_table(connection)
        with connection.cursor() as cursor:
            for row in rows:
                probabilities = row.get("probabilities")
                cursor.execute(
                    """
                    INSERT INTO hg_dtf_run_results
                        (model_id, building_id, image_layer_id, project_id,
                         damage_class, damage_class_name, confidence,
                         probabilities, crop_url, mask_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (model_id, building_id) DO UPDATE SET
                        damage_class = EXCLUDED.damage_class,
                        damage_class_name = EXCLUDED.damage_class_name,
                        confidence = EXCLUDED.confidence,
                        probabilities = EXCLUDED.probabilities,
                        crop_url = EXCLUDED.crop_url,
                        mask_url = EXCLUDED.mask_url,
                        created_at = now()
                    """,
                    (
                        str(model_id),
                        str(row["building_id"]),
                        image_layer_id,
                        project_id,
                        row.get("class"),
                        row.get("class_name"),
                        row.get("confidence"),
                        json.dumps(probabilities) if probabilities is not None else None,
                        row.get("crop_url") or None,
                        row.get("mask_url") or None,
                    ),
                )
        connection.commit()
        return len(rows)
    finally:
        if owns_conn:
            connection.close()


def update_building_predictions(
    image_layer_id: str,
    model_id: str,
    rows: list[dict],
    review_status: str = "pending_review",
    conn=None,
) -> int:
    """Write DamageTriageFormer results back, one row (building) at a time.

    ``rows`` — as produced by
    ``hastegeo.workflows.damage_triage_aoi_infer.run_damage_triage_aoi`` —
    is ``[{"building_id", "class", "class_name", "confidence", "crop_url",
    "mask_url"}, ...]``.

    The ``geometry`` column is never referenced in the ``SET`` clause —
    plan.pdf's invariant: "the polygon never changed, only its attribute
    changed."

    Returns:
        Number of rows updated.
    """
    # A failed/unclassified row must never overwrite a good earlier result.
    rows = [r for r in rows if r.get("class") is not None and r["class"] >= 0]
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE hg_buildings SET
                    result_status = 'completed',
                    damage_class = %(class)s,
                    damage_class_name = %(class_name)s,
                    confidence = %(confidence)s,
                    mask_url = %(mask_url)s,
                    crop_url = %(crop_url)s,
                    review_status = %(review_status)s,
                    last_aoi_model_id = %(model_id)s,
                    updated_at = now()
                WHERE image_layer_id = %(image_layer_id)s
                  AND building_id = %(building_id)s
                """,
                [
                    {
                        **row,
                        "review_status": review_status,
                        "model_id": model_id,
                        "image_layer_id": image_layer_id,
                    }
                    for row in rows
                ],
            )
            updated = cursor.rowcount
            connection.commit()
        logger.info(
            "Updated %d building rows in hg_buildings for image layer %s "
            "(model %s)",
            updated,
            image_layer_id,
            model_id,
        )
        return updated
    finally:
        if owns_conn:
            connection.close()


def set_review_status(
    image_layer_id: str, building_id: str, review_status: str, conn=None
) -> bool:
    """Analyst Accept/Reject action — flips one building's review_status.

    Never touches ``damage_class``/``geometry``/any other column. Mirrors
    the existing embedding sub-model's ``PutBuildingValidation`` review
    pattern (``api/hastefuncapi/function_app.py``), applied to
    ``hg_buildings`` instead of the validation-labels store.

    Returns:
        True if a row was updated, False if no matching row was found.
    """
    if review_status not in (
        "unclassified",
        "pending_review",
        "accepted",
        "rejected",
    ):
        raise ValueError(f"Invalid review_status: {review_status}")

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_buildings
                SET review_status = %s, updated_at = now()
                WHERE image_layer_id = %s AND building_id = %s
                """,
                (review_status, image_layer_id, building_id),
            )
            updated = cursor.rowcount
            connection.commit()
        return updated > 0
    finally:
        if owns_conn:
            connection.close()


def get_classified_buildings_geojson(image_layer_id: str, conn=None) -> dict:
    """Every ``hg_buildings`` row with a prediction, as a GeoJSON
    FeatureCollection — backs the map's AOI-predictions vector layer
    (``ui/src/Components/Visualizer/Visualizer.jsx``), which is what makes
    plan.pdf Module 6.7's "click the polygon, see Building ID / crop /
    mask / class / confidence / Accept / Reject" possible: unlike the
    whole-layer ``dtf`` path's raster output, this is a real per-building
    vector layer, one feature per classified building.

    Only rows with ``damage_class IS NOT NULL`` are returned — buildings
    never touched by a ``dtf-aoi`` run don't clutter the layer.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT building_id, ST_AsGeoJSON(geometry), damage_class,
                       damage_class_name, confidence, mask_url, crop_url,
                       review_status, last_aoi_model_id
                FROM hg_buildings
                WHERE image_layer_id = %s AND damage_class IS NOT NULL
                """,
                (image_layer_id,),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": json.loads(r[1]),
                "properties": {
                    "id": r[0],
                    "dtf_class": r[2],
                    "dtf_class_name": r[3],
                    "dtf_confidence": r[4],
                    "mask_url": r[5],
                    "crop_url": r[6],
                    "review_status": r[7],
                    "model_id": r[8],
                },
            }
            for r in rows
        ],
    }


def export_aoi_gpkg(
    image_layer_id: str,
    model_id: str,
    output_path: str,
    conn=None,
    *,
    exclude_rejected: bool = True,
) -> int:
    """Write only the buildings a given dtf-aoi run touched to a GeoPackage.

    Unlike the whole-layer path's full-layer export, this is scoped to
    ``last_aoi_model_id = model_id`` — the buildings this specific AOI run
    classified — so ``Model.gpkgUrl`` for a ``dtf-aoi`` row is small and
    AOI-scoped, not a restatement of the entire layer.

    Args:
        exclude_rejected: When True (the default —
            docs/OptionB_AOI_Integration_Plan.md "Decide: human review or
            fully automated?"), buildings an analyst has marked
            ``review_status="rejected"`` are dropped from the export
            entirely, since a rejected prediction is one the analyst has
            said is wrong and downstream GIS consumers shouldn't treat as
            a real classification. Every exported row also carries a
            ``reviewed`` boolean column (``review_status != "pending_review"``)
            so consumers can distinguish reviewed from unreviewed
            predictions even when rejects aren't excluded.

    Returns:
        Number of features written.
    """
    import geopandas as gpd  # type: ignore
    from shapely import wkb  # type: ignore

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                SELECT building_id, ST_AsBinary(geometry), subtype, class,
                       damage_class, damage_class_name, confidence,
                       mask_url, crop_url, review_status
                FROM hg_buildings
                WHERE image_layer_id = %s AND last_aoi_model_id = %s
            """
            params = [image_layer_id, model_id]
            if exclude_rejected:
                query += " AND review_status != 'rejected'"
            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    gdf = gpd.GeoDataFrame(
        [
            {
                "id": r[0],
                "subtype": r[2],
                "class": r[3],
                "dtf_class": r[4],
                "dtf_class_name": r[5],
                "dtf_confidence": r[6],
                "mask_url": r[7],
                "crop_url": r[8],
                "review_status": r[9],
                "reviewed": r[9] != "pending_review",
                "geometry": wkb.loads(bytes(r[1])),
            }
            for r in rows
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.to_file(output_path, layer="predictions", driver="GPKG")
    return len(gdf)
