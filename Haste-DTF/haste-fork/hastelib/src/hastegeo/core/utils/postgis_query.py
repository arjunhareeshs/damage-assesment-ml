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

    ``building_ids`` takes precedence when non-empty (an explicit
    click-selection, "Method 2 style" per plan.pdf); otherwise every
    footprint that intersects ``aoi_geojson`` (a GeoJSON Polygon/
    MultiPolygon in EPSG:4326, drawn on the map) is returned. Exactly one
    of the two must be supplied.

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
            if building_ids:
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
            if building_ids:
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
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE hg_buildings SET
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
