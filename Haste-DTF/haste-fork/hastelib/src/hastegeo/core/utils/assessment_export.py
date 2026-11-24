# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tabular and spatial exports of one large-AOI assessment.

One row per building, one fixed column set (``TABLE_COLUMNS``), whatever the
format:

* ``json`` / ``csv`` / ``parquet``: the plain table (centroid lon/lat, no
  geometry);
* ``geoparquet``: the table plus footprint polygons;
* ``gpkg``: a GeoPackage with three layers: ``buildings`` (footprints + the
  table), ``aoi`` (the drawn AOI polygon with the assessment's metadata) and
  ``summary`` (class x review-status counts, a non-spatial table);
* ``aoi``: the AOI alone as a GeoJSON Feature.

Rejected results are left out unless ``include_rejected`` is set, so a
downstream GIS never treats an analyst-rejected prediction as real.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .dtf_constants import DTF_CATEGORIES
from .postgis_connect import get_connection

FORMATS = ("json", "csv", "parquet", "geoparquet", "gpkg", "aoi")
REVIEW_FILTERS = ("all", "accepted", "pending_review", "rejected", "unclassified")


def _slug(name: str) -> str:
    return "p_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


PROB_COLUMNS = [_slug(n) for n in DTF_CATEGORIES]

TABLE_COLUMNS = (
    ["assessment_id", "image_layer_id", "building_id",
     "damage_class", "damage_class_name", "confidence"]
    + PROB_COLUMNS
    + ["result_status", "review_status", "reviewed_by", "reviewed_at",
       "boundary_intersecting", "aoi_overlap_fraction", "winning_unit_id",
       "model_digest", "preprocessing_version", "footprint_version",
       "assessed_at", "centroid_lon", "centroid_lat", "footprint_area_m2"]
)


def fetch_results(
    assessment_id: str,
    review: str = "all",
    include_rejected: bool = False,
    conn=None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(assessment metadata, rows)``; metadata is ``None`` if unknown.

    Each row has the ``TABLE_COLUMNS`` keys plus ``_wkb`` (footprint, EPSG:4326).
    """
    if review not in REVIEW_FILTERS:
        raise ValueError(f"review must be one of {REVIEW_FILTERS}.")
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        from .assessment_store import ensure_review_schema

        ensure_review_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT assessment_id, project_id, image_layer_id, name, status,
                       aoi_area_km2, ST_AsGeoJSON(aoi_geometry), model_digest,
                       preprocessing_version, created_at, completed_at, config
                FROM hg_assessments WHERE assessment_id = %s
                """,
                (assessment_id,),
            )
            a = cursor.fetchone()
            if a is None:
                return None, []
            meta = {
                "assessment_id": a[0], "project_id": a[1], "image_layer_id": a[2],
                "name": a[3], "status": a[4], "aoi_area_km2": a[5],
                "aoi_geometry": json.loads(a[6]), "model_digest": a[7],
                "preprocessing_version": a[8],
                "created_at": a[9].isoformat() if a[9] else None,
                "completed_at": a[10].isoformat() if a[10] else None,
                "config": a[11] or {},
            }
            where = ["d.assessment_id = %s"]
            params: List[Any] = [assessment_id]
            if review != "all":
                where.append("d.review_status = %s")
                params.append(review)
            elif not include_rejected:
                where.append("d.review_status <> 'rejected'")
            cursor.execute(
                f"""
                SELECT d.assessment_id, d.image_layer_id, d.building_id,
                       d.damage_class, d.damage_class_name, d.confidence,
                       e.probs, d.result_status, d.review_status, d.reviewed_by,
                       d.reviewed_at, d.boundary_intersecting,
                       d.aoi_overlap_fraction, d.winning_unit_id, d.model_digest,
                       d.preprocessing_version, d.footprint_version, d.updated_at,
                       ST_X(ST_Centroid(b.geometry)), ST_Y(ST_Centroid(b.geometry)),
                       ST_Area(b.geometry::geography), ST_AsBinary(b.geometry)
                FROM hg_damage_assessments d
                JOIN hg_buildings b
                  ON b.image_layer_id = d.image_layer_id AND b.building_id = d.building_id
                LEFT JOIN hg_prediction_evidence e
                  ON e.assessment_id = d.assessment_id
                 AND e.building_id = d.building_id AND e.unit_id = d.winning_unit_id
                WHERE {" AND ".join(where)}
                ORDER BY d.building_id
                """,
                params,
            )
            raw = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()
    return meta, [_row(r) for r in raw]


def _row(r) -> Dict[str, Any]:
    probs = list(r[6]) if r[6] is not None else [None] * len(DTF_CATEGORIES)
    row = {
        "assessment_id": r[0], "image_layer_id": r[1], "building_id": r[2],
        "damage_class": r[3], "damage_class_name": r[4], "confidence": r[5],
        **{c: (None if p is None else float(p)) for c, p in zip(PROB_COLUMNS, probs)},
        "result_status": r[7], "review_status": r[8], "reviewed_by": r[9],
        "reviewed_at": r[10].isoformat() if r[10] else None,
        "boundary_intersecting": r[11], "aoi_overlap_fraction": r[12],
        "winning_unit_id": r[13], "model_digest": r[14],
        "preprocessing_version": r[15], "footprint_version": r[16],
        "assessed_at": r[17].isoformat() if r[17] else None,
        "centroid_lon": r[18], "centroid_lat": r[19],
        "footprint_area_m2": r[20],
        "_wkb": bytes(r[21]) if r[21] is not None else None,
    }
    return row


def table_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows reduced to exactly ``TABLE_COLUMNS`` (no geometry)."""
    return [{c: r.get(c) for c in TABLE_COLUMNS} for r in rows]


def summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Counts per damage class and review status, plus a total row."""
    statuses = ("accepted", "pending_review", "rejected", "unclassified")
    out = []
    total = len(rows)
    for idx, name in enumerate(DTF_CATEGORIES):
        sel = [r for r in rows if r.get("damage_class") == idx]
        out.append({
            "damage_class": idx, "damage_class_name": name, "buildings": len(sel),
            "share": round(len(sel) / total, 6) if total else 0.0,
            **{s: sum(1 for r in sel if r.get("review_status") == s) for s in statuses},
        })
    none = [r for r in rows if r.get("damage_class") is None]
    out.append({
        "damage_class": None, "damage_class_name": "No result", "buildings": len(none),
        "share": round(len(none) / total, 6) if total else 0.0,
        **{s: sum(1 for r in none if r.get("review_status") == s) for s in statuses},
    })
    out.append({
        "damage_class": None, "damage_class_name": "Total", "buildings": total,
        "share": 1.0 if total else 0.0,
        **{s: sum(1 for r in rows if r.get("review_status") == s) for s in statuses},
    })
    return out


def aoi_feature(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The drawn AOI as a GeoJSON Feature carrying the assessment's metadata."""
    return {
        "type": "Feature",
        "geometry": meta["aoi_geometry"],
        "properties": {
            k: meta.get(k)
            for k in ("assessment_id", "project_id", "image_layer_id", "name",
                      "status", "aoi_area_km2", "model_digest",
                      "preprocessing_version", "created_at", "completed_at")
        },
    }


def write_csv(rows: List[Dict[str, Any]], path: str) -> int:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        w.writerows(table_rows(rows))
    return len(rows)


def _arrow_table(rows: List[Dict[str, Any]]):
    import pyarrow as pa

    return pa.Table.from_pylist(table_rows(rows), schema=_arrow_schema())


def _arrow_schema():
    import pyarrow as pa

    ints = {"damage_class"}
    bools = {"boundary_intersecting"}
    floats = set(PROB_COLUMNS) | {
        "confidence", "aoi_overlap_fraction", "centroid_lon", "centroid_lat",
        "footprint_area_m2",
    }
    fields = []
    for c in TABLE_COLUMNS:
        t = pa.int32() if c in ints else pa.bool_() if c in bools \
            else pa.float64() if c in floats else pa.string()
        fields.append(pa.field(c, t))
    return pa.schema(fields)


def write_parquet(rows: List[Dict[str, Any]], path: str) -> int:
    import pyarrow.parquet as pq

    tmp = path + ".tmp"
    pq.write_table(_arrow_table(rows), tmp, compression="zstd")
    os.replace(tmp, path)
    return len(rows)


def write_geoparquet_file(rows: List[Dict[str, Any]], path: str) -> int:
    from .geoparquet import write_geoparquet

    t = _arrow_table(rows)
    return write_geoparquet(
        path,
        {c: t.column(c) for c in TABLE_COLUMNS},
        [r["_wkb"] for r in rows],
    )


def write_gpkg(meta: Dict[str, Any], rows: List[Dict[str, Any]], path: str) -> int:
    """GeoPackage with ``buildings``, ``aoi`` and ``summary`` layers."""
    import geopandas as gpd
    import pandas as pd
    from shapely import wkb
    from shapely.geometry import shape

    if os.path.exists(path):
        os.remove(path)
    buildings = gpd.GeoDataFrame(
        table_rows(rows),
        geometry=[wkb.loads(r["_wkb"]) for r in rows],
        crs="EPSG:4326",
    )
    buildings.to_file(path, layer="buildings", driver="GPKG")
    aoi = aoi_feature(meta)
    gpd.GeoDataFrame(
        [aoi["properties"]], geometry=[shape(aoi["geometry"])], crs="EPSG:4326"
    ).to_file(path, layer="aoi", driver="GPKG", mode="a")
    import pyogrio

    pyogrio.write_dataframe(
        pd.DataFrame(summary_rows(rows)), path, layer="summary", driver="GPKG",
        append=True,
    )
    return len(rows)


def write_export(
    fmt: str, meta: Dict[str, Any], rows: List[Dict[str, Any]], path: str
) -> int:
    """Write ``rows`` to ``path`` in a file ``fmt`` (csv/parquet/geoparquet/gpkg)."""
    if fmt == "csv":
        return write_csv(rows, path)
    if fmt == "parquet":
        return write_parquet(rows, path)
    if fmt == "geoparquet":
        return write_geoparquet_file(rows, path)
    if fmt == "gpkg":
        return write_gpkg(meta, rows, path)
    raise ValueError(f"{fmt!r} is not a file format.")
