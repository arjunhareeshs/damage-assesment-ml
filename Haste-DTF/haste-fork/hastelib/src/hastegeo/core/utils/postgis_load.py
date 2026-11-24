# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Load a footprints GeoPackage into the ``hg_buildings`` PostGIS table.

Called once per ``ImageLayer`` by the imageryprep workflow
(``hastegeo.workflows.prepare_imagery``), immediately after
``hastegeo.core.utils.footprints`` writes the cached footprints
GeoPackage. Idempotent: re-running imagery prep (e.g. after a
footprints re-upload) replaces this layer's rows rather than
duplicating them, and re-loading the same file twice is a no-op change.

See ``db/migrations/0001_hg_buildings.sql`` for the table schema and
``docs/OptionB_AOI_Integration_Plan.md`` Module 2 for why this table
exists: ``hastegeo.core.utils.postgis_query.buildings_in_aoi`` runs a
live ``ST_Intersects`` query against it for every AOI an analyst draws,
instead of re-reading and re-clipping the full footprints GeoPackage
(potentially millions of rows) on every draw.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, List, Optional, Set, Tuple

import geopandas as gpd  # type: ignore
import pyogrio  # type: ignore
import shapely  # type: ignore
import shapely.geometry  # type: ignore
import shapely.wkb  # type: ignore

from .postgis_connect import get_connection

logger = logging.getLogger(__name__)


LOAD_CHUNK = int(os.getenv("HASTE_FOOTPRINT_LOAD_CHUNK", "50000"))
_MISSING_IDS = ("", "none", "nan", "null")


def _ensure_building_columns(cursor) -> None:
    for ddl in (
        "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS imagery_updated_at TIMESTAMPTZ",
        "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS footprint_source TEXT",
        "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS footprint_version TEXT",
    ):
        cursor.execute(ddl)


def _footprint_rows(
    gpkg_local_path: str, source: str, chunk_size: int, stats: dict
):
    """Yield lists of ``(building_id, wkb, subtype, class)`` per chunk.

    Reads ``chunk_size`` features at a time (never the whole file). A
    missing id becomes ``canonical_building_id``; a repeated id gets a
    ``-2``, ``-3`` suffix so no building silently overwrites another.
    """
    from .building_ids import canonical_building_id

    info = pyogrio.read_info(gpkg_local_path)
    total = int(info.get("features") or 0)
    if info.get("crs") is None and total:
        raise ValueError(f"{gpkg_local_path} has no CRS; cannot load into PostGIS.")
    seen: Set[str] = set()
    for offset in range(0, total, chunk_size):
        gdf = pyogrio.read_dataframe(
            gpkg_local_path, skip_features=offset, max_features=chunk_size
        )
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        ids = gdf["id"] if "id" in gdf.columns else [None] * len(gdf)
        subtypes = gdf["subtype"] if "subtype" in gdf.columns else [None] * len(gdf)
        classes = gdf["class"] if "class" in gdf.columns else [None] * len(gdf)
        rows = []
        for raw_id, geom, subtype, klass in zip(ids, gdf.geometry, subtypes, classes):
            if geom is None or geom.is_empty:
                stats["skipped_empty"] += 1
                continue
            bid = None if raw_id is None else str(raw_id).strip()
            if not bid or bid.lower() in _MISSING_IDS:
                bid = canonical_building_id(geom, source)
                stats["derived_ids"] += 1
            if bid in seen:
                n = 2
                while f"{bid}-{n}" in seen:
                    n += 1
                bid = f"{bid}-{n}"
                stats["duplicate_ids"] += 1
            seen.add(bid)
            rows.append((
                bid,
                shapely.to_wkb(geom, hex=True),
                None if subtype is None or subtype != subtype else str(subtype),
                None if klass is None or klass != klass else str(klass),
            ))
        yield rows


def load_footprints_into_postgis(
    image_layer,
    gpkg_local_path: str,
    conn=None,
    preserve_results: bool = False,
    footprint_source: Optional[str] = None,
    footprint_version: Optional[str] = None,
    chunk_size: int = LOAD_CHUNK,
) -> int:
    """Make ``hg_buildings`` hold exactly the buildings in ``gpkg_local_path``.

    The file is read in chunks and staged into a temporary table with
    batched inserts, then applied in ONE transaction (all or nothing):

    * buildings no longer in the file are removed;
    * buildings in the file are inserted or updated (geometry, subtype,
      class, footprint source/version). Existing damage results are never
      wiped;
    * results that may no longer hold are flagged, not deleted, by setting
      ``imagery_updated_at`` (a result older than it is out of date):
      every existing building on a full re-preparation (the imagery itself
      was replaced), only buildings whose outline changed on an imagery
      update (``preserve_results``; the updated area is stamped separately
      by ``mark_imagery_updated``).

    Returns the number of buildings now loaded for the layer.
    """
    from psycopg2.extras import execute_values

    image_layer_id = image_layer.imageLayerId
    project_id = image_layer.projectId
    source = footprint_source or "unknown"
    stats = {"skipped_empty": 0, "derived_ids": 0, "duplicate_ids": 0}

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            _ensure_building_columns(cursor)
            cursor.execute(
                "CREATE TEMP TABLE _hg_fp_stage (building_id TEXT PRIMARY KEY, "
                "geometry geometry(Geometry, 4326), subtype TEXT, class TEXT) "
                "ON COMMIT DROP"
            )
            staged = 0
            for rows in _footprint_rows(gpkg_local_path, source, chunk_size, stats):
                if not rows:
                    continue
                execute_values(
                    cursor,
                    "INSERT INTO _hg_fp_stage (building_id, geometry, subtype, class) "
                    "VALUES %s",
                    rows,
                    template="(%s, ST_SetSRID(ST_GeomFromWKB(decode(%s, 'hex')), 4326), %s, %s)",
                    page_size=5000,
                )
                staged += len(rows)
            cursor.execute(
                """
                DELETE FROM hg_buildings b
                WHERE b.image_layer_id = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM _hg_fp_stage s WHERE s.building_id = b.building_id
                  )
                """,
                (image_layer_id,),
            )
            removed = cursor.rowcount
            cursor.execute(
                """
                INSERT INTO hg_buildings
                    (building_id, image_layer_id, project_id, geometry,
                     subtype, class, footprint_source, footprint_version)
                SELECT building_id, %s, %s, geometry, subtype, class, %s, %s
                FROM _hg_fp_stage
                ON CONFLICT (image_layer_id, building_id) DO UPDATE SET
                    geometry = EXCLUDED.geometry,
                    subtype = EXCLUDED.subtype,
                    class = EXCLUDED.class,
                    footprint_source = EXCLUDED.footprint_source,
                    footprint_version = EXCLUDED.footprint_version,
                    imagery_updated_at = CASE
                        WHEN %s THEN NOW()
                        WHEN NOT ST_Equals(hg_buildings.geometry, EXCLUDED.geometry)
                            THEN NOW()
                        ELSE hg_buildings.imagery_updated_at
                    END
                """,
                (
                    image_layer_id, project_id, footprint_source,
                    footprint_version, not preserve_results,
                ),
            )
        connection.commit()
        logger.info(
            "Loaded %d footprints into hg_buildings for image layer %s "
            "(removed %d no longer present; %d ids derived from geometry, "
            "%d duplicate ids suffixed, %d empty geometries skipped; source "
            "%s %s)",
            staged, image_layer_id, removed, stats["derived_ids"],
            stats["duplicate_ids"], stats["skipped_empty"], source,
            footprint_version or "",
        )
        return staged
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_conn:
            connection.close()


def compute_canonical_building_id(
    raw_id: Optional[Any],
    geom: Any,
    footprint_source: str = "microsoft",
    seen_ids: Optional[Set[str]] = None,
) -> str:
    """Compute canonical building ID.

    Uses source file's ID if present and non-duplicate.
    Otherwise derives sha1(footprint_source + "|" + WKB with coordinates rounded to 7 decimals).
    Never uses a row index.
    """
    if raw_id is not None:
        cand = str(raw_id).strip()
        if cand and cand.lower() not in ("none", "nan", "null"):
            if seen_ids is None or cand not in seen_ids:
                if seen_ids is not None:
                    seen_ids.add(cand)
                return cand

    from .building_ids import canonical_building_id

    derived_id = canonical_building_id(geom, footprint_source)
    if seen_ids is not None:
        seen_ids.add(derived_id)
    return derived_id


def load_footprints_streaming(
    image_layer,
    path: str,
    footprint_source: str = "microsoft",
    footprint_version: str = "1.0",
    chunk_size: int = 50000,
    conn=None,
) -> int:
    """Stream a large footprints file in bounded chunks into hg_buildings.

    Never loads the entire file into memory at once. Idempotent: deletes
    existing rows for this image layer before inserting.

    Args:
        image_layer: an ImageLayer object or object with imageLayerId/projectId.
        path: local path to the footprints file (.geojson, .gpkg, .shp, .jsonl).
        footprint_source: provider identifier (e.g. 'microsoft', 'overture').
        footprint_version: dataset version identifier.
        chunk_size: number of features to read and insert per batch.
        conn: optional existing database connection.

    Returns:
        int: Total number of building footprints loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Footprints file not found at: {path}")

    image_layer_id = getattr(image_layer, "imageLayerId", str(image_layer))
    project_id = getattr(image_layer, "projectId", "")

    owns_conn = conn is None
    connection = conn or get_connection()
    seen_ids: Set[str] = set()
    total_loaded = 0

    insert_sql = """
        INSERT INTO hg_buildings
            (building_id, image_layer_id, project_id, geometry,
             subtype, class, footprint_source, footprint_version)
        VALUES
            (%s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 4326),
             %s, %s, %s, %s)
        ON CONFLICT (image_layer_id, building_id) DO UPDATE SET
            geometry = EXCLUDED.geometry,
            subtype = EXCLUDED.subtype,
            class = EXCLUDED.class,
            footprint_source = EXCLUDED.footprint_source,
            footprint_version = EXCLUDED.footprint_version
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM hg_buildings WHERE image_layer_id = %s",
                (image_layer_id,),
            )

            # Check if file is newline-delimited GeoJSON
            is_jsonl = path.lower().endswith((".geojsonl", ".jsonl"))

            if is_jsonl:
                batch_rows: List[Tuple[Any, ...]] = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            feat = json.loads(line)
                            geom = shapely.geometry.shape(feat.get("geometry"))
                            if geom is None or geom.is_empty:
                                continue
                            props = feat.get("properties") or {}
                            raw_id = (
                                feat.get("id")
                                or props.get("id")
                                or props.get("ID")
                                or props.get("building_id")
                            )
                            b_id = compute_canonical_building_id(
                                raw_id, geom, footprint_source, seen_ids
                            )
                            batch_rows.append(
                                (
                                    b_id,
                                    image_layer_id,
                                    project_id,
                                    geom.wkt,
                                    props.get("subtype"),
                                    props.get("class"),
                                    footprint_source,
                                    footprint_version,
                                )
                            )
                            if len(batch_rows) >= chunk_size:
                                cursor.executemany(insert_sql, batch_rows)
                                total_loaded += len(batch_rows)
                                batch_rows = []
                        except Exception as e:
                            logger.warning("Skipping invalid JSON line: %s", e)
                if batch_rows:
                    cursor.executemany(insert_sql, batch_rows)
                    total_loaded += len(batch_rows)
            else:
                # One sequential pass over the file in Arrow record batches
                # (pyogrio.open_arrow): memory stays bounded by chunk_size
                # and, unlike skip_features paging, the file is not re-read
                # from the start for every chunk. A read error propagates —
                # a partial load must not be reported as a success.
                with pyogrio.open_arrow(
                    path, batch_size=chunk_size, use_pyarrow=True
                ) as (meta, reader):
                    geom_col = meta.get("geometry_name") or "wkb_geometry"
                    src_crs = meta.get("crs")
                    for batch in reader:
                        columns = batch.schema.names
                        geoms = gpd.GeoSeries(
                            shapely.from_wkb(batch.column(geom_col).to_numpy(zero_copy_only=False)),
                            crs=src_crs,
                        )
                        if src_crs is not None and geoms.crs.to_epsg() != 4326:
                            geoms = geoms.to_crs(epsg=4326)

                        id_col = next(
                            (c for c in ("id", "ID", "building_id", "BUILDING_ID", "Id") if c in columns),
                            None,
                        )
                        raw_ids = batch.column(id_col).to_pylist() if id_col else [None] * len(geoms)
                        subtypes = batch.column("subtype").to_pylist() if "subtype" in columns else [None] * len(geoms)
                        classes = batch.column("class").to_pylist() if "class" in columns else [None] * len(geoms)

                        chunk_rows = []
                        for geom, raw_id, subtype, cls in zip(geoms, raw_ids, subtypes, classes):
                            if geom is None or geom.is_empty:
                                continue
                            b_id = compute_canonical_building_id(
                                raw_id, geom, footprint_source, seen_ids
                            )
                            chunk_rows.append(
                                (
                                    b_id,
                                    image_layer_id,
                                    project_id,
                                    geom.wkt,
                                    subtype,
                                    cls,
                                    footprint_source,
                                    footprint_version,
                                )
                            )
                        if chunk_rows:
                            cursor.executemany(insert_sql, chunk_rows)
                            total_loaded += len(chunk_rows)
                        logger.debug(
                            "Streamed %d footprints (cumulative: %d)",
                            len(chunk_rows),
                            total_loaded,
                        )

            connection.commit()
            logger.info(
                "Streaming load finished: %d footprints loaded into hg_buildings for layer %s",
                total_loaded,
                image_layer_id,
            )
            return total_loaded
    finally:
        if owns_conn:
            connection.close()



def mark_imagery_updated(image_layer_id: str, area_geojson: dict, conn=None) -> dict:
    """Stamp the buildings under an imagery update.

    Sets ``imagery_updated_at`` on every building of the layer that the
    updated area touches. A damage result written before that time
    (``updated_at < imagery_updated_at``) was made on imagery the layer no
    longer shows and should be re-run.

    Returns ``{"buildings": n touched, "results_out_of_date": n of them
    that already had a damage result}``.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS imagery_updated_at TIMESTAMPTZ"
            )
            cursor.execute(
                """
                UPDATE hg_buildings
                SET imagery_updated_at = NOW()
                WHERE image_layer_id = %s
                  AND ST_Intersects(geometry, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                RETURNING damage_class IS NOT NULL
                """,
                (image_layer_id, json.dumps(area_geojson)),
            )
            flags = [r[0] for r in cursor.fetchall()]
        connection.commit()
        return {"buildings": len(flags), "results_out_of_date": int(sum(flags))}
    finally:
        if owns_conn:
            connection.close()
