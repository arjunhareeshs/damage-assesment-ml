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


def load_footprints_into_postgis(
    image_layer, gpkg_local_path: str, conn=None
) -> int:
    """Replace ``image_layer``'s rows in ``hg_buildings`` with ``gpkg_local_path``'s contents.

    Args:
        image_layer: an ``ImageLayer`` (or any object exposing
            ``.imageLayerId`` / ``.projectId``).
        gpkg_local_path: local path to the footprints GeoPackage, with the
            schema written by ``hastegeo.core.utils.footprints``
            (``id``, ``geometry``, ``subtype``, ``class``).
        conn: optional existing psycopg2 connection. Tests inject a mock;
            production callers omit it and a connection is opened and
            closed here.

    Returns:
        Number of building rows written.
    """
    image_layer_id = image_layer.imageLayerId
    project_id = image_layer.projectId

    gdf = gpd.read_file(gpkg_local_path)
    if gdf.crs is None:
        raise ValueError(
            f"{gpkg_local_path} has no CRS; cannot load into PostGIS."
        )
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    rows = [
        (
            str(row.get("id")),
            image_layer_id,
            project_id,
            row.geometry.wkt,
            row.get("subtype"),
            row.get("class"),
        )
        for _, row in gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    ]

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM hg_buildings WHERE image_layer_id = %s",
                (image_layer_id,),
            )
            if rows:
                cursor.executemany(
                    """
                    INSERT INTO hg_buildings
                        (building_id, image_layer_id, project_id, geometry,
                         subtype, class)
                    VALUES
                        (%s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 4326),
                         %s, %s)
                    ON CONFLICT (image_layer_id, building_id) DO UPDATE SET
                        geometry = EXCLUDED.geometry,
                        subtype = EXCLUDED.subtype,
                        class = EXCLUDED.class
                    """,
                    rows,
                )
            connection.commit()
        logger.info(
            "Loaded %d footprints into hg_buildings for image layer %s",
            len(rows),
            image_layer_id,
        )
        return len(rows)
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

    # Derive sha1(footprint_source + "|" + rounded_wkb)
    try:
        rounded_geom = shapely.set_precision(geom, grid_size=1e-7)
    except Exception:
        rounded_geom = geom

    wkb_bytes = shapely.wkb.dumps(rounded_geom)
    h = hashlib.sha1()
    h.update(f"{footprint_source}|".encode("utf-8"))
    h.update(wkb_bytes)
    derived_id = h.hexdigest()
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

