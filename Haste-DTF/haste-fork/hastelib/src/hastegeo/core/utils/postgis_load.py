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

import logging

import geopandas as gpd  # type: ignore

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
