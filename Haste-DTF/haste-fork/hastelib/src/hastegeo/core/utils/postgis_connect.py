# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared PostgreSQL/PostGIS connection helper for the ``hg_buildings``
table (the AOI-scoped DamageTriageFormer path, ``modelType="dtf-aoi"``).

Mirrors the connection pattern in
``hastegeo.core.data_layer.azure_postgresql_data_layer`` but is kept
separate: that class owns a generic identifier/data_type/JSONB key-value
table, while ``hg_buildings`` is a dedicated per-building row table with
real columns (see ``db/migrations/0001_hg_buildings.sql``) queried with
``ST_Intersects``, not looked up by identifier.

Two auth modes, selected by whether ``POSTGRES_PASSWORD`` is set:

- **Local/dev** (a plain ``postgres``+PostGIS container in
  ``docker/docker-compose.yml``): password auth via the same
  ``POSTGRES_HOST``/``POSTGRES_PORT``/``POSTGRES_DATABASE``/``POSTGRES_USER``/
  ``POSTGRES_PASSWORD`` env vars already read by ``Config``'s
  ``StorageType.POSTGRES`` block.
- **Azure Database for PostgreSQL**: no password configured, so a
  ``DefaultAzureCredential`` AAD token is fetched per-connection (tokens
  are short-lived and must not be cached across connections) — the same
  approach ``AzurePostgreSQLDataLayer`` uses.
"""

from __future__ import annotations

import os

import psycopg2  # type: ignore


def get_connection():
    """Open a new psycopg2 connection to the hg_buildings database.

    Callers are responsible for closing the connection (or use it as a
    context manager). Every ``hastegeo.core.utils.postgis_query`` /
    ``postgis_load`` function accepts an optional ``conn`` parameter so
    tests can inject a mock instead of calling this.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DATABASE", "mydatabase")
    user = os.getenv("POSTGRES_USER", "myuser")
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        from azure.identity import DefaultAzureCredential  # type: ignore

        credential = DefaultAzureCredential()
        password = credential.get_token(
            "https://ossrdbms-aad.database.windows.net/.default"
        ).token

    # TLS is required by default so a production deployment against Azure
    # Database for PostgreSQL never silently negotiates plaintext. The
    # local-dev postgis container in docker/docker-compose.yml is built
    # without TLS support, so that stack (and only that stack) opts out by
    # setting POSTGRES_SSLMODE=disable -- without this the footprint load
    # fails with "server does not support SSL, but SSL was required" and
    # AOI-scoped DamageTriageFormer runs see an empty hg_buildings table.
    sslmode = os.getenv("POSTGRES_SSLMODE", "require")

    connection_string = (
        f"host={host} port={port} dbname={database} user={user} "
        f"password={password} sslmode={sslmode}"
    )
    return psycopg2.connect(connection_string)
