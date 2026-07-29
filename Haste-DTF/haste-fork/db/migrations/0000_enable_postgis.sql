-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
-- Must run before 0001_hg_buildings.sql (filename-sorted, so 0000 < 0001).
-- Mounted into the postgis service's /docker-entrypoint-initdb.d/ in
-- docker/docker-compose.yml, where the postgres image runs every .sql
-- file in this directory, in name order, on first container start.

CREATE EXTENSION IF NOT EXISTS postgis;
