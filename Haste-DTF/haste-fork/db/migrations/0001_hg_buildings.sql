-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
-- hg_buildings: per-building spatial index backing the AOI-scoped
-- DamageTriageFormer path (modelType="dtf-aoi"). Loaded once per
-- ImageLayer by hastegeo.core.utils.postgis_load.load_footprints_into_postgis
-- (called from the imageryprep workflow right after the footprints
-- GeoPackage is written), then queried live with ST_Intersects by
-- hastegeo.core.utils.postgis_query.buildings_in_aoi for every AOI an
-- analyst draws. See docs/OptionB_AOI_Integration_Plan.md Module 2.
--
-- Requires the PostGIS extension:
--   CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS hg_buildings (
    building_id       TEXT NOT NULL,
    image_layer_id    TEXT NOT NULL,
    project_id        TEXT NOT NULL,
    geometry          GEOMETRY(Geometry, 4326) NOT NULL,
    subtype           TEXT,
    class             TEXT,
    damage_class      INTEGER,
    damage_class_name TEXT,
    confidence        DOUBLE PRECISION,
    mask_url          TEXT,
    crop_url          TEXT,
    -- unclassified | pending_review | accepted | rejected
    review_status     TEXT NOT NULL DEFAULT 'unclassified',
    last_aoi_model_id TEXT,
    updated_at        TIMESTAMPTZ,
    PRIMARY KEY (image_layer_id, building_id)
);

CREATE INDEX IF NOT EXISTS hg_buildings_geom_gix
    ON hg_buildings USING GIST (geometry);

CREATE INDEX IF NOT EXISTS hg_buildings_image_layer_id_idx
    ON hg_buildings (image_layer_id);
