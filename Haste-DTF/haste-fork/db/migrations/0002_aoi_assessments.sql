-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
-- 0002_aoi_assessments.sql: Large-AOI batched damage assessment schema.
-- Backs the AOI-first workflow for multi-TIF bounded tile batching.

-- 1. Multi-TIF imagery catalogue
CREATE TABLE IF NOT EXISTS hg_imagery (
    imagery_id        TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL,
    image_layer_id    TEXT NOT NULL,
    file_uri          TEXT NOT NULL,
    footprint         GEOMETRY(Polygon, 4326) NOT NULL,
    crs               TEXT NOT NULL,
    width             INTEGER NOT NULL,
    height            INTEGER NOT NULL,
    pixel_size_x      DOUBLE PRECISION NOT NULL,
    pixel_size_y      DOUBLE PRECISION NOT NULL,
    band_count        INTEGER NOT NULL,
    dtype             TEXT NOT NULL,
    nodata            DOUBLE PRECISION,
    acquired_at       TIMESTAMPTZ,
    event_id          TEXT,
    role              TEXT NOT NULL DEFAULT 'post',
    source_version    TEXT DEFAULT '1.0',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_hg_imagery_layer_file UNIQUE (image_layer_id, file_uri)
);

CREATE INDEX IF NOT EXISTS hg_imagery_footprint_gix
    ON hg_imagery USING GIST (footprint);

CREATE INDEX IF NOT EXISTS hg_imagery_image_layer_id_idx
    ON hg_imagery (image_layer_id);

-- 2. AOI Assessments
CREATE TABLE IF NOT EXISTS hg_assessments (
    assessment_id         TEXT PRIMARY KEY,
    project_id            TEXT NOT NULL,
    image_layer_id        TEXT NOT NULL,
    event_id              TEXT,
    name                  TEXT,
    aoi_geometry          GEOMETRY(Geometry, 4326) NOT NULL,
    aoi_area_km2          DOUBLE PRECISION NOT NULL,
    inclusion_policy      TEXT NOT NULL DEFAULT 'intersects_aoi',
    status                TEXT NOT NULL DEFAULT 'submitted',
    config                JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_repo            TEXT,
    model_digest          TEXT,
    preprocessing_version TEXT DEFAULT '1.0.0',
    eligible_buildings    INTEGER,
    total_units           INTEGER,
    completed_units       INTEGER NOT NULL DEFAULT 0,
    failed_units          INTEGER NOT NULL DEFAULT 0,
    submitted_by          TEXT,
    error                 TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS hg_assessments_aoi_gix
    ON hg_assessments USING GIST (aoi_geometry);

CREATE INDEX IF NOT EXISTS hg_assessments_project_layer_idx
    ON hg_assessments (project_id, image_layer_id);

-- 3. Batches
CREATE TABLE IF NOT EXISTS hg_batches (
    batch_id          TEXT PRIMARY KEY,
    assessment_id     TEXT NOT NULL REFERENCES hg_assessments(assessment_id) ON DELETE CASCADE,
    status            TEXT NOT NULL DEFAULT 'pending',
    attempts          INTEGER NOT NULL DEFAULT 0,
    runner_job_id     TEXT,
    runner_task_id    TEXT,
    lease_expires_at  TIMESTAMPTZ,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS hg_batches_assessment_status_idx
    ON hg_batches (assessment_id, status);

-- 4. Processing Units (1024x1024 tile windows)
CREATE TABLE IF NOT EXISTS hg_processing_units (
    unit_id               TEXT PRIMARY KEY,
    assessment_id         TEXT NOT NULL REFERENCES hg_assessments(assessment_id) ON DELETE CASCADE,
    imagery_id            TEXT NOT NULL REFERENCES hg_imagery(imagery_id) ON DELETE CASCADE,
    batch_id              TEXT REFERENCES hg_batches(batch_id) ON DELETE SET NULL,
    col_off               INTEGER NOT NULL,
    row_off               INTEGER NOT NULL,
    width                 INTEGER NOT NULL,
    height                INTEGER NOT NULL,
    bounds                GEOMETRY(Polygon, 4326) NOT NULL,
    kind                  TEXT NOT NULL DEFAULT 'grid', -- 'grid' | 'building_centred'
    status                TEXT NOT NULL DEFAULT 'pending',
    attempts              INTEGER NOT NULL DEFAULT 0,
    valid_pixel_fraction  DOUBLE PRECISION,
    error                 TEXT
);

CREATE INDEX IF NOT EXISTS hg_processing_units_bounds_gix
    ON hg_processing_units USING GIST (bounds);

CREATE INDEX IF NOT EXISTS hg_processing_units_assessment_idx
    ON hg_processing_units (assessment_id, status);

CREATE INDEX IF NOT EXISTS hg_processing_units_batch_idx
    ON hg_processing_units (batch_id);

-- 5. Unit to Building Association
CREATE TABLE IF NOT EXISTS hg_unit_buildings (
    unit_id            TEXT NOT NULL REFERENCES hg_processing_units(unit_id) ON DELETE CASCADE,
    building_id        TEXT NOT NULL,
    coverage_fraction  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (unit_id, building_id)
);

CREATE INDEX IF NOT EXISTS hg_unit_buildings_building_idx
    ON hg_unit_buildings (building_id);

-- 6. Prediction Evidence
CREATE TABLE IF NOT EXISTS hg_prediction_evidence (
    assessment_id      TEXT NOT NULL REFERENCES hg_assessments(assessment_id) ON DELETE CASCADE,
    building_id        TEXT NOT NULL,
    unit_id            TEXT NOT NULL REFERENCES hg_processing_units(unit_id) ON DELETE CASCADE,
    probs              DOUBLE PRECISION[] NOT NULL,
    class              INTEGER NOT NULL,
    confidence         DOUBLE PRECISION NOT NULL,
    coverage_fraction  DOUBLE PRECISION NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (assessment_id, building_id, unit_id)
);

CREATE INDEX IF NOT EXISTS hg_prediction_evidence_assessment_bldg_idx
    ON hg_prediction_evidence (assessment_id, building_id);

-- 7. Aggregated Damage Assessments
CREATE TABLE IF NOT EXISTS hg_damage_assessments (
    assessment_id          TEXT NOT NULL REFERENCES hg_assessments(assessment_id) ON DELETE CASCADE,
    building_id            TEXT NOT NULL,
    image_layer_id         TEXT NOT NULL,
    damage_class           INTEGER,
    damage_class_name      TEXT,
    confidence             DOUBLE PRECISION,
    damage_percentage      DOUBLE PRECISION, -- Always NULL for this model
    result_status          TEXT NOT NULL DEFAULT 'completed', -- completed | uncertain | insufficient_data | failed
    boundary_intersecting  BOOLEAN NOT NULL DEFAULT FALSE,
    aoi_overlap_fraction   DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    winning_unit_id        TEXT,
    model_digest           TEXT,
    preprocessing_version  TEXT,
    footprint_version      TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (assessment_id, building_id)
);

CREATE INDEX IF NOT EXISTS hg_damage_assessments_layer_bldg_idx
    ON hg_damage_assessments (image_layer_id, building_id);

CREATE INDEX IF NOT EXISTS hg_damage_assessments_status_idx
    ON hg_damage_assessments (assessment_id, result_status);

-- 8. Alter existing hg_buildings to support footprint source versioning
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS footprint_source TEXT;
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS footprint_version TEXT;

-- 9. Native raster geotransform for each catalogued TIF (GDAL order:
--    a, b, c, d, e, f of the rasterio Affine). Window bounds are computed in
--    the raster's own CRS from this, then reprojected to EPSG:4326 — the
--    4326 footprint alone cannot reproduce pixel positions for projected
--    (e.g. UTM) imagery.
ALTER TABLE hg_imagery ADD COLUMN IF NOT EXISTS transform DOUBLE PRECISION[];

-- 10. Share of a building's mask pixels that were valid (non-nodata) in the
--     window that produced this evidence row.
ALTER TABLE hg_prediction_evidence ADD COLUMN IF NOT EXISTS valid_pixel_fraction DOUBLE PRECISION;
