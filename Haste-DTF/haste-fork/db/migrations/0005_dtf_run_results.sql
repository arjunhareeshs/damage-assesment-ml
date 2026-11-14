-- Per-run DamageTriageFormer results (AOI runs).
--
-- hg_buildings keeps only each building's CURRENT result, so a later run
-- overwrites an earlier one. This table keeps every run's result for every
-- building it scored -- the history behind hg_buildings -- keyed by the run
-- (model_id). hastegeo also creates it on first use (postgis_query.
-- ensure_run_results_table) for databases created before this migration.
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
