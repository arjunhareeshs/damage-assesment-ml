-- Imagery updates: newer TIFs replace part of an image layer's imagery.
--
-- hg_imagery keeps every catalogued raster; an update deactivates the rows
-- it supersedes instead of deleting them. Large-AOI assessments only read
-- active rows.
--
-- hg_buildings.imagery_updated_at is when the imagery under a building last
-- changed. A damage result older than that (updated_at < imagery_updated_at)
-- was made on imagery the layer no longer shows, so it should be re-run.
--
-- hastegeo also applies these on first use (imagery_catalog._ensure_columns,
-- postgis_load._ensure_imagery_update_column), because existing databases
-- never re-run init scripts.
ALTER TABLE hg_imagery ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE hg_imagery ADD COLUMN IF NOT EXISTS retired_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS hg_imagery_active_idx ON hg_imagery (image_layer_id) WHERE active;

ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS imagery_updated_at TIMESTAMPTZ;
