-- hg_buildings is the single "current damage" table per building: its
-- geometry is the original footprint and its damage_* columns always hold
-- the latest successful assessment. These columns record which run wrote it.
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS last_assessment_id TEXT;
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS result_status TEXT;

-- Spatial index over assessed buildings only. Serves the viewport query
-- (bbox + damage_class IS NOT NULL) without scanning the whole layer.
DROP INDEX IF EXISTS hg_buildings_damaged_idx;
CREATE INDEX IF NOT EXISTS hg_buildings_damaged_gix
    ON hg_buildings USING GIST (geometry)
    WHERE damage_class IS NOT NULL;
