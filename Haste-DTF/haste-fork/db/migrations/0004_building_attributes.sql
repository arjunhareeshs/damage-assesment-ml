-- Optional per-building source attributes (e.g. TWDB use type, population, SVI).
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS attributes JSONB;
