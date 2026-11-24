-- Analyst review of large-AOI assessment results.
--
-- Every completed building result starts as 'pending_review'; an analyst
-- accepts or rejects it (or HASTE_REQUIRE_REVIEW=false auto-accepts at
-- aggregation). hg_buildings.review_status mirrors the decision for the
-- assessment that last published the building, and rejected results are never
-- published. hastegeo also applies these on first use
-- (assessment_store.ensure_review_schema) for databases that already exist.
ALTER TABLE hg_damage_assessments
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'pending_review';
ALTER TABLE hg_damage_assessments ADD COLUMN IF NOT EXISTS reviewed_by TEXT;
ALTER TABLE hg_damage_assessments ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS hg_damage_assessments_review_idx
    ON hg_damage_assessments (assessment_id, review_status);
