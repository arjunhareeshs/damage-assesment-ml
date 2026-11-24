-- hg_imagery.footprint is the outline of a raster's valid (non-nodata)
-- pixels, which can be several disjoint areas, so it is any geometry type
-- rather than only a Polygon. hastegeo applies this on first use too
-- (imagery_catalog._ensure_columns).
ALTER TABLE hg_imagery
    ALTER COLUMN footprint TYPE geometry(Geometry, 4326) USING footprint;

-- Which footprint dataset (and version) each building row came from.
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS footprint_source TEXT;
ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS footprint_version TEXT;
