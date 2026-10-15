# Repository Audit: Large-AOI Batched Building Damage Assessment

**Audit Date:** 2026-10-12  
**Target Project:** HASTE ↔ DamageTriageFormer Integration  
**Scope:** Complete repository audit per `docs/see-this-is-the-lexical-goose.md` (§9)

---

## 1. Executive Summary

This repository audit provides a complete structural, architectural, and operational map of both codebases:
1. **HASTE** (Hierarchical Annotation & Satellite Triage Engine): A scalable geospatial platform built by Microsoft / AI for Good Lab, composed of an Azure Functions API (`hastefuncapi`), queue worker (`hastefuncqueues`), task runners (`LocalRunner`, `AzureBatchRunner`), PostGIS database, TiTiler tile server, and a React frontend utilizing the Azure Maps Web SDK.
2. **DamageTriageFormer**: A state-of-the-art vision transformer architecture for building damage classification based on multi-scale feature hierarchies and masked building attention.

The goal of this integration is the **Large-AOI Batched Assessment Workflow**: enabling users to define arbitrary polygonal Areas of Interest (AOIs) spanning multiple square kilometers and multiple aerial imagery tiles (e.g. NAIP GeoTIFFs), batching inference across available compute resources, associating building footprints, tracking multi-unit evidence, and presenting aggregated results interactively in Azure Maps.

---

## 2. Codebase Structure & Directory Layout

### 2.1 Top-Level Organization
```
c:/SSG projects/dmgassesment/
├── Damage-TriageFormer/         # Upstream DTF training repo (READ-ONLY, CC-BY-NC-4.0)
│   ├── configs/                 # Model configs (convnext, swin, vit)
│   ├── src/                     # PyTorch model definitions, losses, datasets
│   └── tools/                   # Evaluation and inference scripts
├── external/damage-triage-former/ # Submodule mirror (READ-ONLY)
├── i-dtf/                       # Interactive DTF playground (READ-ONLY)
├── haste/                       # HASTE root
│   ├── docs/                    # Technical specifications and advancement docs
│   └── Haste-DTF/haste-fork/   # Active working fork (MODIFICATIONS HERE ONLY)
│       ├── api/                 # Azure Functions v4 (hastefuncapi, hastefuncqueues, titilerfuncapi)
│       ├── db/migrations/       # PostgreSQL / PostGIS SQL migration scripts
│       ├── docker/              # Docker Compose and container definitions
│       ├── hastelib/            # Python core library (hastegeo) & CLI tools
│       │   ├── src/hastegeo/    # Core logic, runners, models, workflows, utils
│       │   └── tests/           # Unit and integration test suites
│       └── ui/                  # React + Vite frontend using Azure Maps Web SDK
```

### 2.2 Invariant Constraints
- **Strict Read-Only Directories:** `Damage-TriageFormer/`, `external/damage-triage-former/`, and `i-dtf/` are strictly unmodified.
- **Migration Invariant:** Existing migrations `0000_init_db.sql` and `0001_dtf_support.sql` are preserved unmodified; all schema extensions reside in `0002_aoi_assessments.sql`.
- **Workflow Invariant:** Legacy workflows (`dtf` whole-layer and `dtf-aoi` single-mosaic) remain completely functional and unaltered.
- **Mapping SDK:** Exclusively Azure Maps Web SDK (`atlas`) and `azure-maps-drawing-tools`; no Mapbox dependencies.

---

## 3. Model Location & Loading Mechanism

### 3.1 Model Checkpoint Storage
- Model weights reside in the host filesystem under `models/` or mounted into Docker containers at `/app/models/`.
- Pretrained weights for DamageTriageFormer include:
  - ConvNeXt-Tiny / Small / Base backbones
  - Swin-Transformer backbones
  - ViT backbones
- Checkpoint file format: PyTorch `.pth` or `.pt` state dictionaries.

### 3.2 Loading Pipeline in HASTE
- Located in `hastelib/src/hastegeo/workflows/damage_triage_infer.py`:
  - `load_model(checkpoint_path, model_name, device)` instantiates the architecture matching `model_name` and loads weights via `torch.load(..., map_location=device)`.
  - Checkpoint state dictionaries are loaded using `strict=False` or key prefix stripping (`module.` or `backbone.`) to handle distributed training artifacts.
  - The model is placed in `.eval()` mode with gradient computation disabled (`torch.no_grad()`).
- Reusable Window Scorer:
  - Extracted as `score_window(model, raster, window, geoms, ...)` in `damage_triage_infer.py`.
  - Invoked directly by the sibling batch container `damage_triage_region_infer.py`.
  - Evaluates pre- and post-event window chips with building binary masks, feeding batches of shape `(B, C, H, W)` into the model.

---

## 4. Data Directory & Storage Layout

### 4.1 Host vs Container Paths
- Data files (NAIP GeoTIFFs, building footprint GeoJSON/GPKG, and job outputs) reside on the host in `data/` or `haste/data/`.
- Mounted in `docker-compose.yml`:
  - Azure Functions / LocalRunner: host path mapped via `LOCAL_STORAGE_MOUNT_PATH` (defaults to `/mnt/storage` or `./storage`).
  - `haste-training` container: `docker/training/` mounts `/mnt/storage` for input TIFs and outputs.
  - Sibling container execution in `LocalRunner`: mounts host volume `/mnt/storage` into each spawned task container.

### 4.2 Output Artifacts
- Legacy workflows write to `<storage>/jobs/<job_id>/outputs/`.
- Large-AOI batched tasks write to `<storage>/jobs/<job_id>/outputs/`:
  - `evidence.jsonl`: One JSON line per evaluated `(unit_id, building_id)` with `probs`, `class`, `confidence`, `coverage_fraction`, and `valid_pixel_fraction`.
  - `region_manifest.json`: Execution metadata, timing, device info, unit counts, and any skipped units.

---

## 5. Footprint Ingestion & Storage Architecture

### 5.1 Ingestion Sources
1. **Overture Maps Foundation:**
   - Downloader queries Azure Blob Storage (`az://overturemaps-us-west-2/release/...`) using parquet partitioning by bounding box.
   - Clipped to the requested AOI.
2. **Microsoft Building Footprints (User Upload):**
   - User-supplied GeoJSON or GeoPackage files.
   - Streamed via `postgis_load.load_footprints_streaming()` in chunks of 5,000 to prevent OOM errors.

### 5.2 Canonical ID Strategy
- To deduplicate footprints across multiple TIF boundaries and overlapping downloads:
  $$\text{building\_id} = \text{sha1}(\text{footprint\_source} \parallel "|" \parallel \text{rounded\_wkb})$$
  where coordinates are rounded to 7 decimal places (~1.1 cm precision).
- Preserves user ID if present and valid; otherwise applies the canonical SHA-1 digest.

### 5.3 Database Storage (`hg_buildings`)
- Stored with `geometry geometry(MultiPolygon, 4326)`.
- Spatial index: `CREATE INDEX idx_buildings_geom ON hg_buildings USING GIST (geometry)`.
- Columns added in `0002_aoi_assessments.sql`:
  - `footprint_source VARCHAR(64) DEFAULT 'user'`
  - `footprint_version VARCHAR(32)`

---

## 6. Database Schema Additions (`0002_aoi_assessments.sql`)

The migration introduces seven dedicated tables isolated from legacy workflow schemas:

1. **`hg_imagery`**:
   - Stores cataloged GeoTIFF headers (`imagery_id`, `path`, `bounds`, `crs`, `width`, `height`, `pixel_size_x`, `pixel_size_y`, `nodata_value`, `footprint`).
   - Indexed spatially on `footprint` using GIST.

2. **`hg_assessments`**:
   - Master assessment tracking (`assessment_id`, `name`, `status`, `aoi_geometry`, `aoi_area_km2`, `config`, timestamps, error message).
   - Status transitions: `submitted` → `planning` → `processing` → `aggregating` → `completed` (or `failed` / `cancelled`).

3. **`hg_processing_units`**:
   - Individual evaluation tiles (`unit_id`, `assessment_id`, `imagery_id`, `col_off`, `row_off`, `width`, `height`, `bounds`, `kind`, `status`, `assigned_batch_id`, `lease_expires_at`).
   - `kind`: `'grid'` (regular tiling) or `'building_centred'` (dedicated tile for partial coverage).

4. **`hg_unit_buildings`**:
   - Many-to-many relationship mapping `(unit_id, building_id)` with `coverage_fraction`.

5. **`hg_batches`**:
   - Grouped units dispatched to runner tasks (`batch_id`, `assessment_id`, `imagery_id`, `status`, `runner_job_id`, `attempt_count`).

6. **`hg_prediction_evidence`**:
   - Raw model outputs per building observation (`evidence_id`, `assessment_id`, `unit_id`, `building_id`, `probs`, `class`, `confidence`, `coverage_fraction`, `valid_pixel_fraction`).

7. **`hg_damage_assessments`**:
   - Final aggregated building assessments (`assessment_id`, `building_id`, `damage_class`, `confidence`, `damage_probabilities`, `damage_percentage` [NULL], `result_status`, `boundary_intersecting`, `primary_unit_id`).

---

## 7. Gaps, Unknowns, and Technical Debt Identified

1. **Image Normalization Discrepancy (Resolved):**
   - DTF training uses ImageNet mean/std normalization: `(img/255.0 - mean) / std`.
   - Legacy HASTE used raw unit normalization: `img / 255.0`.
   - **Resolution:** Implemented `dtf_preprocess.py` supporting both `imagenet` and `unit`. The new region assessment workflow uses `imagenet` by default (`HASTE_DTF_NORMALIZATION=imagenet`), ensuring maximum model accuracy while preserving backward compatibility for legacy callers.

2. **Docker vs Host Environment Differences:**
   - Host Windows Anaconda environment lacks `fiona` and `boto3`.
   - Container environments (`haste-training`, `haste-imageryprep`) contain full GDAL, fiona, and boto3 stacks.
   - Code and tests use `pyogrio` and conditional skips so development and unit tests execute cleanly on both host and inside containers.

3. **Database Volume Initialization:**
   - PostGIS container mounts `db/migrations/` to `/docker-entrypoint-initdb.d/`.
   - Pre-existing Docker volumes do not re-run entrypoint scripts. Migration `0002_aoi_assessments.sql` must be applied explicitly if the volume already exists.

---

## 8. Conclusion
The repository audit confirms that all requirements of `docs/see-this-is-the-lexical-goose.md` are accounted for, with clean separation of concerns, strict backward compatibility, and complete isolation of the new Large-AOI pipeline.
