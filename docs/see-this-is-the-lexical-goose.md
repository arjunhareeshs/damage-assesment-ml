# Handoff brief for the implementing agent: Large-AOI batched damage assessment

You are implementing a new AOI-first assessment workflow in an existing repository. Read this whole file before touching code. Everything stated as "existing" was verified by reading the source; everything marked **NEW** is yours to create. If something you find in the repo contradicts this file, stop and report it instead of guessing.

Repo root is `haste/` (git root). All paths below are relative to it. `HF` means `Haste-DTF/haste-fork`.

---

## 1. Goal

Today an analyst draws a rectangle inside one image layer, and the `dtf-aoi` job scores one building per forward pass and overwrites that building's row. Build a second, parallel workflow:

draw AOI (small inside one TIF, or very large across many TIFs) → find the TIFs and buildings the AOI touches → score in bounded 1024-px tile batches → store one result per building per assessment → the map only reads stored results.

The existing `dtf` and `dtf-aoi` workflows must keep working unchanged.

## 2. Decisions already made (do not reopen)

| Topic | Decision |
|---|---|
| Map library | **Azure Maps**, as the UI already uses. Do **not** add Mapbox or any other map SDK. |
| Runtime | The Docker stack in `HF/docker/docker-compose.yml` (Azure Functions API, Azurite queues, PostGIS, `haste-training` task container). |
| Imagery | Many source TIFs per image layer, **post-event only**. No statewide mosaic. |
| Pre/post pairing | Not required by this model. Store `role='post'`; no pairing logic. |
| Damage percentage | The model outputs class probabilities only. `damage_percentage` is always `NULL`. |
| Building inclusion | A building is eligible if its footprint **intersects** the AOI polygon. |
| Data location | `data/` and `models/` exist only on the user's machine and are gitignored. Never commit them, never assume their filenames. |

## 3. Facts about the existing code you must respect

### 3.1 Model contract (do not change)
- Loader: `build_dtf_model(checkpoint_path, config_path, device) -> (model, use_gated)` in `HF/hastelib/src/hastegeo/workflows/damage_triage_infer.py:149`.
- Forward: `_, damage_out = model(tile_t, [masks_t])` where `tile_t` is `(1, 3, 1024, 1024)` float and `masks_t` is `(N, 1024, 1024)` float, one binary mask per building.
- Decode: `decode_gated_predictions(damage_out) -> (N, 5)` (:252), or `decode_severity_predictions` (:281) when `use_gated` is false.
- Tile size: `DTF_IM_SIZE = 1024` (:123). Import it. It is a model constraint, not a tunable.
- Classes: `DTF_CATEGORIES` (:116). Index order is fixed by the checkpoint:
  `0 Undamaged`, `1 Partial Roof Damage`, `2 Total Roof Damage`, `3 Partial Structural Collapse`, `4 Total Structural Collapse`. Import the list; never retype it. `UNCLASSIFIED = -1`.
- Imagery is read as bands 1,2,3 uint8 RGB (`read_tile`, :294).

### 3.2 Reusable functions (reuse, do not reimplement)
| Function | File | Use for |
|---|---|---|
| `compute_tile_windows(width, height, tile_size)` | `damage_triage_infer.py:204` | Grid windows over a TIF |
| `rasterize_instance_masks(footprints, window, transform, tile_size)` | same, :217 | Per-building masks for a window |
| `read_tile(raster, window, tile_size)` | same, :294 | Windowed read with zero-pad at edges |
| `crop_building`, `crop_building_mask`, `_reproject_geometry` | `HF/hastelib/src/hastegeo/core/utils/crop_service.py` | Building-centred window maths, reprojection |
| `get_connection()` | `core/utils/postgis_connect.py:34` | Every DB connection |
| `AzureQueueHandler`, `UnifiedRunner`, `UnifiedDataLayer`, `MetadataUtils` | `core/utils/queues.py`, `core/runners/unified_runner.py`, `core/data_layer/unified.py`, `core/utils/metadata.py` | Queue, task execution, storage, IDs/timestamps |
| `Logger.get_logger` | `core/utils/logs.py` | Logging |

### 3.3 Patterns to copy
- **Job state machine:** `HF/hastelib/src/hastegeo/core/processors/damage_triage_aoi.py`. A Preprocessor enqueues; a Postprocessor handles one status per queue message and re-enqueues itself while work is running.
- **Queue trigger:** `HF/api/hastefuncqueues/function_app.py:696-798`.
- **HTTP route:** `HF/api/hastefuncapi/function_app.py:2440-2579` (decorator, `auth_level=AUTH_LEVEL`, `asyncio.to_thread`, error handling).
- **DB functions:** `core/utils/postgis_query.py`. Every function takes `conn=None`, opens/closes its own connection only when none is passed, and uses parameterised SQL.
- **Tests:** `HF/hastelib/tests/core/utils/test_postgis_query.py` and `tests/core/processors/test_damage_triage_aoi.py` (mocked connection and patched runner/queue).
- **Task container CLI:** `[project.scripts]` in `HF/hastelib/pyproject.toml:65-69` plus the shell shim in `HF/docker/training/Dockerfile:80-86`.

### 3.4 Known defect you must handle
Training normalises as `(img/255 - mean) / std` with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` (`Haste-DTF/Damage-TriageFormer/src/data/dataset.py:9-10,161,183`). The two existing HASTE workflows only do `/255` (`damage_triage_infer.py:439`, `damage_triage_aoi_infer.py:257`). The new workflow must use ImageNet normalisation. **Do not change the two existing workflows' behaviour**; add the option there but leave their default as it is and report the difference to the user.

## 4. Hard rules

### Never
1. Edit anything under `Haste-DTF/Damage-TriageFormer/`, `HF/external/damage-triage-former/` or `i-dtf/`.
2. Edit `HF/db/migrations/0000_*.sql` or `0001_*.sql`. Add a new numbered file.
3. Change behaviour, signature or response shape of any existing route, queue trigger, processor, workflow or DB function. Refactors must keep outputs identical.
4. Read a whole raster into memory. Every read passes `window=`. (`core/utils/aoi.py:extract_aoi_polygon` does `f.read()` on the full file; do not use it on source TIFs.)
5. Load the whole footprint file with `gpd.read_file(path)` in new code. Stream it.
6. Build a mosaic of all TIFs.
7. Write `Undamaged` (class 0) for a building that was not scored. Use a status instead.
8. Compute, estimate or fill `damage_percentage`.
9. Invent building IDs, counts, confidences or imagery metadata. Unknown counts are `null`.
10. Start inference from a map click or from any GET route.
11. Use string formatting to build SQL. Parameters only.
12. Run `docker system prune --volumes`, drop the PostGIS volume, or delete task directories.
13. Commit `data/`, `models/`, `.env`, weights, TIFs, GeoJSON or GeoPackages.
14. Add a new web framework, ORM, task queue, or map library.

### Do not hardcode
| Thing | Get it from |
|---|---|
| Imagery filenames or paths (e.g. the Texas NAIP tile names in `run_texas_assessment.py`) | `hg_imagery.file_uri` |
| Developer paths (the `"Damage assesment/damage-assesment-ml"` candidate in the existing checkpoint lookup is a bad example; do not copy it) | env vars |
| Class names, class count, colours | `DTF_CATEGORIES`, `DTF_IDX_TO_COLOR` on the backend; one shared constant on the frontend |
| Tile size | `DTF_IM_SIZE` |
| Queue names, DB host/user/password, ports, storage URLs | `Config` / env vars, as existing code does |
| Hugging Face repo IDs, checkpoint paths | existing env vars `DTF_CHECKPOINT_PATH`, `DTF_CHECKPOINT_CONFIG_PATH`, `DTF_BACKBONE_DIR` |
| Limits, batch sizes, thresholds, retry counts | the parameters in section 6 |
| Project, layer, assessment or building IDs | request / DB |
| "Texas", any state, any bounding box | nothing; the code is region-agnostic |

Allowed constants: storage CRS `EPSG:4326` (matches `hg_buildings`), area CRS `EPSG:6933` (already used at `damage_triage_infer.py:502`). Define each once in one module and import it.

### Stop and ask the user when
- The AOI area limit, batch size or concurrency needs a production value (use the env parameter, do not pick a number in code).
- A source TIF is not 3-band uint8 RGB.
- `prepare_imagery.py`, `core/config.py`, `core/runners/azure_batch.py` or the API auth code does not match the patterns described here.
- You are about to change an existing default.

## 5. What to build

### 5.1 Database: **NEW** `HF/db/migrations/0002_aoi_assessments.sql`
All statements idempotent (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`). Geometry columns are SRID 4326 with GIST indexes.

| Table | Key columns |
|---|---|
| `hg_imagery` | `imagery_id` PK, `project_id`, `image_layer_id`, `file_uri`, `footprint` Polygon, `crs`, `width`, `height`, `pixel_size_x`, `pixel_size_y`, `band_count`, `dtype`, `nodata`, `acquired_at` (nullable), `event_id` (nullable), `role` (`'post'`), `source_version`, `created_at`. Unique `(image_layer_id, file_uri)`. |
| `hg_assessments` | `assessment_id` PK, `project_id`, `image_layer_id`, `event_id` (nullable), `aoi_geometry` (never updated after insert), `aoi_area_km2`, `inclusion_policy`, `status`, `config` jsonb (resolved section-6 parameters), `model_repo`, `model_digest` (sha256 of the checkpoint file), `preprocessing_version`, `eligible_buildings`, `total_units`, `completed_units`, `failed_units`, `submitted_by`, `error`, `created_at`, `updated_at`, `completed_at`. |
| `hg_processing_units` | `unit_id` PK, `assessment_id` FK, `imagery_id` FK, `batch_id` (nullable), `col_off`, `row_off`, `width`, `height`, `bounds` Polygon, `kind` (`grid` / `building_centred`), `status`, `attempts`, `valid_pixel_fraction`, `error`. |
| `hg_unit_buildings` | PK `(unit_id, building_id)`, `coverage_fraction`. |
| `hg_batches` | `batch_id` PK, `assessment_id` FK, `status`, `attempts`, `runner_job_id`, `runner_task_id`, `lease_expires_at`, `error`, timestamps. |
| `hg_prediction_evidence` | PK `(assessment_id, building_id, unit_id)`, `probs` float8[], `class`, `confidence`, `coverage_fraction`, `created_at`. |
| `hg_damage_assessments` | PK `(assessment_id, building_id)`, `image_layer_id`, `damage_class` (nullable int), `damage_class_name` (nullable), `confidence` (nullable), `damage_percentage` (nullable, always NULL), `result_status`, `boundary_intersecting` bool, `aoi_overlap_fraction`, `winning_unit_id`, `model_digest`, `preprocessing_version`, `footprint_version`, timestamps. |
| `hg_buildings` (existing) | add nullable `footprint_source`, `footprint_version` only. |

PostGIS init scripts run only on first container start. Document the manual apply command for an existing volume in the audit doc (`docker compose exec postgis psql -U haste -d haste -f ...`); do not reset the volume.

### 5.2 Status values (exact strings)
- Assessment: `submitted`, `validating`, `rejected`, `discovering`, `tiling`, `queued`, `running`, `aggregating`, `completed`, `completed_with_errors`, `failed`, `cancelling`, `cancelled`.
- Unit and batch: `pending`, `running`, `completed`, `failed`, `skipped`.
- Building result: `completed`, `uncertain`, `insufficient_data`, `failed`.
Define them once as enums in `core/models/projects.py` and import everywhere, including tests.

### 5.3 Backend modules (all **NEW**, under `HF/hastelib/src/hastegeo/`)

| Module | Functions | Notes |
|---|---|---|
| `core/utils/dtf_preprocess.py` | `normalize_tile(tile_uint8, mode) -> float32`, `PREPROCESSING_VERSION` | `mode` ∈ `imagenet`, `unit`; anything else raises `ValueError`. |
| `core/utils/aoi_validate.py` | `validate_aoi(geojson, max_area_km2) -> (shapely_geom, area_km2)` | Accept `Polygon`/`MultiPolygon` only. Reject invalid, empty, zero-area, out-of-range lon/lat, antimeridian-crossing, and over-limit geometries with a specific message each. Do not auto-repair. |
| `core/utils/imagery_catalog.py` | `register_imagery(image_layer, file_uri, local_path, conn=None)`, `imagery_for_aoi(image_layer_id, aoi_geojson, conn=None)` | Read the raster header only. Footprint = raster bounds reprojected to 4326. |
| `core/utils/postgis_load.py` (add function) | `load_footprints_streaming(image_layer, path, footprint_source, footprint_version, chunk_size, conn=None)` | Chunked read + `COPY` or batched insert. Leave `load_footprints_into_postgis` untouched. |
| `core/utils/region_windows.py` | `select_units(...)`, `building_coverage(...)`, `group_batches(units, windows_per_batch)` | Pure functions, no torch, no DB. |
| `core/utils/region_aggregate.py` | `aggregate(evidence_rows, config) -> final_rows` | Pure function. |
| `core/utils/assessment_store.py` | create/get/update assessment, insert units/batches, upsert evidence, upsert results, claim/renew/expire leases, statistics, bbox queries | `conn=None` pattern. |
| `core/processors/aoi_assessment.py` | `AoiAssessmentPreprocessor`, `AoiAssessmentPostprocessor` | Copy the shape of `damage_triage_aoi.py`. |
| `workflows/damage_triage_region_infer.py` | `main()` with `--config` | Runs inside the task container; no DB access. |

Also: add `AoiAssessment` pydantic model in `core/models/projects.py`; add `assessment_queue_name` to the queue config in `core/config.py` next to `dtf_aoi_queue_name`; add the CLI entry in `pyproject.toml` and the shim in the training `Dockerfile`.

**Canonical building ID.** Use the source file's ID field when it exists and is unique. Otherwise `sha1(footprint_source + "|" + WKB of the geometry with coordinates rounded to 7 decimals)`. Never use a row index.

### 5.4 Processing rules

1. **Imagery discovery:** `ST_Intersects(hg_imagery.footprint, aoi)` scoped to the image layer. No rows → assessment `failed`, error "no imagery covers this AOI".
2. **Building discovery:** `ST_Intersects(hg_buildings.geometry, aoi)` scoped to the image layer. Zero rows → `completed` with zero results (a valid outcome). Set `boundary_intersecting = NOT ST_Within(geometry, aoi)` and `aoi_overlap_fraction = area(intersection)/area(footprint)`.
3. **Unit selection, per TIF:** take `compute_tile_windows`; keep a window only if it intersects the AOI **and** contains at least one eligible building. `coverage_fraction` = share of the footprint area inside the window. If a building's best coverage across all units is below 1.0 and its bounding box fits in 1024 px, add one `building_centred` unit for it.
4. **A window is a compute unit, not a building list for assessment.** Rasterise masks only for eligible buildings. Other buildings in the tile are ignored.
5. **Batches:** group units of the same TIF, at most `windows_per_batch` each.
6. **Inference inside a unit:** if a window holds more buildings than `masks_per_forward`, split the masks across several forward passes of the same tile.
7. **OOM:** on `torch.cuda.OutOfMemoryError`, halve the mask chunk and retry; at chunk size 1, run that window on CPU; if that fails, mark the unit `failed` and continue.
8. **Valid data:** `valid_pixel_fraction` = share of non-nodata pixels under the building mask. Below `min_valid_pixel_fraction` → no evidence row; building ends as `insufficient_data`.
9. **Task output:** `outputs/evidence.jsonl`, one line `{unit_id, building_id, probs[5], class, confidence, coverage_fraction, valid_pixel_fraction}`, and `outputs/region_manifest.json` with per-unit status, counts, `model_digest`, `preprocessing_version`, and the checkpoint's missing/unexpected key counts.
10. **Checkpoint load check:** the existing loader uses `strict=False`. In the region workflow, fail the task if any missing key starts with `backbone.` or `damage_head.`.
11. **Evidence ingest:** `INSERT ... ON CONFLICT (assessment_id, building_id, unit_id) DO UPDATE`. Re-running a batch must not add rows.
12. **Aggregation, per `(assessment_id, building_id)`:** choose the evidence with the highest `coverage_fraction`; tie → highest `confidence`; tie → lowest `unit_id`. Never average. Never depend on arrival order.
13. **Result status:** evidence chosen → `completed`, or `uncertain` if `uncertain_confidence_threshold` is set and confidence is below it. No evidence because of nodata → `insufficient_data`. No evidence because every unit failed → `failed`. For the last two, `damage_class`, `damage_class_name` and `confidence` are `NULL`.
14. **Final assessment status:** all units completed → `completed`; some failed after `max_attempts` → `completed_with_errors`; discovery or every batch failed → `failed`.
15. **Leases:** a running batch has `lease_expires_at = now + lease_seconds`, renewed on each poll. An expired lease returns the batch to `pending` and increments `attempts`; at `max_attempts` it becomes `failed`.
16. **Concurrency:** never more than `max_concurrent_batches` batches in `running`.
17. **Cancel:** set `cancelling`; submit no new batches; stop in-flight tasks through the runner; then `cancelled`. Keep all rows.
18. **Version pinning:** compute `model_digest` and `preprocessing_version` once at submit and pass them in every batch config. A batch whose manifest reports a different digest is `failed`.
19. **History:** a new assessment never updates or deletes rows of another assessment, and never writes the prediction columns of `hg_buildings`.

### 5.5 API routes (**NEW**, in `HF/api/hastefuncapi/function_app.py`, existing naming style)

| Route | Method | Body / query | Returns |
|---|---|---|---|
| `PutAoiAssessment` | PUT | `{projectId, imageLayerId, aoi, eventId?, name?}` | `{assessmentId, status, eligibleBuildings: null, totalUnits: null, completedUnits: 0, failedUnits: 0}` |
| `GetAoiAssessment` | GET | `assessmentId` | status, counts, error, config, versions |
| `PutCancelAoiAssessment` | PUT | `{assessmentId}` | updated status |
| `GetAoiAssessmentPreview` | POST | `{imageLayerId, aoi}` | `{areaKm2, imageryCount, buildingCount}`; read-only, creates nothing |
| `GetAoiAssessmentBuildings` | GET | `assessmentId, bbox, page, pageSize` | paged results |
| `GetAoiAssessmentStatistics` | GET | `assessmentId` | counts per class and per result status |
| `GetAoiAssessmentMapData` | GET | `assessmentId, bbox, zoom` | below `map_feature_min_zoom`: grid aggregates; otherwise building features, capped at `map_max_features` with a `truncated` flag |
| `GetBuildingAssessmentHistory` | GET | `imageLayerId, buildingId` | that building's results across assessments |

Add the queue trigger for `assessment_queue_name` in `HF/api/hastefuncqueues/function_app.py`. Use the same auth level and the same project-access checks as the neighbouring routes; read them first. Validate every input; return 400 with a specific message on bad input.

### 5.6 UI (Azure Maps; under `HF/ui/src/Components/`)
- Add polygon AOI drawing. Today AOI mode is rectangle-only (`LabelingTool/LabelingToolHelper.js:16`, `AOI_DRAWING_MODE = "draw-rectangle"`). Keep the rectangle.
- Send the shape's geographic coordinates as GeoJSON. Never screen pixels or viewport bounds.
- Before submit, call `GetAoiAssessmentPreview` and show area, TIF count, building count.
- After submit, keep `assessmentId` in the URL and poll `GetAoiAssessment`. A refresh must restore the view.
- Results layer: call `GetAoiAssessmentMapData` on map move end. Styles for the five classes plus `insufficient_data`, `failed` and not-yet-processed.
- Building click → show the stored result. It must not call any `PutRun*` route.
- Use `apiGet/apiPut/apiPost` from `ui/src/util/api.js`. Do not call the local bridge at `127.0.0.1:8787`.
- Leave the existing "Run DamageTriageFormer on AOI" button and `DtfAoiReviewPanel` working.

## 6. Parameters

All **NEW**. Read env defaults in `core/config.py`; allow per-assessment override where marked; store the resolved values in `hg_assessments.config`. Values marked "set by user" must come from env with no numeric fallback in code: if unset, reject the submit with a clear error.

| Parameter | Env var | Per-assessment override | Default | Meaning |
|---|---|---|---|---|
| Queue name | `ASSESSMENT_QUEUE_NAME` | no | compose: `local-assessment-queue` | Assessment queue |
| Max AOI area (km²) | `HASTE_AOI_MAX_AREA_KM2` | no | set by user | Submit rejected above this |
| Max units per assessment | `HASTE_AOI_MAX_UNITS` | no | set by user | Rejected after tiling if exceeded |
| Normalisation | `HASTE_DTF_NORMALIZATION` | yes | `imagenet` | `imagenet` or `unit` |
| Windows per batch | `HASTE_REGION_WINDOWS_PER_BATCH` | yes | set by user after benchmark | Units in one runner task |
| Masks per forward | `HASTE_REGION_MASKS_PER_FORWARD` | yes | set by user after benchmark | Building masks per forward pass |
| Concurrent batches | `HASTE_REGION_MAX_CONCURRENT_BATCHES` | no | `1` | Runner tasks in flight |
| Max attempts | `HASTE_REGION_MAX_ATTEMPTS` | no | `3` | Per batch |
| Lease seconds | `HASTE_REGION_LEASE_SECONDS` | no | set by user after benchmark | Must exceed the longest expected batch |
| Min valid pixel fraction | `HASTE_REGION_MIN_VALID_FRACTION` | yes | `0.5` | Below → `insufficient_data` |
| Uncertain threshold | `HASTE_REGION_UNCERTAIN_CONFIDENCE` | yes | unset (disabled) | Below → `uncertain` |
| Inclusion policy | none | yes | `intersects_aoi` | Only value supported; reject others |
| Footprint chunk size | `HASTE_FOOTPRINT_LOAD_CHUNK` | no | `50000` | Rows per load chunk |
| Map max features | `HASTE_MAP_MAX_FEATURES` | no | `5000` | Cap per map-data response |
| Map feature min zoom | `HASTE_MAP_FEATURE_MIN_ZOOM` | no | `14` | Below → aggregates |

Add each env var to both `hastefuncapi` and `hastefuncqueues` in `docker-compose.yml`, and to `LocalRunner`'s passthrough list (`core/runners/local.py:321-330`) where the task container needs it.

## 7. Order of work

Finish and test each phase before starting the next. Run the existing test suite after every phase; it must stay green.

0. **Audit (no code).** Read `prepare_imagery.py`, `core/config.py`, `core/runners/local.py`, `core/runners/azure_batch.py`, the TiTiler setup and the API auth helpers. Write `docs/haste-advancement/REPOSITORY_AUDIT.md` and `docs/haste-advancement/IMPLEMENTATION_PLAN.md`, and copy the user's brief there. Report anything that conflicts with this file.
1. `dtf_preprocess.py`, `aoi_validate.py`, enums, migration `0002`, `assessment_store.py`, config entries. Unit tests.
2. Imagery catalogue, catalogue ingest mode in `prepare_imagery.py` (existing mosaic mode unchanged), streaming footprint loader, canonical ID.
3. `region_windows.py` with tests on synthetic geometries.
4. Extract the tile-scoring body of `run_damage_triage` into a shared function used by both it and the new workflow, with identical output for the old path. Then `damage_triage_region_infer.py`, `region_aggregate.py`, the processor, the queue trigger.
5. API routes.
6. UI.
7. End-to-end run on the user's data, recovery tests, benchmarks to set the "set by user" parameters.

After any change under `HF/hastelib/`, four images must be rebuilt (`hastefuncapi`, `hastefuncqueues`, `haste-training`, `haste-imageryprep`) and containers recreated with `docker compose up -d --build --force-recreate`.

## 8. Tests required

Command: `cd Haste-DTF/haste-fork/hastelib && PYTHONPATH=src pytest tests/`

- AOI validation: each rejection case; a valid polygon's coordinates are stored unchanged.
- A TIF outside the AOI produces no units.
- A TIF partly inside the AOI schedules only windows with eligible buildings; buildings outside the AOI get no result row.
- A boundary building is included, flagged, and keeps its ID.
- A building across two windows, and across two TIFs, ends with exactly one result row.
- Aggregation gives the same result for any ordering of evidence rows.
- Ingesting the same batch output twice leaves row counts unchanged.
- An expired lease is re-queued; `max_attempts` is enforced; partial failure yields `completed_with_errors`.
- A building under nodata ends `insufficient_data` with `damage_class` NULL.
- Result class names are exactly the five in `DTF_CATEGORIES`; `damage_percentage` is NULL in every row.
- A second assessment over the same AOI leaves the first assessment's rows untouched.
- Zero buildings in the AOI → `completed`, zero rows.
- Existing tests for `dtf-aoi` still pass unmodified.

Tests must not need the real checkpoint, real imagery, the network or a live database. Use small synthetic rasters and mocked connections.

## 9. Definition of done

- Every item in section 8 passes, plus the pre-existing suite.
- On the user's machine: a small AOI inside one TIF and an AOI spanning two TIFs both finish; `hg_damage_assessments` has one row per eligible building; a building click triggers no inference request (check the queue log).
- The old `dtf` and `dtf-aoi` flows behave as before.
- No value from the "do not hardcode" table appears as a literal in new code.
- The audit doc lists every assumption you made and every question still open.

## 10. Report back to the user

- The measured difference between `unit` and `imagenet` normalisation on the same buildings, so they can decide whether to switch the existing workflows.
- Measured seconds per window and per building on their hardware, with recommended values for the "set by user" parameters.
- The pixel size of their imagery. The checkpoint's training resolution is unconfirmed; do not resample silently.
- How the map will display many TIFs, based on what the TiTiler setup supports.
- Reminder: the model weights are CC-BY-NC-4.0 (non-commercial).
