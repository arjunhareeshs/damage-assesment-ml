**HASTE fork — Full Technical Audit**  
**Audited directory:** `Haste-DTF/haste-fork/`  
   
 **Companion report:** `docs/audit/IDTF_AUDIT.md` covers `i-dtf/` separately. The two stacks share only the read-only model architecture source in `external/damage-triage-former/src`; this report does not merge their workflows.  
  
---  
**0. Scope, naming, and method**  
The user's `haste-fork/` maps to `Haste-DTF/haste-fork/`. Everything under that directory is in scope: the Azure Functions API and queue workers, TiTiler, the `hastegeo` library (`hastelib/`), the external DamageTriageFormer source, the React UI, the PostGIS migrations, Docker/compose infrastructure, setup scripts, and tests. Runtime artifacts (`hastelib/logs/*.log`, `.pytest_cache/`) are noted but not treated as source.  
Each file entry answers the three requested questions:  
1. **Functionality** — what the file does.  
2. **Pipeline position** — where it sits in execution.  
3. **Data flow & dependencies** — what it consumes, who produces that input, what it emits, and who consumes the output.  
The per-directory architecture diagrams are in §18; the gap analysis is in §19.  
**Method and limits.** The audit is source-level: every file was read, cross-referenced against its callers, and checked against the shipped model/config metadata. Targeted runtime checks were limited to non-destructive inspections (Python import resolution, file/line verification, model-card and checkpoint key inspection); the full Docker stack, GPU inference and Azure Batch were **not** executed. Media and help assets are grouped by directory rather than itemised individually; every code, config, SQL, Docker and script file is named. Findings marked P0/P1 are supported by a concrete file/line reference.  
  
---  
**1. System overview**  
HASTE is a geospatial disaster-damage assessment application. A user creates a project, uploads pre- and post-event imagery, the system builds COGs/mosaics/footprints, and one of several model workflows produces damage predictions that are rendered with TiTiler on an Azure Maps front end.  
```  
Browser (React UI)  
    │ HTTPS/JSON, multipart  
    ▼  
 NGINX (local) ──► Azure Functions HTTP app (hastefuncapi)  
                      │ writes metadata + queue messages  
                      ▼  
                 Azure Queue / Azurite  
                      │ triggered  
                      ▼  
                 Azure Functions queue app (hastefuncqueues)  
                      │ processors write task configs, submit jobs  
                      ▼  
         UnifiedRunner ──► Azure Batch pools   (production)  
                      └─► sibling Docker containers (local)  
                               │  
                               ├─ haste-imageryprep : prepare-imagery  
                               └─ haste-training    : damage-triage-*, embed-buildings, segmentation inference  
                      │ outputs uploaded to blob  
                      ▼  
                 PostGIS hg_* tables  +  blob artifacts  
                      │  
                      ▼  
      hastefuncapi read APIs ──► UI map / reports / downloads  
                      │  
                      └─► TiTiler (COG tiles)  
   
```  
**Services and ports (local compose)**  
  
|  |  |  |  |  
| - | - | - | - |  
| **Service** |  **Image** |  **Port** |  **Role** |  
| `azurite` |  `haste-azurite` |  10000/10001/10002 |  Blob, queue, table emulator |  
| `data-init` |  `haste-data-init` |  — |  Seeds containers/queues/config; copies model weights into the Azurite volume |  
| `api-proxy` |  nginx |  7071→80 |  CORS proxy to Functions and TiTiler |  
| `titiler` |  `haste-titiler` |  8000 |  COG tile server |  
| `postgis` |  `postgis/postgis:16-3.4` |  5433 (host, localhost) |  `hg_*` tables |  
| `hastefuncapi` |  `haste-api` |  internal 8080 |  HTTP API |  
| `hastefuncqueues` |  `haste-queues` |  internal 8080 |  9 queue triggers |  
| `ui` |  `haste-ui` |  4280 |  React app |  
| `haste-training` |  task image |  — |  DTF/embedding/inference containers |  
| `haste-imageryprep` |  task image |  — |  Imagery preparation containers |  
  
   
**Three damage pipelines (do not conflate)**  
1. **Whole-layer DTF** (`modelType="dtf"`): tiles the layer's post-event mosaic, rasterizes every footprint per tile, keeps the best-confidence result per building, writes a predictions GPKG + two map rasters.  
2. **AOI-scoped DTF** (`modelType="dtf-aoi"`): the analyst draws an AOI or selects buildings; PostGIS resolves the shortlist; one forward pass per building; predictions enter a review workflow (`pending_review` → `accepted`/`rejected`).  
3. **Large-AOI assessment** (`hg_assessments`): a durable PostGIS state machine plans 1024 px units, groups them into batches, runs `damage-triage-region-infer` per batch, ingests evidence, aggregates, publishes "current damage", and exports GeoParquet.  
  
---  
**2. Entry points**  
  
|  |  |  |  
| - | - | - |  
| **Entry** |  **Trigger** |  **First code** |  
| UI |  user navigation/actions |  `ui/src/main.jsx` → `App.jsx` → `AppBody.jsx` routes |  
| HTTP API |  Function HTTP trigger |  `api/hastefuncapi/function_app.py` |  
| Queue processing |  Azure Queue message |  `api/hastefuncqueues/function_app.py` |  
| Tile requests |  HTTP |  `api/titilerfuncapi/app/__init__.py` |  
| Task execution |  container command |  `hastegeo.workflows.*` console scripts |  
| Local build |  Docker/compose |  `docker/*` + `hastelib/haste_build.py` |  
  
   
  
---  
**3. API —** `api/`  
**3.1** `api/__init__.py`  
- **Functionality:** package marker for the `api` tree.  
- **Pipeline position:** none at runtime.  
- **Data flow:** none.  
**3.2** `api/hastefuncapi/function_app.py` **(\~5,292 lines)**  
- **Functionality:** the entire HTTP surface, implemented as Azure Functions Python v2 decorators. It covers:  
  - **Identity/auth:** `ValidateUser`; `DEVELOPMENT_MODE` bypass (anonymous identity); GUID/email/short-integer identifier validation; role gating (`administrators`, `contributors`); Azure Maps token issuance.  
  - **Projects:** dashboard/project details, create/update/delete project, stats.  
  - **Image layers:** create/update/delete layer, layer detail view, downloadable URLs, chunked upload (`UploadFileByChunk`), imagery update submission, layer buildings.  
  - **Models:** create/update/delete model; catalog CRUD; enqueue embedding, whole-layer DTF, AOI DTF; cancel a queued/running model job; model results and visualizer URLs.  
  - **Labels/validation:** save labels from the labeling tool, interactive labels, building validation, validation report.  
  - **PostGIS-backed building APIs:** `GetBuildingsInAoi`, `GetBuildingEmbeddingsGeoJSON`, `GetHgBuildingsGeoJSON`, `PutBuildingPredictions`, `PutBuildingDTFReview`, `GetAoiPredictionsGpkg`, `GetCurrentBuildingDamage` (GeoJSON/GeoParquet), `GetBuildingAssessmentHistory`.  
  - **Large-AOI assessments:** `PutAoiAssessment`, `GetAoiAssessment`, `PutCancelAoiAssessment`, `GetAoiAssessmentPreview`, `GetAoiAssessmentBuildings`, `GetAoiAssessmentStatistics`, `GetAoiAssessmentMapData`. Validation includes EPSG:6933 area, duplicate-active assessment detection, `canSubmit` gating (422 otherwise), frozen config, `ast-<id>` identifiers, and queue enqueue.  
  - **Admin:** admin settings, users, invitations, source types, Azure Maps token.  
- **Pipeline position:** first backend hop for every UI action. It writes metadata through `MetadataProcessor`, puts messages on the queues, and reads PostGIS for map/report endpoints. It never runs inference synchronously.  
- **Data flow & dependencies:**  
  - **Inputs:** HTTP requests (JSON/multipart), the caller's identity, `Config` env, metadata storage, PostGIS, Azure Maps credentials.  
  - **Outputs:** JSON/GeoJSON/GeoPackage/GeoParquet responses; metadata records (`project`, `imagelayer`, `model`, `labels`, `validation`, …); queue messages consumed by `hastefuncqueues`; assessment rows created by the assessment store helpers.  
  - **Downstream:** `utils/blob.py` (uploads/ranges), `utils/url_allowlist.py` (imagery/footprint URL checks), `utils/gdal_security.py` (upload format checks), `utils/postgis_query.py` / `assessment_store.py` / `current_damage.py` (reads), `utils/user.py` (invitations), `utils/aoi_validate.py` (assessment AOI).  
  - **Error behavior:** validation errors return 4xx; storage/DB failures propagate to 500; `GetAoiAssessment*` endpoints use paging (`_MAX_PAGE_SIZE`) and `_bbox_param`; `_assessment_id_param` normalizes IDs.  
**3.3** `api/hastefuncapi/startup.py`  
- **Functionality:** container pre-start hook. Waits for Azurite (up to 30×2 s), then rebuilds the project stats cache (`project` metadata → `stats` record) so the dashboard survives restarts.  
- **Pipeline position:** executed by `entrypoint.sh` **before** the Functions host starts.  
- **Data flow:** Azurite blob listing → `MetadataProcessor(PROJECT).load_all()` → `MetadataProcessor.save("stats", {...})`. Failure is logged but does not necessarily block startup (exit code 1 on stats failure).  
**3.4** `api/hastefuncapi/entrypoint.sh`**,** `Dockerfile`  
- **Functionality:** `entrypoint.sh` runs `startup.py` then `exec`s the Azure Functions host. The Dockerfile starts from `mcr.microsoft.com/azure-functions/python:4.30.0-python3.11`, creates a non-root `appuser`, installs `requirements.txt`, copies `hastelib/src/hastegeo` into the site root, installs `hastelib` metadata, copies the function app, exposes 8080.  
- **Pipeline position:** packaging/runtime bootstrap.  
- **Data flow:** image build inputs (API code + `hastegeo` + requirements) → runtime process → loads `function_app.py`.  
**3.5** `api/hastefuncapi/host.json`**,** `requirements.txt`**,** `README.md`**,** `start_local_func.sh`**,** `__init__.py`**,** `.gitignore`  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `host.json` |  Functions host config (extension bundle, logging, concurrency) |  runtime |  read by Functions host |  
| `requirements.txt` |  Python deps for the API image |  build |  pip install in Dockerfile |  
| `README.md` |  How to run the API locally |  docs |  none |  
| `start_local_func.sh` |  Local `func start` helper |  dev only |  shell env → Functions host |  
| `__init__.py` |  package marker |  import |  none |  
| `.gitignore` |  hygiene |  repo |  none |  
  
   
**3.6** `api/hastefuncqueues/function_app.py` **(\~989 lines)**  
- **Functionality:** nine queue-triggered functions, each loading the relevant entity through `MetadataProcessor`, constructing the processor, and running it:  
  1. **Image-layer queue** — `ImageryPreProcessor` / `ImageryPostProcessor`.  
  2. **Embedding queue** — `EmbeddingPreprocessor` / `EmbeddingPostprocessor`.  
  3. **DTF queue** — `DTFInferencePreprocessor` / `DTFInferencePostprocessor` (whole-layer).  
  4. **DTF-AOI queue** — `DTFAoiInferencePreprocessor` / `DTFAoiInferencePostprocessor`.  
  5. **Assessment queue** — `AoiAssessmentPreprocessor` / `AoiAssessmentPostprocessor` (large AOI).  
  6. **Generic inference queue** — segmentation-model training/inference processors.  
  7. **Stats queue** — `StatsPreProcessor` / `StatsPostProcessor`.  
  8. **Image poison queue** — failure isolation for malformed image messages.  
  9. **Artifacts zip queue** — `ArtifactProcessor` zip jobs.  
- **Pipeline position:** the asynchronous kernel. Every queue handler re-enqueues the entity's JSON while status is `InProgress`, which doubles as the polling mechanism.  
- **Data flow & dependencies:**  
  - **Input:** queue message JSON (entity ID + status); metadata storage (authoritative entity); image-layer/model/layer records.  
  - **Output:** processor side effects — task submission, status/progress writes, artifact URLs, PostGIS updates, log messages; re-enqueued messages until a terminal state.  
  - **Failure modes:** missing/deleted model guarded by loading metadata before processing; failures set `FAILED` and append `statusMessage`; poison queues capture undecodable messages. There is no assessment queue poison counterpart in the local seed script (see §16).  
**3.7** `api/hastefuncqueues/Dockerfile`**,** `host.json`**,** `requirements.txt`**,** `README.md`**,** `__init__.py`**,** `.vscode/extensions.json`  
Same pattern as the API package: Functions Python 3.11 base, non-root user, `hastegeo` copy, dependencies, host config. The queue app has no `startup.py`/`entrypoint.sh` (it does not need the stats rebuild).  
**3.8** `api/titilerfuncapi/`  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `app/__init__.py` |  Wraps a full TiTiler FastAPI app (COG, STAC, mosaicjson, TMS routers, CORS, compression, cache headers, `/healthz`, landing page) in a Functions ASGI middleware |  tile serving |  request → TiTiler → COG bytes from blob URLs |  
| `app/function.json` |  HTTP trigger binding for the ASGI function |  trigger config |  Functions host → `app.main` |  
| `host.json` |  Functions host config |  runtime |  host |  
| `requirements.txt` |  TiTiler + app deps (Production deployment path) |  build |  pip |  
| `README.md` |  deployment notes |  docs |  none |  
  
   
In the local stack the `titiler` service uses `ghcr.io/developmentseed/titiler:0.18.6` directly (`docker/titiler/Dockerfile`), not this Functions wrapper.  
  
---  
**4.** `hastelib/` **— package metadata and build**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `pyproject.toml` |  hatchling package metadata, deps (GDAL, rasterio, geopandas, pyarrow, psycopg2, azure-\*, boto3, cv2), console scripts (`prepare-imagery`, `zip-artifacts`, `damage-triage-infer`, `damage-triage-aoi-infer`, `damage-triage-region-infer`) |  build/runtime |  wheel + entry points used by task containers |  
| `haste_build.py` |  Custom hatch build hook: resolves the next version from the `haste-binaries` blob container, rewrites `__about__.py`, uploads the wheel, rewrites `requirements.txt` in the API/queue/imageryprep dirs and `env.yml` |  release pipeline |  Azure CLI identity → blob listing/upload → sibling requirement files |  
| `README.md` |  library usage/architecture |  docs |  none |  
| `.gitignore` |  hygiene |  repo |  none |  
| `src/hastegeo/__about__.py` |  `__version__` |  import |  read by packaging |  
| `src/hastegeo/__init__.py` |  package marker |  import |  none |  
| `logs/damage_triage_aoi_friendly.log`, `logs/embedding_friendly.log`, `logs/region_infer_progress.log` |  runtime progress logs produced by workflows |  runtime artifacts |  written by task containers; read by processors via the runner |  
| `.pytest_cache/*` |  pytest cache |  tooling artifact |  none |  
  
   
  
---  
**5.** `hastelib/src/hastegeo/core/config.py`  
- **Functionality:** the configuration and enumeration hub.  
  - `StorageType` (local/blob/cosmos/datalake/postgres), `ArtifactTypes` (template names for every artifact, including DTF GPKGs, AOI GPKGs, embeddings, PMTiles, sidecars, predictions), metadata types (`PROJECT`, `IMAGELAYER`, `MODEL`, `DTF_CONFIG`, `DTF_AOI_CONFIG`, `AOI_ASSESSMENT*`, `IMAGERY_CONFIG`, …), data formats, roles, user statuses, task statuses.  
  - Storage configs from env; artifact storage selection; runner selection (`azure_batch` default).  
  - Queue names for image, train, inference, stats, zip, embedding, dtf, dtf-aoi, assessment.  
  - Azure Batch pool/image config for `dev`/`prod`, including VM size and node-agent SKU.  
  - Runtime tuning properties: AOI max area/units, DTF normalization, region windows/masks/concurrency/attempts/lease/min-valid/uncertain-confidence, footprint load chunk, map feature limits.  
  - `get_dtf_config()` — checkpoint URL + config URL.  
- **Pipeline position:** imported by nearly every backend module; it is the only place env-var names are centralized.  
- **Data flow:** process environment → typed properties/dicts → processors, runners, workflows, data layers. No outputs beyond return values.  
- **Notable coupling:** queue names and storage types must match between the API and queue containers; compose sets both explicitly.  
  
---  
**6.** `hastelib/src/hastegeo/core/models/`  
These are Pydantic schemas shared by the API, processors, and UI payloads.  
  
|  |  |  |  
| - | - | - |  
| **File** |  **Main models** |  **Functionality / pipeline role** |  
| `projects.py` |  `PrimaryClass`, `Geometry`, `Properties`, `Label`, `LabelingImagery`, `Feature`, `LabelProject`, `TrainingJob`, `InferenceJob`, `Checkpoint`, `ModelRequest`, `Model`, `ZipTypeEnum`, `ZipJob`, `ModelArtifacts`, `ImageryPreprocessJob`, `ImageLayer`, `ValidationLabel`, `BuildingValidation`, `Project`, `Projects`, `AoiAssessmentStatus`, `ProcessingUnitStatus`, `BatchStatus`, `BuildingResultStatus`, `AoiAssessment` |  The domain model. `Model` carries `modelType`, `predictedDamageLayerUrl`, `gpkgUrl`, `parquetUrl`, `dtfJob`, `dtfAoi`, `dtfBuildingIds`, `dtfAoiJob`. `ImageLayer` carries pre/post imagery URLs, previews, mosaic/processed COG URLs, footprints URL, valid-area mask, progress and status. The enums are the state vocabulary for the large-AOI machine. |  
| `admin.py` |  `SourceType`, `BaseModels`, `DrawingTools`, `Grid`, `DefaultPrimaryClass`, `LabelingToolSettings`, `AdminConfig` |  Admin settings blob used by the labeling tool and model catalog |  
| `stats.py` |  `ImageLayerStats`, `ProjectStats`, `StatsRequest`, `ProjectsSummary` |  Dashboard aggregates |  
| `training.py` |  `ModelSource`, `Imagery`, `Inference`, `Labels`, `Training`, `ExperimentConfig`, `CatalogModel` |  Training/inference job configuration and model catalog entries |  
| `uploader.py` |  `FileUploadRequest` |  Chunked upload request schema |  
| `users.py` |  `User`, `Users`, `Invite`, `Invites`, `AddUsersRequest` |  Identity/invitation payloads |  
| `visualizer.py` |  `Imagery`, `Visualizer` |  Visualizer metadata (imagery and model layer URLs) |  
| `__init__.py` |  — |  package marker |  
  
   
**Pipeline position:** request/response validation at the API boundary; deserialization/serialization inside processors; the `stats` and `admin` records are persisted blobs read by the UI.  
   
 **Data flow:** HTTP JSON → model instances → `MetadataProcessor.save` → storage → queue message JSON → processors → status fields → API responses.  
  
---  
**7.** `hastelib/src/hastegeo/core/processors/`  
Every processor follows the same contract: load entity by ID from metadata storage, act, persist the updated entity, and either enqueue the next state or finish.  
**7.1** `metadata.py` **—** `MetadataProcessor`  
- **Functionality:** the storage facade used by every processor and many API routes. Wraps the configured data layer (`UnifiedDataLayer`) for `save`, `load`, `load_all`, `update`, `delete`, `get_file_remote_path`, and partition operations.  
- **Pipeline position:** directly beneath every read/write of project/layer/model/label/validation state.  
- **Data flow:** entity dict + `data_type` + identifier → serialized JSON/YAML → local/blob/cosmos/datalake/postgres. Consumers receive dicts or Pydantic models. Failures bubble to the API/queue handler.  
**7.2** `imagery.py` **—** `ImageryPreProcessor` **/** `ImageryPostProcessor`  
- **Functionality:**  
  - `ImageryPreProcessor` accepts a new/edited layer, validates inputs, writes the `imageryprep_config` YAML artifact, and submits a `prepare-imagery` task. For an imagery update it adds `imagery_update.new_post_event_imagery_urls` and keeps the previous TIFs as the base grid. It records an `ImageryPreprocessJob`, sets `IN_PROGRESS`, and enqueues the layer.  
  - `ImageryPostProcessor` polls the task. On completion it reads `imagery_manifest.json`, sets pre/post mosaic and processed COG URLs, previews, normalization means/stds, footprint URL, valid-area mask, loads footprints into PostGIS, registers imagery in `hg_imagery`, handles update retirement (old imagery rows become inactive), and clears `pendingImageryUpdateUrls`. On failure it records the error, and for a failed *update* keeps the layer `COMPLETED` so the previous imagery remains usable.  
- **Pipeline position:** first stage of every layer's life; everything downstream (DTF, embeddings, assessment) depends on its outputs.  
- **Data flow:**  
  - **Input:** `ImageLayer` JSON from the image queue; imagery/footprint URLs (user uploads, blob URLs); config YAML.  
  - **Output:** `imagery_manifest.json` (in task outputs) → layer URL fields; `hg_buildings` rows via `postgis_load`; `hg_imagery` rows via `imagery_catalog`; progress/status messages.  
  - **Downstream:** `damage_triage*.py` uses `postEventMosaicCogImageryUrl` and `buildingFootprintsUrl`; the UI uses previews/mosaics; the assessment machine uses `hg_imagery`.  
- **Failure modes:** missing/invalid URLs, task failure, footprint download failure (`building_footprints_error` in the manifest), no valid-area mask (warning only), imagery update alignment failure (layer keeps old imagery).  
**7.3** `damage_triage.py` **— whole-layer DTF**  
- **Functionality:**  
  - `_find_local_checkpoint()` searches `DTF_CHECKPOINT_PATH`/`DTF_CHECKPOINT_CONFIG_PATH`, `models/damage_triage`, parent dirs, HF cache, `/opt/dtf-checkpoint`, `/app/models/damage_triage`.  
  - `_download_hf_checkpoint()` falls back to `hf_hub_download` for `best_model.pth` and `config.json` into the cache.  
  - `DTFInferencePreprocessor` validates the layer has a post-event mosaic and footprints, builds a DTF config JSON (`files`: imagery, footprints, predictions GPKG, checkpoint, checkpoint config; `checkpoint_cache`: local paths; `pipeline.tile_size=1024`), submits a `damage-triage-infer` task, records `dtfJob`, sets `IN_PROGRESS`, enqueues.  
  - `DTFInferencePostprocessor` is a state machine: Pending → submit; InProgress → `get_task_status`; completed → read `damage_triage_manifest.json`, set `gpkgUrl` and `predictedDamageLayerUrl`, copy friendly logs; failed → mark failed and record logs; otherwise re-enqueue.  
- **Pipeline position:** middle stage: layer imagery/footprints → model job → prediction artifacts.  
- **Data flow:**  
  - **Input:** `Model` + `ImageLayer` metadata; post-event mosaic; footprints GPKG; checkpoint/config.  
  - **Output:** predictions GPKG, predictions/visualizer COGs, manifest → `model.gpkgUrl` / `model.predictedDamageLayerUrl`; status/progress. The whole-layer processor does **not** write PostGIS; map and download endpoints read the artifact URLs.  
  - **Downstream:** `GetVisualizerResults`/map tiles; results download; validation reports.  
**7.4** `damage_triage_aoi.py` **— AOI-scoped DTF**  
- **Functionality:** mirrors the whole-layer processor but resolves the scope from `dtfAoi` GeoJSON or `dtfBuildingIds` using `postgis_query.buildings_in_aoi`, raising if the shortlist is empty. The config embeds the full building list (`[{building_id, geometry}]`) and submits `damage-triage-aoi-infer`. On completion it:  
  - updates `hg_buildings` via `update_building_predictions(..., review_status="pending_review")`;  
  - records run history in `hg_dtf_run_results`;  
  - sets the AOI GPKG and GeoParquet URLs;  
  - falls back to GeoParquet publishing if needed.  
- **Pipeline position:** analyst-triggered review path; consumes the same layer artifacts as whole-layer but writes per-building rows instead of a monolithic raster.  
- **Data flow:**  
  - **Input:** `Model` with `modelType="dtf-aoi"`, `dtfAoi`/`dtfBuildingIds`; PostGIS `hg_buildings`; mosaic/footprints; checkpoint.  
  - **Output:** `dtf_aoi_predictions_<model>.gpkg` + `.parquet`; updated building rows (`result_status`, class, confidence, `review_status=pending_review`, crop/mask URLs); run history.  
  - **Downstream:** `DtfAoiReviewPanel` (`PutBuildingDTFReview`), `GetAoiPredictionsGpkg`, assessment statistics, review-aware exports.  
**7.5** `aoi_assessment.py` **— large-AOI durable state machine**  
- **Functionality:** the most complex processor. It owns the `hg_assessments` lifecycle:  
  - `AoiAssessmentPreprocessor.send_to_queue` creates/loads the assessment and enqueues.  
  - `AoiAssessmentPostprocessor.process` dispatches on status:  
    - `submitted` → `validating` → AOI validation.  
    - `discovering` → resolve imagery for the AOI (`imagery_for_aoi`), count eligible buildings, compute/evaluate the checkpoint digest, zero buildings → `completed`.  
    - `tiling` → select 1024 px units (`region_windows.select_units`), reject above `HASTE_AOI_MAX_UNITS`, group batches (`group_batches`), insert units/batches, → `queued`.  
    - `queued/running` → expire leases, poll batch statuses, ingest outputs (manifest + digest check, evidence upsert, unit statuses, batch completion), requeue failed batches up to `HASTE_REGION_MAX_ATTEMPTS`, start pending batches up to `HASTE_REGION_MAX_CONCURRENT_BATCHES` using `claim_batch_lease`, re-enqueue.  
    - `aggregating` → `region_aggregate.aggregate` → `upsert_damage_assessments` → `current_damage.publish_current_results` → `export_after_publish` → `completed` / `completed_with_errors`.  
    - `cancelling` → cancel tasks, skip pending units/batches, → `cancelled`.  
  - Config values come from the frozen assessment config with environment fallbacks (`cfg_int`/`cfg_float`).  
- **Pipeline position:** orchestrator between the HTTP submission and the per-batch region workflow.  
- **Data flow:**  
  - **Input:** assessment row (`hg_assessments`), `hg_imagery`, `hg_buildings`, checkpoint digest, environment knobs.  
  - **Output:** `hg_processing_units`, `hg_unit_buildings`, `hg_batches`; runner tasks; `hg_prediction_evidence`; `hg_damage_assessments`; updated `hg_buildings` current-result columns; GeoParquet export; status/counters.  
  - **Downstream:** assessment read APIs, map data, statistics, `GetCurrentBuildingDamage`.  
**7.6** `embedding.py` **— building embeddings**  
- **Functionality:** `EmbeddingPreprocessor` submits `embed-buildings`; `EmbeddingPostprocessor` polls and publishes per-building embeddings GPKG, PMTiles, binary feature sidecar, and (optionally) the interactive labeler's prediction GPKG.  
- **Pipeline position:** alternative model path on the same layer artifacts; feeds the Interactive Labeler and MOSAIKS-style classification.  
- **Data flow:** layer mosaic + footprints → training container → artifact URLs on `Model` → UI/interactive labeler.  
**7.7** `inference.py` **— generic segmentation inference**  
- **Functionality:** `BaseInferenceProcessor`, `InferencePreprocessor`, `InferenceLogRecord`, `InferencePostprocessor`; the pre-DTF path for torchgeo segmentation checkpoints (training/inference jobs).  
- **Pipeline position:** legacy/general model path; shares the same poll/status pattern.  
- **Data flow:** model checkpoint + imagery → `inference`/`run_workflow` scripts in the training image → prediction GPKG + visualizer COG → model artifact URLs.  
**7.8** `labels.py` **—** `LabelTaskGenerator`  
- **Functionality:** converts labeling-tool GeoJSON (primary classes, geometries, imagery references) into per-label training samples/tasks (chips + masks/label JSON), driven by label metadata and layer imagery.  
- **Pipeline position:** between the labeling UI and model training.  
- **Data flow:** `labels` metadata + `imagelayer` metadata → training container inputs / `train_labels` artifacts.  
**7.9** `stats.py` **—** `StatsPreProcessor` **/** `StatsPostProcessor`  
- **Functionality:** asynchronous recomputation of project/image-layer statistics; the preprocessor enqueues and the postprocessor writes the stats record.  
- **Pipeline position:** after layer/model changes; feeds dashboards.  
- **Data flow:** project/layer metadata → stats aggregator → `stats` metadata record → `GetDashboardData`/project details.  
**7.10** `uploader.py` **—** `FileUploader`  
- **Functionality:** assembles chunked uploads (`UploadFileByChunk`): validates chunk size/format via `gdal_security`, stages blocks through the data layer, finalizes into a blob, and returns the URL.  
- **Pipeline position:** API upload path for imagery and footprint files.  
- **Data flow:** HTTP chunks → staged blob blocks → finalized blob → `ImageLayer` URL fields (later consumed by `prepare-imagery`).  
**7.11** `artifacts.py` **—** `ArtifactProcessor`  
- **Functionality:** builds downloadable ZIPs for model artifacts/training/inference artifacts, tracks zip jobs, writes the archive to artifact storage, and updates the model's artifact/zip URLs.  
- **Pipeline position:** post-training/post-inference packaging; triggered by the zip queue.  
- **Data flow:** artifact metadata + stored files → zip in artifact storage → download URL on the model.  
**7.12** `__init__.py`  
- Package marker; no runtime behavior.  
  
---  
**8.** `hastelib/src/hastegeo/core/runners/`  
**8.1** `base.py`  
- **Functionality:** `BaseRunner` ABC: `get_filecontent_from_task`, `get_task_status`, `add_task`, `cleanup_task`, `cancel_task`; constructs `Config()` if none is given.  
- **Pipeline position:** interface implemented by both backends; consumed by every processor.  
- **Data flow:** task identifiers and resource descriptors in; status/file content out.  
**8.2** `unified_runner.py`  
- **Functionality:** `UnifiedRunner(runner_type, ...)` dynamically imports `AzureBatchRunner` (`azure_batch`) or `LocalRunner` (`local`) and delegates every method. Logs resource files.  
- **Pipeline position:** the processor-facing facade; selected by `RUNNER_TYPE`.  
- **Data flow:** passthrough. Consequence: both backends must accept the exact same `add_task(**kwargs)` surface, which `local.py` effectively defines.  
**8.3** `azure_batch.py`  
- **Functionality:** Azure Batch lifecycle.  
  - `AzureBatchRunner` maps task state/files.  
  - `AzureBatchJob` authenticates with account key or `DefaultAzureCredential`; creates/resizes pools; creates jobs with task retention; adds tasks with container settings (`--rm --shm-size=32g --user 0:0`), resource files, and output-file uploads; retries 5xx with `retry_on_server_error`; cleanup deletes task files and disables the job; cancel/delete helpers.  
- **Pipeline position:** production task execution.  
- **Data flow:** config artifact + resource URLs → Batch resource files; container command (`damage-triage-infer`, etc.) → task outputs uploaded to blob → `get_filecontent_from_task` / `get_task_status` → processors.  
- **Notable:** containers run as root; no per-task timeout; retries are blind to non-idempotent operations.  
**8.4** `local.py`  
- **Functionality:** local emulation of Batch using the Docker socket.  
  - Creates `/shared/azurite/task_work/<job>/<task>/` with downloaded resources, `outputs/`, `status.json`, `resolved_command.txt`, `output.log`.  
  - Downloads resource files (blob/Azurite URLs) into the task dir, substitutes `$AZ_BATCH_*` variables, and runs a sibling container named `haste-<image>` attached to `haste_default` with the Azurite volume, shm/mem limits, and GPU device requests when enabled (CPU fallback on attach error).  
  - Streams container logs, validates required outputs (`imagery_manifest.json` for imagery prep; non-empty log), writes status, uploads outputs through the data layer.  
  - `get_task_status`: reads `status.json`; **missing status file is treated as COMPLETED**.  
  - `cleanup_task` deletes the task dir unless `PRESERVE_LOCAL_TASK_DIRS`; `cancel_task` only writes a `cancelled` status file and does not stop the container.  
- **Pipeline position:** local/dev task execution; called by every processor when `RUNNER_TYPE=local`.  
- **Data flow:** same as Azure Batch, but resources/outputs move through the shared Azurite volume and the storage abstraction; task dirs are the handoff point.  
- **Notable:** hard-coded Azurite key and DTF model paths; synchronous `container.wait()`; broad env passthrough; no concurrency.  
  
---  
**9. Storage seams**  
**9.1** `core/data_layer/`  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `abstract_data_layer.py` |  ABC for metadata CRUD + chunked saves + file loading |  interface |  processor ↔ backend |  
| `unified.py` |  Hashes partition keys, maps `local/blob/cosmos/datalake/postgres`, delegates; adds `get_file_path`, `get_file_remote_path`, `get_base_url` |  facade |  `MetadataProcessor` → backend |  
| `local_file_system_data_layer.py` |  `<dir>/<partition>/<type>_<id>.<fmt>`; JSON/YAML; no chunk support |  local dev/tests |  processor → disk |  
| `azure_blob_storage_data_layer.py` |  Container/policy setup, block staging + commit, load/list/delete, base URL |  production metadata |  processor → blob |  
| `azure_cosmos_db_data_layer.py` |  `{data_type}_{id}` docs, partition key, `STARTSWITH` listing; no chunks |  alternative metadata |  processor → Cosmos |  
| `azure_data_lake_data_layer.py` |  DataLake save/load (dict/bytes, JSON) |  alternative metadata |  processor → ADLS |  
| `azure_postgresql_data_layer.py` |  Single JSONB table with geometry/bounds, validated table identifier |  alternative metadata |  processor → PostgreSQL |  
| `__init__.py` |  package marker |  import |  — |  
  
   
**9.2** `core/artifact_storage/`  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `abstract_artifact_storage.py` |  Artifact CRUD/download-URL interface |  interface |  processor ↔ storage |  
| `unified_artifact_storage.py` |  Partition hashing + mutual-exclusion validation + delegation |  facade |  artifact consumers → backend |  
| `azure_blob_artifact_storage.py` |  Container policy, upload/fetch, SAS download URLs; **base URL hard-coded to `*.blob.core.windows.net`** |  production artifacts |  task outputs → blob → API download |  
| `local_file_system_artifact_storage.py` |  Local dirs and `file://` URLs; `copytree` without destination creation |  local artifacts |  task outputs → disk |  
| `__init__.py` |  package marker |  import |  — |  
  
   
**Pipeline position (both seams):** every processor read/write and every task resource/output passes through these. **Data flow:** entity/artifact bytes → storage backend → URL/path returned to processor/API/UI.  
  
---  
**10.** `hastelib/src/hastegeo/core/workflows/`  
These are the console scripts executed **inside task containers**. They receive a JSON config downloaded by the runner, read `inputs/`, and write `outputs/`.  
**10.1** `prepare_imagery.py` **(\~1,230 lines)**  
- **Functionality:** end-to-end imagery preparation. Downloads pre/post imagery (Azure blob, S3, local upload URLs), optionally aligns an update onto the existing grid (`imagery_update`), builds mosaics and processed COGs, computes normalization statistics, optionally writes per-TIF catalogue COGs (`catalog_mode`), downloads Overture footprints or clips/reprojects user footprints, aligns footprints to the raster (soft-fail), produces a valid-area mask, and writes `imagery_manifest.json`.  
- **Pipeline position:** first workflow; triggered by `ImageryPreProcessor`.  
- **Data flow:**  
  - **Input:** `imageryprep_config` YAML (`project_id`, layer, pre/post URLs, source types, fine-tune, user footprints URL, catalog mode, optional `imagery_update`).  
  - **Output:** raw/preview files, mosaics, processed COGs, footprint GPKG, valid-area mask, normalization stats, `imagery_manifest.json` → runner uploads → `ImageryPostProcessor`.  
  - **Failure behavior:** footprint failure is captured in the manifest (`building_footprints_error`) so the layer can be marked failed by the postprocessor without losing the COGs already produced; a valid-area-mask failure is recorded separately.  
**10.2** `damage_triage_infer.py` **(\~950 lines)**  
- **Functionality:** whole-layer DTF.  
  - `build_dtf_model` detects a crop-mode checkpoint by `gate.weight`, loads `GatedDinoCrop` only when `allow_crop_mode=True`, otherwise constructs `src.model.net.Model` from the config's flat training keys and audits missing `backbone.*`/`damage_head.*` weights.  
  - `score_window` normalizes a tile, batches masks, halves the batch on CUDA OOM, and falls back to CPU.  
  - `decode_gated_predictions` / `decode_severity_predictions`.  
  - `rasterize_instance_masks`, `read_tile` (zero-pad), `write_visualizer_rasters` (predictions GeoTIFF + RGBA visualizer COG via GDAL).  
  - `run_damage_triage`: reads footprints, tiles the raster, scores each intersecting footprint per tile, keeps best confidence, preserves row order (`id=0..N-1`), writes per-building GPKG, two rasters, and `damage_triage_manifest.json`.  
- **Pipeline position:** task body for `damage-triage-infer`.  
- **Data flow:** config JSON → imagery/footprints/checkpoint inputs → GPKG + rasters + manifest → runner upload → `DTFInferencePostprocessor` → model URLs.  
- **Critical detail:** `score_window` is called with `normalization="unit"` even though the training pipeline (and the region/AOI crop paths) use ImageNet normalization (see §19, P0-4).  
**10.3** `damage_triage_aoi_infer.py` **(\~570 lines)**  
- **Functionality:** per-building AOI scoring. Reuses `build_dtf_model`, decoders, palette, and device resolution. For each building: `crop_building` + `crop_building_mask`, gates on valid-pixel fraction and minimum footprint pixels, scores crop mode (`score_crop`, ImageNet) or mask mode (`crop/255`), writes crop/mask PNGs, GPKG (`id, dtf_class, dtf_class_name, dtf_confidence, crop_url, mask_url, geometry`) and GeoParquet (`geometry, damage_class, confidence, status`), plus `damage_triage_aoi_manifest.json`.  
- **Pipeline position:** task body for `damage-triage-aoi-infer`.  
- **Data flow:** config (imagery, buildings list, checkpoint) → per-building results → runner upload → `DTFAoiInferencePostprocessor` → `hg_buildings` + model URLs.  
- **Critical detail:** mask-mode normalization is `crop / 255.0` (unit), inconsistent with ImageNet training.  
**10.4** `damage_triage_region_infer.py` **(\~400 lines)**  
- **Functionality:** batch scoring for the large-AOI machine. Loads a checkpoint once (crop or mask mode; `allow_crop_mode=True`), reads one TIF, and processes its assigned units with windowed reads. Uses a valid-pixel gate, rasterized masks or crop boxes, a prefetch `ThreadPoolExecutor`, `score_window`/`score_crops`, and emits `evidence.jsonl` rows (`unit_id`, `building_id`, probs, class, confidence, coverage/valid fractions, status) plus `region_manifest.json` with model digest, preprocessing version, and counts.  
- **Pipeline position:** task body for `damage-triage-region-infer`, one per batch.  
- **Data flow:** batch config (imagery, units, checkpoint) → `evidence.jsonl` + manifest → runner upload → `AoiAssessmentPostprocessor` ingest → `hg_prediction_evidence` → aggregation.  
- **Critical detail:** it defaults to and enforces `imagenet` for crop-mode checkpoints; this is the only DTF path that matches the checkpoint's declared preprocessing.  
**10.5** `embed_buildings.py`  
- **Functionality:** per-building MOSAIKS/DINOv2 embeddings; writes the embeddings GPKG, PMTiles, binary HFTR sidecar, and maintains the row-order/ID invariant (`id=0..N-1` in footprints file order).  
- **Pipeline position:** task body for the embedding queue.  
- **Data flow:** mosaic + footprints → embeddings artifacts → `EmbeddingPostprocessor` → model artifact URLs → labeler/map.  
**10.6** `zip_artifacts.py`  
- **Functionality:** packages model artifacts into a ZIP, writes it, and reports the path/URL.  
- **Pipeline position:** zip queue task body.  
- **Data flow:** artifact files → zip → artifact storage → model download URL.  
  
---  
**11.** `hastelib/src/hastegeo/core/utils/`  
The utilities are mostly **not leaves**: many perform DB or filesystem side effects. Grouped by role.  
**11.1 PostGIS**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `postgis_connect.py` |  Opens a psycopg2 connection: password auth in dev, `DefaultAzureCredential` token in production; `sslmode` from env |  every DB caller |  env + `get_connection()` → connection |  
| `postgis_load.py` |  `load_footprints_into_postgis` (delete/replace or preserve results), canonical SHA1 building IDs, streaming loader (pyogrio Arrow/JSONL), `mark_imagery_updated` |  imagery postprocessing / footprint ingestion |  footprints GPKG → `hg_buildings` |  
| `postgis_query.py` |  `buildings_in_aoi`, `count_buildings_in_aoi`, `building_ids_in_aoi`, `layer_buildings_feature_collection` (20k cap), run-results table, `record_run_results`, `update_building_predictions`, `set_review_status`, classified GeoJSON, `export_aoi_gpkg` |  AOI processor, API reads, map |  `hg_buildings` ⇄ processors/API |  
| `assessment_store.py` |  Full large-AOI persistence: assessment CRUD, units/batches/associations, lease claim/renew/expire, unit results, evidence upsert, damage results, statistics, paged buildings, map data, history |  assessment processor + read APIs |  assessment state machine ⇄ `hg_*` tables |  
| `current_damage.py` |  Publishes completed assessment rows into `hg_buildings` current columns (set-based update, never geometry), current GeoJSON/keyset pagination, GeoParquet export |  after aggregation |  `hg_damage_assessments` → `hg_buildings` → API/export |  
| `imagery_catalog.py` |  `describe_raster`, deterministic imagery IDs, register/retire `hg_imagery`, `imagery_for_aoi` |  imagery postprocessing + assessment discovery |  manifest COG metadata → `hg_imagery` → discovery |  
| `twdb_buildings.py` |  Loads TWDB Cycle 2 building footprints with attributes JSONB |  alternative footprint source |  shapefiles/GPKG → `hg_buildings` |  
| `geoparquet.py` |  Writes GeoParquet with correct EPSG:4326 PROJJSON, WKB, bbox, zstd, atomic replace |  exports |  rows/WKB → `.parquet` |  
  
   
**11.2 Geo/AOI**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `aoi.py` |  Extract AOI polygon from a raster's valid data; bbox; save GeoJSON |  imagery prep |  COG → AOI polygon |  
| `aoi_validate.py` |  Validate geometry/Feature/FeatureCollection; reject non-polygons, empty/invalid, out-of-bounds, antimeridian, over-area; returns EPSG:6933 area |  assessment submission |  AOI GeoJSON → validated geom + area |  
| `aoi_analysis.py` |  Coverage estimate (`READY`, `INCOMPLETE_IMAGERY`, `NO_BUILDINGS`, …) and 1024-window estimate |  assessment preview |  AOI + imagery → preview |  
| `region_windows.py` |  `building_coverage`, `select_units` (grid + STRtree + building-centred units), `group_batches` |  tiling phase |  imagery raster + eligible buildings → units/batches |  
| `region_aggregate.py` |  Deterministic per-building aggregation: valid-fraction filter, sort by coverage/confidence/unit id, uncertain/insufficient/failed statuses |  aggregation phase |  evidence rows → damage results |  
| `crop_service.py` |  `crop_building`, `crop_building_mask`, `write_png` |  AOI/region task bodies |  raster + building geometry → crop/mask |  
| `footprint_alignment.py` |  FFT edge cross-correlation to estimate/apply a footprint shift; leave-alone when uncertain |  imagery prep |  raster + footprints → shifted footprints + shift report |  
| `tiles.py` |  Appends `nodata=0` to TiTiler tile URLs |  UI/API tile helpers |  tile URL → tile URL |  
  
   
**11.3 Imagery**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `downloader.py` |  `ImageryDownloader`: Azure blob/S3/local URLs with tenacity retries, redirect refusal, size caps; Maxar/Planet are stubs |  imagery prep |  source URLs → local files |  
| `imagery.py` |  `ImageryUtils`: GTiff/COG/RGB checks, fine-tune, JPEG previews, band selection, mosaic (VRT + Warp), RGB COG conversion, normalization stats; GDAL hardening at import |  imagery prep |  raw TIFs → mosaics/COGs/stats |  
| `imagery_update.py` |  Coregistration and grid alignment for imagery updates: valid footprints, planning, grid snapping, warps, shift measurement, retire rules (`RETIRE_COVERAGE=0.98`) |  imagery update |  old + new TIFs → aligned set + update report |  
| `footprints.py` |  Overture Maps download/clip (anonymous Azure Parquet), latest release lookup, user footprint clip/normalize |  imagery prep |  AOI + release → footprints GPKG |  
| `footprint_sources.py` |  Multi-format footprint discovery (GPKG/GDB/GeoJSON/Parquet/ZIP/folder), zip-slip protection, source caps, ID normalization |  user footprint upload |  uploaded archive → normalized GPKG |  
| `blob.py` |  Azurite/Azure blob URL parsing, range reads, async tempfile download |  uploads/ranges |  URL/range → bytes |  
| `metadata.py` |  UUID/int IDs, timestamps, short IDs, hashing (partition keys), status-message append |  everywhere |  values → IDs/partition keys |  
| `data.py` |  URL regex extraction, JSON→GeoJSON, label class discovery, directory cleanup, role filtering |  processors/workflows |  strings/dicts → cleaned data |  
| `artifacts.py` |  Empty module (header only) |  — |  — |  
| `exceptions.py` |  `UnsupportedFormatError` |  upload validation |  — |  
  
   
**11.4 Security / operations**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `gdal_security.py` |  GDAL driver allowlist, HDF/netCDF denylist, `GDAL_SKIP`, magic-byte sniffing (incl. GPKG vs SQLite), declared-format match, upload/download size caps |  import-time + upload/download boundaries |  file bytes → validated type/size |  
| `url_allowlist.py` |  Scheme/host/port allowlists for imagery and footprints; local-host opt-in; loopback spoof tests |  downloader/processors/API |  URL → validated URL or error |  
| `queues.py` |  `AzureQueueHandler`: get-or-create queue, send/receive/peek/delete with visibility timeout |  processors/API |  message JSON ⇄ queue |  
| `logs.py` |  `Logger.get_logger`, console/file handlers, level control |  workflows/processors |  log records → stdout/file |  
| `user.py` |  `InvitationManager` (ARM invitation + email) and `UserManager` (list/delete) |  admin API |  email/roles → invitation/user operations |  
  
   
**11.5 Model constants and evaluation helpers**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow** |  
| `dtf_constants.py` |  Torch-free mirror of the DTF label contract: `DTF_CATEGORIES` (class names/colours), `UNCLASSIFIED`, `DTF_IM_SIZE`, shared enums re-exported by the workflows. |  Imported by API, processors and both inference paths; exists so containers can read labels without importing the training stack. |  Checkpoint/class metadata → canonical names written to GPKG, PostGIS and UI legends. |  
| `dtf_preprocess.py` |  `normalize_tile(tile, mode)` for `unit` (/255) and `imagenet` (mean/std), input-shape validation, `PREPROCESSING_VERSION`. |  Every scoring path. |  uint8 RGB tile → float32 tensor-ready array. This is the single place where the normalisation policy should be enforced; callers currently choose inconsistently. |  
| `assessment.py` |  Precision/recall/F1 and confidence-interval maths over building results; JSON-finiteness guards. |  Reporting/statistics endpoints and tests. |  Ground truth + predictions → metric dict → assessment statistics response. |  
  
   
All remaining `__init__.py` files under `hastegeo/` and `hastelib/tests/` (including `core/`, `core/models/`, `core/processors/`, `core/runners/`, `core/data_layer/`, `core/artifact_storage/`, `tests/core/`, `tests/workflows/`) are empty package markers with no runtime behaviour.  
`utils/__init__.py` contains a legacy `sys.path` append that makes the data-layer modules' non-package-relative imports work.  
  
---  
**12.** `hastelib/tests/`  
The suite is mostly unit tests with mocked cursors, plus a handful of real-raster and opt-in PostGIS integration tests. It is important to the gap analysis because it shows what is **not** exercised.  
  
|  |  |  
| - | - |  
| **Test file** |  **Covers** |  
| `conftest.py` |  shared fixtures/env |  
| `core/processors/test_aoi_assessment_processor.py` |  assessment state machine dispatch, cancel, zero buildings |  
| `core/processors/test_damage_triage_aoi.py` |  AOI preprocessing config, building shortlist, postprocessing updates/review status |  
| `core/processors/test_imagery_preprocess_config.py` |  imagery config fields (user footprints, catalog mode) |  
| `core/processors/test_imagery_update_pipeline.py` |  update retire/keep, failure keeps old imagery, catalogue deactivation |  
| `core/processors/test_uploader_format.py` |  upload allowlist/size/content checks |  
| `core/processors/test_artifacts.py` |  live Azurite zip integration |  
| `core/runners/test_local_host_cuda.py` |  host CUDA mount and GPU-attach error classification only |  
| `core/test_aoi_assessment_integration.py` |  real PostGIS end-to-end state machine (opt-in; truncates `hg_*`) |  
| `core/test_aoi_assessment_spec_complete.py` |  requirement-by-requirement matrix (mostly mocks) |  
| `core/test_real_data_aoi.py` |  developer `data/` smoke test (skipped when absent) |  
| `core/utils/test_aoi.py`, `test_aoi_analysis.py`, `test_aoi_validate.py` |  AOI extraction/coverage/validation matrix |  
| `core/utils/test_assessment.py` |  AP/PR math, CI, JSON finiteness |  
| `core/utils/test_assessment_store.py` |  SQL shape for units/batches/leases/evidence |  
| `core/utils/test_blob.py` |  URL parsing/range math |  
| `core/utils/test_current_damage.py` |  publish update excludes geometry, rollback, GeoJSON pagination, Parquet |  
| `core/utils/test_downloader.py` |  retries, redirect refusal, size caps |  
| `core/utils/test_dtf_preprocess.py` |  normalization math |  
| `core/utils/test_footprint_alignment.py` |  shift estimation and leave-alone behavior |  
| `core/utils/test_footprint_sources.py` |  archive discovery, zip-slip, duplicate IDs |  
| `core/utils/test_footprints.py` |  Overture release/path/download/clip/user normalization |  
| `core/utils/test_gdal_security.py` |  magic bytes and driver hardening |  
| `core/utils/test_geoparquet.py` |  GeoParquet round-trip |  
| `core/utils/test_imagery_catalog.py` |  catalogue registration/query/deactivation |  
| `core/utils/test_imagery_update.py` |  valid footprints, grid snapping, retire/keep, registration recovery |  
| `core/utils/test_postgis_load.py` |  delete/insert, canonical IDs, streaming |  
| `core/utils/test_postgis_query.py` |  AOI predicates, review statuses, truncation, geometry never updated |  
| `core/utils/test_region_aggregate.py` |  aggregation determinism and statuses |  
| `core/utils/test_region_windows.py` |  unit selection and batch grouping |  
| `core/utils/test_tiles.py` |  nodata URL |  
| `core/utils/test_twdb_buildings.py` |  TWDB load |  
| `core/utils/test_url_allowlist.py` |  allowlist matrix |  
| `workflows/test_damage_triage_aoi_infer.py` |  off-raster buildings, crop-checkpoint refusal |  
| `workflows/test_damage_triage_crop_batching.py` |  crop boxes, batch sizing, OOM halving/CPU fallback, prefetch |  
| `workflows/test_embed_buildings.py` |  row-order/sidecar invariants with stubbed torch |  
| `workflows/test_prepare_imagery.py` |  footprint failures, mask, user footprints, alignment, update mosaic |  
  
   
**Not covered:** `LocalRunner` task execution/upload/cleanup/cancel, `AzureBatchRunner` entirely, data-layer/artifact-storage implementations, `utils/user.py`/`queues.py`/`logs.py`, real model forwards, HF downloads, real PostGIS for most tests, and the UI. The spec-complete test includes a tautological assertion.  
**13.** `external/damage-triage-former/` **— every file**  
This directory is a vendored copy of the research training repository. It is **not** a service and is **not** executed end-to-end in production; it contributes (a) the model classes that both the mask-pooling and crop-mode inference paths import, and (b) a reference training/evaluation stack that produced the shipped weights. The runtime container copies the whole tree to `/opt/damage-triage-former` (`docker/training/Dockerfile:164-175`) and `hastegeo` inserts that path on `sys.path` (`workflows/damage_triage_infer.py:129,272-274`).  
**13.1 File-by-file**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Pipeline position** |  **Data flow (provider → output → destination)** |  
| `main.py` (880) |  Training/eval entry point; DDP-aware argparse over every ablation flag; builds loaders, model, optimizer (two param groups), warmup+cosine, EMA, resume, per-epoch validation, best-checkpoint selection on `Damage_Macro_F1_e2e`, optional visualisation. |  **Training only.** Never invoked by HASTE. |  `get_dataloaders()` (from `src/data/loader.py`) → `train_one_epoch`/`validate` (from `src/training/engine.py`) → writes `runs/<name>/config.json`, `last_checkpoint.pth`, `best_model.pth`, `logs/`, `vis/` → uploaded manually to HF (`Ymx1025/DamageTriageFormer-model`), which is what `processors/damage_triage.py` downloads at run time. |  
| `src/config.py` (136) |  Central constants: `IM_SIZE=1024`, `PATCH_SIZE=16`, `CATEGORIES`, `COLOR_TO_CAT`, `NUM_CLASSES=5`, `IGNORE_INDEX`, `DEVICE`; resolves the DINOv3 backbone at **import time** by probing `$DTF_BACKBONE_DIR`, ancestor `models/backbone/...`, `/opt/dtf-backbone`, `/app/models/backbone`, `$DTF_BACKBONE_REPO`, then HF id `facebook/dinov3-vitl16-pretrain-lvd1689m`. |  Imported by every model/data/training module, **including at inference**. |  Env/filesystem → module-level `BACKBONE` string used by `backbone.py` and `GatedDinoCrop` → determines whether inference needs network access. A directory is accepted only if it has `config.json` plus a `.safetensors`/`.bin`, so half-downloaded caches are rejected. Importing it pulls in `torch`, which is why `hastegeo/core/utils/dtf_constants.py` exists as a torch-free mirror. |  
| `src/model/backbone.py` (129) |  `DinoV3Backbone`: loads HF ViT, freezes all then unfreezes last N blocks + final norm (+ optionally all LayerNorms), applies optional `drop_path_rate`; forward takes `last_hidden_state` and slices the **last** `gh*gw` tokens to `(B,D,H/16,W/16)`. |  Runtime dependency of the **mask-pooling** path. Not used by `GatedDinoCrop`, which instantiates `AutoModel` directly. |  `(B,3,H,W)` from `Model.forward` → `(B,1024,64,64)` at 1024 px → `SegHead` and SFP in `net.py`. Raises `ValueError` if the transformer blocks cannot be located. Silently assumes prefix tokens (CLS/registers) come **first**; a backbone that appends tokens would scramble the grid. |  
| `src/model/net.py` (191) |  `Model`: assembles backbone → `SegHead` → optional `SimpleFeaturePyramid` → one of three mutually exclusive damage heads (gated / hierarchical / plain severity). Exposes `forward(x, inst_masks)` returning `(seg_logits, damage_out)` and parameter groups. `ModelFrozen` is a convenience subclass. |  Runtime dependency of the **mask-pooling** path (`build_dtf_model` imports `src.model.net.Model`). |  `x` tile + list of per-footprint masks → `seg_logits (B,1,H,W)` + `(gate_list, leaf_list[, aux_list])` / `(type_list, extent_list)` / `[sev_list]` → `damage_triage_infer.score_window` → `decode_gated_predictions`. |  
| `src/model/gated_head.py` (262) |  Gated 5-class head: binary gate `g`, 4-way leaf `l`; `P(0)=1−σ(g)`, `P(k)=σ(g)·softmax(l)[k−1]`. Contains `decode_gated_predictions` / `predict_5class_from_gate` used by tile inference. |  Runtime dependency of both paths: mask-pooling uses the head and decoder; crop mode reimplements the same gate/leaf composition inline in `GatedDinoCrop`. |  Feature grid + masks (nearest-downsampled, attention-pooled or masked mean) → gate/leaf logits → decoder → 5-class probabilities → GPKG attribute `damage_class`/`confidence`. Missing-mask placeholders are sized from `input_proj[0].out_features`; correct only while that module is the `Linear`. |  
| `src/model/heads.py` (314) |  Reusable blocks: `SegHead` (two 3×3 convs → 1×1 → bilinear upsample to input size), `ResidualMLP`, `SpatialAttentionPooling`, `MultiScaleROIPooling` (ASPP-like; disabled in the released recipe), `SeverityHead`, `SeverityHeadSimple`. |  Runtime dependency of mask-pooling; `SegHead` is the building segmenter used by **i-dtf** as well. |  Feature map + masks → pooled instance embedding / segmentation logits. Empty-mask paths return correctly shaped zeros/`(0,C)`. |  
| `src/model/hierarchical_head.py` (265) |  Alternative Type×Extent head (No/Roof/Structural × Partial/Total) composed into 5 classes, plus `class_to_type_extent` mapping. |  Runtime dependency only when the checkpoint config sets `use_hierarchical`; not used by the released checkpoint. |  Feature grid + masks → `(type_logits, extent_logits)` per instance → composed class. `loss.py` imports the mapping but re-derives the labels inline. |  
| `src/model/sfp.py` (89) |  ViTDet-style `SimpleFeaturePyramid`: validates target stride < source stride and power-of-two ratio, then a stack of ConvTranspose2d + GroupNorm + GELU to upsample stride-16 to `sfp_stride` (8 in the released recipe → 128×128 feature map at 1024 px). |  Runtime dependency when `use_sfp`; the released checkpoint uses it. |  Stride-16 backbone map from `backbone.py` → stride-8 map consumed by `GatedDamageHead`. Docstring claims near-identity init; code uses `kaiming_normal_`. |  
| `src/training/loss.py` (437) |  Segmentation BCE+Dice; focal/label-smoothing CE; hierarchical loss with class weights and logit adjustment; gated loss = gate BCE + leaf CE (damaged only) + optional auxiliary SmoothL1; Menon logit adjustment applied to training logits only. |  **Training only.** |  Model outputs + labels from `dataset.py` → scalar loss tuple → `engine.py` backward. Returns a **5-tuple** for the gated path; every caller must unpack 5. NaNs are deliberately not scrubbed. |  
| `src/training/engine.py` (1318) |  Train/validate loops; derives end-to-end metrics by running connected components on predicted segmentation and majority-voting GT masks; emits `Damage_Macro_F1{,_oracle,_e2e}`, per-class F1, Type/Extent F1, confusion tensors. Handles DDP, NaN batches (skips, aborts epoch above 10 %), and grad-norm skips. |  **Training only.** |  Loader batches → forward/loss/backward → metric dict → `main.py` logging and best-checkpoint selection. Raises `RuntimeError` on a NaN storm; returns NaN stats when zero batches ran. |  
| `src/data/preprocessing.py` (158) |  Decodes label rasters: unified-colour majority mapping, legacy dot labels, `mask_to_instances` (OpenCV connected components, `min_area=30`), `make_sample`. |  **Training only**, but defines the **training-time** minimum instance area (30 px) that the serving-side min-area (200 px in i-dtf, checks in HASTE) no longer matches. |  Label PNG + image → `image, sem_mask, inst_masks, inst_sev` → `dataset.py`/`weights.py`/`engine.py`. Unknown colours map to `IGNORE_INDEX`. |  
| `src/data/dataset.py` (210) |  Geometric (flip/rot90) and photometric augmentations; dataset returns `image, sem_mask, inst_masks, inst_sev, image_path`. |  **Training only.** |  Triplets from `loader.py` → augmented tensors → `get_dataloaders`. Copy-paste is applied before augmentation by design; geometric transforms are applied to masks consistently. |  
| `src/data/loader.py` (691) |  Tile-triplet discovery with mtime-validated cache, `DistributedWeightedSampler`, filtered/stratified splits, `get_dataloaders`. |  **Training only.** |  `DINOV3_DATA_ROOT` tile triplets + `stratified_splits.json` → `DataLoader`s → `main.py`. Raises `ValueError` on an empty split fold; warns and returns `[]` for missing dirs. |  
| `src/data/copy_paste.py` (301) |  Rare-class copy-paste augmentation with an `InstanceBank` of cached crops (classes 2 and 3), feathering, shared `mp.Value` paste probability. |  **Training only.** |  Bank crops + target tiles → composited tiles → `loader.py`. Ramp is driven from `main.py`. |  
| `src/data/balanced_sampler.py` (151) |  `BalancedClassSampler` driven by `class_index.json`; factory `create_balanced_sampler`. |  **Training only.** |  Class index → sample weights → loader when `--use-balanced-sampling`. Missing index degrades to uniform. |  
| `src/utils/weights.py` (81) |  Computes inverse-sqrt class weights over triplets with multiprocessing; reads/writes `class_weights.json`. |  **Training only.** |  Triplets → counts → weights file → `main.py`. Worker exceptions are swallowed and return zero counts. |  
| `src/utils/common.py` (13) |  `seed_everything`. |  **Training only.** |  Seed → global RNG state. |  
| `src/utils/visualization.py` (257) |  Renders predictions + GT for gated/hierarchical/plain heads every 5 epochs. |  **Training only.** |  Model forward on a batch → PNG/JPEG in `runs/<name>/vis`. Guards empty gated lists; matplotlib import failures are swallowed by `main.py`. |  
| `src/{,model,data,training,utils}/__init__.py` |  Empty package markers (0 lines). |  Import plumbing. |  Make `src.*` importable when `/opt/damage-triage-former` is on `sys.path`. |  
| `checkpoints/config.json` (48 keys) |  Serialised **training argparse dump** of the released mask-pooling run (`use_gated_head`, `use_sfp`, `sfp_stride=8`, `no_multiscale`, `no_attention_pool`, `use_aux_severity`, plus training hyper-parameters). It is **not** a Hugging Face model config and has no `hidden_size`/`patch_size`. |  Build artefact; read at inference by `build_dtf_model` to choose the head. |  `damage_triage_infer.build_dtf_model` reads only 8 keys; all others are ignored. A **second** config schema exists at repository-root `models/damage_triage/config.json` (model card: `input_shape [3,224,224]`, `mask_pooling:false`, nested `architecture`; mounted read-only at `/app/models/damage_triage`), and `i-dtf/model_loader.py` reads a **third** flat-key interpretation. This three-schema split is the root of the crop-mode loading defect (§19 P0-1). |  
| `requirements.txt` (18) |  Research training pins (torch 2.2, transformers 4.56–5, timm, einops, opencv, torchmetrics 1.3.2). |  Build input for the research image; **not** used by `haste-training` (which uses `docker/training/env/env.yml`) nor by i-dtf (its own requirements). |  Not consumed at runtime; version skew with the runtime torch 2.5.1 is tolerable because checkpoints load as raw state dicts. |  
| `README.md` (135) |  Class definitions, recipe, reported macro-F1 (0.624/0.619), data layout, dataset licence (CC-BY-NC-4.0). |  Human reference. |  References `scripts/train_2gpu.sbatch`, which is **absent** from this tree. |  
| `CITATION.cff` |  Citation metadata; `license: MIT`. |  Human reference. |  — |  
| `LICENSE` |  MIT (Yiming Xiao, 2025) covering **code only**. |  Legal. |  Weights and the DamageTriage-Bench dataset are CC-BY-NC-4.0; the DINOv3 backbone licence is not stated anywhere in the repo. |  
| `.gitignore` |  Ignores `data/`, `runs/`, `*.pth`. |  Repo hygiene. |  Prevents accidental checkpoint/dataset commits. |  
  
   
**13.2 What inference actually loads**  
```  
mask-pooling path (hypothetical / AOI mask mode)  
   workflows/damage_triage_infer.build_dtf_model  
     → sys.path.insert(0, $DTF_REPO_ROOT)  
     → src.config ──────────────► resolves BACKBONE (filesystem or HF)  
     → src.model.net.Model ─────► src.model.backbone, src.model.heads,  
                                   src.model.gated_head | hierarchical_head,  
                                   src.model.sfp  
     → torch.load(checkpoint) → load_state_dict(strict=False)  
     → critical-missing check on backbone.* / damage_head.*  
   
 crop-mode path (the shipped checkpoint)  
   workflows/damage_triage_infer.GatedDinoCrop  
     → src.config.BACKBONE only  
     → AutoModel.from_config / AutoConfig.from_pretrained  
     → gate/leaf/aux Linear heads loaded strict=True  
   
```  
Everything under `src/data/`, `src/training/` and `src/utils/{weights,visualization,common}.py` is **dead weight at serving time**. That matters for cold start (the container installs a full training stack) and for the attack surface (unused but importable code).  
**13.3 The head/weight contract**  
- **Released mask-pooling checkpoint**: `backbone.*` + `seg_head.*` + `sfp.*` + `damage_head.*`, gated with auxiliary severity. It requires `inst_masks`, i.e. per-building rasterised masks in tile space.  
- **Released crop-mode checkpoint**: keys `vit.*`, `gate.*`, `leaf.*`, `aux.*`; `strict=True` load into `GatedDinoCrop`; input is one 224×224 building crop, no mask, ImageNet normalisation.  
- **Whole-layer `dtf` pipeline**: calls `build_dtf_model` **without** `allow_crop_mode` (`damage_triage_infer.py:733-735`), so the crop-mode checkpoint raises `RuntimeError("... is a crop-mode checkpoint; this tile-based workflow needs a mask-pooling one.")` at line 260. This is why the fork's UI hides the whole-layer DTF action and why only AOI/region paths work with the shipped weights.  
- **AOI `dtf-aoi`**: passes `allow_crop_mode=True`; if crop mode, each building is cropped and `score_crop` resizes to 224 with ImageNet normalisation. Mask mode divides the rasterised mask by 255 and uses `unit` normalisation.  
- **Region assessment**: passes `allow_crop_mode=True` and forces `imagenet` for crop mode.  
The mismatch is not a bug in one file; it is a **contract drift** between three producers of "what the checkpoint is" and four consumers.  
  
---  
**14.** `ui/` **— every source file**  
**14.1 Scope and build**  
  
|  |  |  
| - | - |  
| **Item** |  **Detail** |  
| Framework |  React 18.3 + Vite 8, Bootstrap 5.3, Fluent UI 8, React Router 6, Azure Maps drawing tools, Chart.js/D3, geotiff, pmtiles. |  
| Entry |  `index.html` → `src/main.jsx` → `App.jsx` → `AppBody.jsx` route table. |  
| Dev container |  `ui/Dockerfile` runs the **Vite dev server** on 4280 (`CMD npm run dev -- --host --port 4280`), not a production build. `docker/ui/Dockerfile` (nginx, port 5000) is not used by compose. |  
| Env |  `VITE_API_URL`, `VITE_STORAGE_APIM_URL`, `VITE_PROJECT_STORAGE_APIM_URL`, `VITE_TITILER_URL`, `VITE_AZURE_MAPS_CLIENT_ID` from compose. `.env.docker` points at `host.docker.internal:7071/api/` and is overwritten by compose env vars. |  
| Auth |  `apiValidateUser` is a stub that always returns a single `local-user` with role `administrators` (`util/api.js:26-38`). `public/staticwebapp.config.json` still declares SWA AAD login and role gates — the two models coexist and contradict each other. |  
| Hosting |  SPA fallback is only configured in `docker/ui/Dockerfile` (unused); the Vite dev server handles history fallback itself. |  
  
   
**14.2 Routes (**`src/Components/AppBody.jsx:37-104`**)**  
  
|  |  |  |  |  
| - | - | - | - |  
| **Path** |  **Component** |  **Role gate** |  **Purpose** |  
| `/` |  `Home` |  admin/contributor |  Landing page, global stats (`GetDashboardData`), open-project dialog. |  
| `/projects` |  `Projects` |  admin/contributor |  Project list and CRUD. |  
| `/project/:projectId` and `/project/:projectId/:imageLayerId` |  `Project` |  admin/contributor |  Project workspace: layers, models, embedding/DTF rows, uploads, runs. |  
| `/project/:projectId/imageLayer/:imageLayerId` |  `ImageLayer` |  admin/contributor |  Legacy layer detail page (largely superseded by `Project`; see dead code). |  
| `/create-imageLayer/:projectId`, `/edit-imageLayer/:projectId/:imageLayerId` |  `CreateEditImageLayerForm` |  admin/contributor |  Multi-step layer creation/editing (file, URL, imagery source, footprints, DTF modal). |  
| `/labeling-tool/:projectId/:imageLayerId` |  `LabelingTool` |  admin/contributor |  Map drawing/labelling, AOI draw, run DTF-AOI, assessment dashboard. |  
| `/validation/:projectId/:imageLayerId` |  `BuildingValidation` |  admin/contributor |  Per-building prediction review, DTF-AOI review panel, reports. |  
| `/interactive-label/:projectId/:imageLayerId/:modelId` |  `InteractiveLabeler` |  admin/contributor |  Active-learning labeler with GPU logistic regression. |  
| `/visualizer/:projectId/:imageLayerId/:modelId` |  `Visualizer` |  admin/contributor |  Model result visualiser and artifact downloads. |  
| `/help-docs/*` |  `HelpDocs` |  admin/contributor |  Static help pages and guided tours. |  
| `/model-catalog`, `/admin-users`, `/admin-source-types`, `/admin-labeling-tool` |  `ModelCatalog`, `AdminUsers`, `AdminSourceTypes`, `AdminLabelingTool` |  **administrators only** |  Global admin surfaces. |  
| `*` |  `Error404` |  none |  Fallback. |  
  
   
Note the role gate is evaluated only after `apiValidateUser` resolves; because that function is a stub, `userRoles` is always `["administrators"]` and every authenticated-looking surface is reachable without login.  
**14.3 API surface used by the UI**  
  
|  |  |  |  
| - | - | - |  
| **Domain** |  **Endpoints (Functions route names)** |  **Callers** |  
| User/settings |  `GetUsers`, `PutUser`, `DeleteUser`, `GetAdminSettings`, `GetAzureMapsToken` |  `App.jsx`, `AdminUsers`, `AdminLabelingTool`, `UserMetadataCreator`, `util/azureMapsAuth.js` |  
| Projects |  `PutProject`, `DeleteProject`, `GetProjectDetails`, `GetDashboardData` |  `Projects`, `Project`, `Home/OpenProject`, `ProjectInsightsBuble` |  
| Layers/imagery |  `PutLayer`, `PutLayerImageryUpdate`, `GetLayerDetailView`, `UploadFileByChunk` |  `CreateEditImageLayer*`, `UpdateImageryModal`, `CreateEditImageLayerFileUploader` |  
| Buildings/labels |  `GetBuildingsInAoi`, `PutBuildingPredictions`, `PutBuildingValidation`, `PutInteractiveLabels`, `PutLabelsFromLabelTool` |  `LabelingTool`, `BuildingValidation`, `InteractiveLabeler`, `Visualizer` |  
| Model runs |  `PutRunDTFInferenceQueueMessage`, `PutRunDTFAoiInferenceQueueMessage`, `PutRunEmbeddingQueueMessage`, `PutCancelModelQueueMessage` |  `CreateEditDTFModal`, `ProjectManagement/*Row`, `RunDtfAoiButton` |  
| DTF AOI review |  `PutBuildingDTFReview`, `GetAoiPredictionsGpkg` |  `DtfAoiReviewPanel`, `ModelResultsButton` |  
| Assessments |  `PutAoiAssessment`, `GetAoiAssessmentPreview`, `GetAoiAssessment`, `GetAoiAssessmentStatistics`, `GetAoiAssessmentMapData`, `PutCancelAoiAssessment`, `GetBuildingAssessmentHistory` |  `AoiAssessmentModal`, `AoiAssessmentDashboard`, `AoiAssessmentMapLayer`, `BuildingAssessmentResultPanel`, `AssessmentReportModal` |  
| Model catalog |  `GetModelCatalog`, `PutModelCatalog`, `DeleteModelCatalog`, `DeleteModel` |  `ModelCatalog`, `ModelCatalogRow`, `CreateEditBaseModelModal`, `CreateEditModelCheckpoint` |  
| Artifacts/reports |  artifact download endpoints, validation-report endpoints |  `ModelResultsButton`, `ValidationReportModal`, `InfoPanel` |  
  
   
**14.4 File inventory (grouped; every file named)**  
***Entry, shell and global state***  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Position** |  **Data flow** |  
| `index.html` |  Vite HTML shell, root div, module script. |  Build entry. |  Loads `src/main.jsx`. |  
| `src/main.jsx` |  Mounts React root, `BrowserRouter`, `AppContext`. |  Boot. |  `App.jsx` ← DOM. |  
| `src/App.jsx` |  Loads user (`apiValidateUser`), settings, Bootstrap breakpoints, Azure Maps auth; renders header/body/footer, global modal host, guided tour. |  Shell. |  `apiValidateUser` → `appParams` context → `AppBody` route gating. |  
| `src/AppContext.jsx` |  React context for `appParams` and `setAppParams` (user, roles, settings, breakpoints, loading). |  Cross-cutting. |  Providers → all components. |  
| `src/AppHelper.js` |  Helpers for shell-level state (settings bootstrap, breakpoint detection). |  Shell. |  `App.jsx`. |  
| `src/Components/AppBody.jsx` |  Route table + role gates. |  Routing. |  URL → page component. |  
| `src/Components/AppHeader.jsx` |  Top nav, project/layer breadcrumbs, user menu, help links. |  Shell. |  `appParams`; navigation. |  
| `src/Components/AppFooter.jsx` |  Footer, version/link text. |  Shell. |  Static. |  
| `src/Components/AppPanel.jsx` |  Generic panel/card wrapper. |  Presentation. |  Children. |  
| `src/Components/SectionModal.jsx` |  Generic modal section container. |  Presentation. |  Children. |  
| `src/Components/Section/SectionHeader.jsx` |  Section heading row. |  Presentation. |  Children. |  
| `src/Components/SettingsModal.jsx` |  Per-user settings (items per page, guided tour). |  Shell. |  `PutUser` → `appParams`. |  
| `src/Components/Error404.jsx` |  404 page. |  Routing fallback. |  Static. |  
| `src/Components/GuidedTour.jsx`, `GuidedTourHelper.js` |  Step-by-step product tour; helper holds step definitions and target selectors. |  Shell overlay. |  `appParams.userSettings.disableGuidedTour`; DOM targets. |  
| `src/Components/HelpDocs.jsx` + `HelpDocs/HelpDocsNav.jsx`, `HelpDocsOverview.jsx`, `HelpDocsOverviewBubble.jsx`, `HelpDocsImageLayers.jsx`, `HelpDocsLabeling.jsx`, `HelpDocsModelCatalog.jsx`, `HelpDocsModelTraining.jsx`, `HelpDocsProjects.jsx`, `HelpDocsResults.jsx` |  Static documentation pages, per-topic, with embedded images/videos from `src/assets/helpDocs`. |  Help route. |  Static assets; no API. |  
  
   
***Utilities***  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Position** |  **Data flow** |  
| `src/util/api.js` |  All HTTP verbs; URL builder adds APIM `subscription-key`; `LOCAL_USER` stub; `apiValidateUser`. |  Every network call. |  Component → `fetch` → Functions API. **`apiPost` replaces every server error with "Error uploading chunk."**; `apiPut` throws a `Response` object on non-200/409; `apiGet`/`apiDelete` also flatten messages. |  
| `src/util/azureMapsAuth.js` |  Fetches Azure Maps token via `GetAzureMapsToken`, refreshes before expiry. |  Map components. |  API → `atlas` auth options. |  
| `src/util/basemap.js` |  Builds Azure Maps style/basemap options from env. |  Map components. |  Env → map style. |  
| `src/util/blobUrl.js` |  Normalises blob/artifact URLs (SAS, relative-to-absolute). |  Artifact links, imagery URLs. |  API payload → `<img>`/download. |  
| `src/util/conversion.js` |  GeoJSON/geometry/coordinate conversions, bbox helpers. |  Map + API payloads. |  Geometries. |  
| `src/util/file.js` |  Chunking, file size/type checks, download helpers. |  Upload/download. |  `File` → chunks → `UploadFileByChunk`. |  
| `src/util/validation.js` |  Schema/field validation for forms. |  Forms. |  Input → error list. |  
| `src/assets/js/azure-maps-image-exporter.js` |  Screenshot/export helper for maps. |  Visualiser export. |  Map canvas → PNG. |  
| `src/assets/js/azure-maps-swipe-map.min.js` |  Vendored swipe-map control. |  `Visualizer` comparison mode. |  Two layers. |  
| `src/assets/json/settings.json`, `src/assets/json/world.geojson` |  Default UI settings and world outline. |  Boot/map. |  Static. |  
| `src/assets/css/*` (6 files: `root-style`, `style`, `labels`, `drawingToolbar`, `progress-bar`, `visualizer`) |  Global and component styles. |  Presentation. |  — |  
| `src/assets/helpDocs/**` |  Screenshots and MP4s referenced by help pages. |  Help route. |  Static. |  
  
   
***Project and layer management***  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Position** |  **Data flow** |  
| `src/Components/Projects.jsx` |  Project list, pagination, create/delete. |  `/projects`. |  `GetDashboardData`/project endpoints → rows. |  
| `src/Components/Project.jsx` |  Workspace orchestrator: layer list, model lists, uploads, polling, modal host. |  `/project/:id[/:layerId]`. |  `GetLayerDetailView` (20 s polling), `GetProjectDetails`; child rows/modal events. |  
| `src/Components/ProjectManagement/ProjectHeader.jsx` |  Project title/actions. |  Project. |  Props. |  
| `src/Components/ProjectManagement/ProjectsHeader.jsx` |  Projects list header/search. |  Projects. |  Props. |  
| `src/Components/ProjectManagement/ProjectRow.jsx` |  One project row/card. |  Projects. |  Project object. |  
| `src/Components/ProjectManagement/LayerRow.jsx` |  Layer summary, status, actions (label, visualise, DTF, update imagery). Hides whole-layer DTF when the checkpoint is crop-mode. |  Project. |  Layer object → navigation/queue calls. |  
| `src/Components/ProjectManagement/ImageLayerInfoMobile.jsx` |  Mobile layer card; same gating as `LayerRow`. |  Project (mobile). |  Layer object. |  
| `src/Components/ProjectManagement/ModelRow.jsx`, `ModelRowMobile.jsx` |  Model rows (base/DTF/embedding) with status and result actions. |  Project. |  Model object. |  
| `src/Components/ProjectManagement/BaseModelRow.jsx`, `DTFModelRow.jsx`, `EmbeddingModelRow.jsx` |  Type-specific row variants. |  Project. |  Model object. |  
| `src/Components/ProjectManagement/ModelResultsButton.jsx` |  Results menu: visualiser, GeoPackage/artifacts, cancel, reports. |  Project. |  `GetAoiPredictionsGpkg`, artifact endpoints. |  
| `src/Components/ProjectManagement/ModelCatalogHeader.jsx`, `ModelCatalogRow.jsx`, `ModelCatalogAdditionalInfoModal.jsx` |  Catalog list/header/detail modal. |  Model catalog. |  `GetModelCatalog`. |  
| `src/Components/ProjectManagement/NoResults.jsx` |  Empty-state placeholder. |  Lists. |  Props. |  
| `src/Components/ProjectManagement/SourceTypeRow.jsx`, `UserRow.jsx` |  Admin list rows. |  Admin pages. |  `GetUsers`/settings. |  
| `src/Components/ProjectManagement/ProjectInsightsBuble.jsx` (under `Home/`) |  Dashboard stat bubble. |  Home. |  `GetDashboardData`. |  
| `src/Components/Home/OpenProject.jsx` |  Project picker dialog. |  Home. |  Project list. |  
| `src/Components/Home.jsx` |  Landing/dashboard. |  `/`. |  `GetDashboardData`. |  
| `src/Components/StartProjectButton.jsx` |  Create-project CTA. |  Home/Projects. |  `PutProject`. |  
| `src/Components/UpdateImageryModal.jsx` |  Re-upload/update imagery for an existing layer. |  Layer actions. |  `PutLayerImageryUpdate`. |  
| `src/Components/CreateEditProjectModal.jsx` + `CreateEditProjectModalHelper.js` |  Project create/edit form and validation. |  Projects. |  `PutProject`. |  
| `src/Components/CreateEditImageLayerForm.jsx` |  Multi-step layer wizard shell. |  Create/edit layer. |  Child steps; `PutLayer`. |  
| `src/Components/CreateEditImageLayerFormFile.jsx` |  Local file source step. |  Wizard. |  `File` → chunk uploader. |  
| `src/Components/CreateEditImageLayerFileUploader.jsx` |  4 MB chunk upload with progress and per-chunk retry. |  Wizard/update. |  `File` → `UploadFileByChunk` → layer payload. All chunk failures surface as "Error uploading chunk." |  
| `src/Components/CreateEditImageLayerFormURL.jsx`, `CreateEditImageLayerURL.jsx` |  Remote URL source step/validator. |  Wizard. |  URL allowlist check (client-side) → `PutLayer`. |  
| `src/Components/CreateEditImageLayerFormImagerySources.jsx` |  Imagery source/event selection. |  Wizard. |  Admin settings → source id. |  
| `src/Components/CreateEditImageLayerFormBuildingFootprints.jsx` |  Footprint source selection (default/TWDB/custom). |  Wizard. |  Footprint settings → `PutLayer`. |  
| `src/Components/CreateEditImageLayerHelper.js` |  Wizard state, defaults, payload assembly. |  Wizard. |  Step state → `PutLayer`. |  
| `src/Components/CreateEditDTFModal.jsx` |  DTF model creation/run. **No importer from `Project`/`LayerRow` in this fork**, so whole-layer DTF is unreachable from the UI. |  Project (dead). |  Would call `PutRunDTFInferenceQueueMessage`. |  
| `src/Components/CreateEditEmbeddingModal.jsx` |  Embedding model creation/run. |  Project. |  `PutRunEmbeddingQueueMessage`. |  
| `src/Components/CreateEditBaseModelModal.jsx` |  Base model create/edit. |  Model catalog/project. |  `PutModelCatalog`. |  
| `src/Components/CreateEditModelCheckpoint.jsx` + `CreateEditModelCheckpointHelper.js` |  Checkpoint URL/upload and metadata. |  Model forms. |  `PutModelCatalog`. |  
| `src/Components/CreateEditSourceTypeModal.jsx` |  Admin source-type create/edit. |  Admin. |  `PutModelCatalog`/settings. |  
| `src/Components/CreateEditUserModal.jsx` + `CreateEditUserModalHelper.js` |  Admin user create/edit. |  Admin. |  `PutUser`. |  
| `src/Components/UserMetadataCreator.jsx` |  Editable user metadata key/value. |  Settings/user forms. |  `PutUser`. |  
| `src/Components/PrimaryClassCreator.jsx` |  Primary damage class editor. |  Labeling tool. |  `PutLabelsFromLabelTool`. |  
| `src/Components/ModelCatalog.jsx` |  Catalog page. |  `/model-catalog`. |  `GetModelCatalog`; delete/put. |  
| `src/Components/AdminUsers.jsx` |  User admin table. |  `/admin-users`. |  `GetUsers`, `PutUser`, `DeleteUser`. |  
| `src/Components/AdminSourceTypes.jsx` |  Source-type admin. |  `/admin-source-types`. |  Admin settings. |  
| `src/Components/AdminLabelingTool.jsx` + `AdminLabelingTollHelper.js` |  Labeling-tool settings. Save button is wired to no handler. |  `/admin-labeling-tool`. |  Would `PutModelCatalog`/settings. |  
| `src/Components/ImageLayer.jsx` |  Legacy layer detail page. Not reachable from the route table (route still exists, but no inbound links). |  Legacy. |  `GetLayerDetailView`. |  
  
   
***Map/labelling and validation***  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Position** |  **Data flow** |  
| `src/Components/LabelingTool/LabelingTool.jsx` |  Map page orchestrator: imagery layers, drawing, AOI selection, run DTF, assessment dashboard. |  `/labeling-tool/...`. |  Azure Maps → `GetBuildingsInAoi` → `PutBuildingPredictions`; AOI → `PutAoiAssessment`; poll 3 s. Map instances are **not disposed** on unmount. |  
| `src/Components/LabelingTool/LabelingToolHelper.js` |  Map init, layer/crop source construction, geometry helpers. |  Labeling. |  Layer config → map sources. |  
| `src/Components/LabelingTool/LabelingToolLeftPanel.jsx`, `LeftPanel.jsx` |  Tool palette / layer controls. |  Labeling. |  Props. |  
| `src/Components/LabelingTool/LabelingToolRightPanel.jsx`, `RightPanel.jsx` |  Building/label inspector. |  Labeling. |  Selected building + labels. |  
| `src/Components/LabelingTool/DrawingToolbar.jsx` |  Polygon/rectangle drawing controls. |  Labeling. |  Drawing manager events. |  
| `src/Components/LabelingTool/AoiDrawTool.jsx` |  AOI polygon draw/edit. |  Labeling/assessment. |  AOI geometry → assessment modal. |  
| `src/Components/LabelingTool/BuildingSelectTool.jsx` |  Building click/select. |  Labeling. |  Feature → right panel. |  
| `src/Components/LabelingTool/UndoRedo.js` |  Label/geometry undo stack. |  Labeling. |  Action history. |  
| `src/Components/LabelingTool/SplitShape.jsx` |  Split polygon tool. |  Labeling. |  Geometry → labels. |  
| `src/Components/LabelingTool/RunDtfAoiButton.jsx` |  Triggers AOI DTF on the current selection. |  Labeling. |  `PutRunDTFAoiInferenceQueueMessage`. |  
| `src/Components/LabelingTool/AoiAssessmentModal.jsx` |  Large-AOI assessment submission form (name, imagery, policy). |  Labeling. |  `PutAoiAssessment` → assessment id. |  
| `src/Components/LabelingTool/AoiAssessmentDashboard.jsx` |  Assessment progress/statistics panel. |  Labeling. |  `GetAoiAssessment`, `GetAoiAssessmentStatistics`, `PutCancelAoiAssessment`; 3 s polling without abort. |  
| `src/Components/LabelingTool/AoiAssessmentMapLayer.jsx` |  Coloured assessment result layer. |  Labeling map. |  `GetAoiAssessmentMapData` → map source. |  
| `src/Components/LabelingTool/BuildingAssessmentResultPanel.jsx` |  Per-building assessment result/history. |  Labeling/validation. |  `GetBuildingAssessmentHistory`. |  
| `src/Components/BuildingValidation/BuildingValidation.jsx` |  Validation page: imagery + prediction layers, review actions, report modals. |  `/validation/...`. |  `GetBuildingsInAoi`, `PutBuildingValidation`, `GetAoiAssessment...`; map not disposed. |  
| `src/Components/BuildingValidation/BuildingValidationRightPanel.jsx` |  Selected-building detail and label controls. |  Validation. |  Building + prediction. |  
| `src/Components/BuildingValidation/DtfAoiReviewPanel.jsx` |  DTF-AOI accept/reject/correct workflow. |  Validation. |  `PutBuildingDTFReview`. |  
| `src/Components/BuildingValidation/AssessmentReportModal.jsx`, `ValidationReportModal.jsx` |  Assessment/validation report viewers. |  Validation. |  Report endpoints. |  
| `src/Components/InteractiveLabeler/InteractiveLabeler.jsx` (1,959 lines) |  Active-learning labeler: embedding display, GPU logistic regression suggestions, label capture, retrain. |  `/interactive-label/...`. |  `PutInteractiveLabels`, labels endpoint; map not disposed. |  
| `src/Components/InteractiveLabeler/gpuLogreg.js` |  GPU-accelerated logistic regression (WebGL/JS). |  Interactive labeler. |  Embeddings → probabilities → suggestions. |  
| `src/Components/InteractiveLabeler/interactiveModel.js` |  Model state, feature/label buffers, prediction glue. |  Interactive labeler. |  API labels/embeddings → GPU logreg. |  
| `src/Components/Visualizer/Visualizer.jsx` (619 lines) |  Result map, class legend, comparison/swipe, artifact downloads. |  `/visualizer/...`. |  `GetAoiPredictionsGpkg`, artifact endpoints; map not disposed. |  
| `src/Components/Visualizer/VisualizerImageryControls.jsx` |  Layer opacity/visibility controls. |  Visualizer. |  Map layers. |  
| `src/Components/Visualizer/VisualizerInformationMobile.jsx` |  Mobile info sheet. |  Visualizer. |  Props. |  
| `src/Components/Visualizer/InfoPanel.jsx` |  Prediction stats/info. |  Visualizer. |  API stats. |  
| `src/Components/Visualizer/Labels.jsx` |  Class labels/legend. |  Visualizer. |  Class metadata. |  
| `src/Components/OtherComponents/ColorPicker.jsx` |  Colour chooser. |  Labeling/admin. |  Palette. |  
| `src/Components/OtherComponents/ErrorMessage.jsx` |  Error banner/dialog. |  Global. |  Error string. |  
| `src/Components/OtherComponents/Loading.jsx` |  Spinner. |  Global. |  `appParams.isLoading`. |  
| `src/Components/OtherComponents/MessageProgressBar.jsx` |  Determinate/indeterminate progress with message. |  Uploads/runs. |  Chunk/model progress. |  
| `src/Components/OtherComponents/ProgressBar.jsx` |  Progress bar primitive. |  Various. |  Percent. |  
| `src/Components/OtherComponents/ModelCancelButton.jsx` |  Cancel queued/running model job. |  Project rows. |  `PutCancelModelQueueMessage`. |  
| `src/Components/OtherComponents/ModelResultsStatusIndicator.jsx` |  Status chip for model runs. |  Project rows. |  Model status string (brittle substring parsing). |  
| `src/Components/OtherComponents/StatusIndicator.jsx`, `StatusIndicatorModal.jsx` |  Generic status chip/modal. |  Various. |  Status. |  
| `src/Components/OtherComponents/PaginationControls.jsx` |  Page navigation. |  Lists. |  Page state. |  
  
   
Static/public assets: `public/assets/img/*` (favicon, legend, guided-tour images), `public/assets/geo/world.geojson`, `public/assets/json/settings.json`, `public/assets/js/azure-maps-*.js`, `public/staticwebapp.config.json` (AAD + role gates, unused in local compose but shipped).  
**14.5 User-visible sequence: AOI DTF + review**  
```  
LabelingTool  
   ├─ AoiDrawTool → AOI polygon  
   ├─ RunDtfAoiButton → PUT PutRunDTFAoiInferenceQueueMessage  
   │     backend: queue → dtf-aoi worker → per-building crops → hg_buildings review_status  
   ├─ poll GetLayerDetailView / GetBuildingsInAoi (3 s)  
   └─ BuildingValidation  
         ├─ GetBuildingsInAoi (viewport)  
         ├─ DtfAoiReviewPanel → PUT PutBuildingDTFReview  
         └─ reports → AssessmentReportModal / ValidationReportModal  
   
```  
**14.6 User-visible sequence: large-AOI assessment**  
```  
AoiAssessmentModal  
   ├─ PUT PutAoiAssessment {aoi, imagery, policy}  
   │     backend: validate → discover → tile → batches → units → workers → aggregate  
   ├─ AoiAssessmentDashboard  
   │     ├─ GET GetAoiAssessment            (3 s poll, no AbortController)  
   │     ├─ GET GetAoiAssessmentStatistics  
   │     └─ PUT PutCancelAoiAssessment  
   └─ AoiAssessmentMapLayer → GET GetAoiAssessmentMapData  
         BuildingAssessmentResultPanel → GET GetBuildingAssessmentHistory  
   
```  
**14.7 UI findings**  
These are consolidated into §19; the notable ones specific to the UI layer:  
1. **Error messages are destroyed** in `util/api.js`. `apiPost` catches *everything* (including a successfully parsed server `{error}`) and rethrows `"Error uploading chunk."`, so an invalid geometry, auth failure and network loss are indistinguishable. `apiPut` throws a `Response` object, which stringifies as `[object Response]`.  
2. **Polling without cancellation**: `Project.jsx` polls layer details every 20 s and `AoiAssessmentDashboard` every 3 s; both continue after unmount and can interleave stale responses. There is no backoff.  
3. **Map instances are not disposed** in `LabelingTool`, `BuildingValidation`, `InteractiveLabeler` and `Visualizer`; repeated navigation leaks WebGL contexts and Atlas token renewals.  
4. **Dead/unreachable UI**: `CreateEditDTFModal` has no importer, `ImageLayer.jsx` has no inbound link, and `AdminLabelingTool`'s Save has no handler.  
5. **State mutation**: several components write directly into objects from `appParams`/API payloads instead of copying, so a failed save leaves the UI inconsistent with the server.  
6. **Auth mismatch**: the client is a hard-coded admin stub while the shipped SWA config enforces AAD; deploying the client to SWA would show routes that the role gate then blocks (or vice versa, depending on how the Functions app authenticates).  
7. **Fixed sleeps after deletes** (`sleep()` + refetch) and full `navigate(0)`/`location.reload()` calls make the UI slower and mask race conditions rather than fixing them.  
8. **Brittle status parsing** by substring on backend status strings couples the UI to vocabulary in `core/models/projects.py`.  
9. **Mobile parity gap**: the mobile layer card omits the `dtf-aoi` action.  
10. **Missing/typo'd keys** in guided-tour definitions and several unvalidated event payloads mean malformed map/API events can throw inside render.  
**15. Database migrations** `db/migrations/`  
Migrations are bind-mounted to `/docker-entrypoint-initdb.d` (`docker/docker-compose.yml:136-138`), so they run **only when the PostGIS data volume is first initialised**. Several are re-applied at runtime by `hastegeo` (`postgis_query.ensure_run_results_table`, `imagery_catalog._ensure_columns`, `postgis_load._ensure_imagery_update_column`, `current_damage.ensure_schema`), with process-global "already done" flags. There are **no ENUM types**; every "enum" is TEXT with a code-enforced vocabulary.  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Objects** |  **Purpose** |  **Consumers (read → write)** |  
| `0000_enable_postgis.sql` |  `CREATE EXTENSION IF NOT EXISTS postgis` |  Must sort before geometry columns. |  All geometry DDL. |  
| `0001_hg_buildings.sql` |  `hg_buildings` PK `(image_layer_id, building_id)`; GIST on geometry; BTREE on `image_layer_id`. Columns: `project_id`, `geometry GEOMETRY(Geometry,4326)`, `subtype`, `class`, `damage_class`, `damage_class_name`, `confidence`, `mask_url`, `crop_url`, `review_status`, `updated_at`, `footprint_source`, `footprint_version`. |  Canonical per-building table; one row per footprint per layer. |  `postgis_load.load_footprints_into_postgis` / `load_footprints_streaming` (DELETE then INSERT); `postgis_query.buildings_in_aoi`, `count_buildings_in_aoi`, `layer_buildings_feature_collection`, `update_building_predictions`, `set_review_status`, `get_building_history`; `current_damage.publish_current_results`; `assessment_store.eligible_buildings_with_overlap`, `get_building_geometries`; `twdb_buildings.load_twdb_buildings`. |  
| `0002_aoi_assessments.sql` |  `hg_imagery`, `hg_assessments`, `hg_batches`, `hg_processing_units`, `hg_unit_buildings`, `hg_prediction_evidence`, `hg_damage_assessments`; adds `hg_buildings.footprint_source`/`footprint_version` and `hg_imagery.transform`/`hg_prediction_evidence.valid_pixel_fraction`; GIST/BTREE indexes. |  The complete Large-AOI state machine. `hg_assessments.status` vocabulary is `submitted, validating, rejected, discovering, tiling, queued, running, aggregating, completed, completed_with_errors, failed, cancelling, cancelled` (`core/models/projects.py:909-924`). `hg_processing_units.kind` is `grid` or `building_centred`; `hg_damage_assessments.result_status` is `completed`, `uncertain`, `insufficient_data` or `failed`. `damage_percentage` is **always NULL for DTF**. |  Written by `assessment_store.create_assessment`, `insert_batches`, `insert_processing_units`, `insert_unit_buildings`, `claim_next_batch`, `complete_batch`, `fail_batch`, `upsert_prediction_evidence`, `upsert_damage_assessments`; read by `processors/aoi_assessment.py`, `region_windows`, `region_aggregate`, `current_damage`, `postgis_query.get_assessment_*`. |  
| `0003_current_building_damage.sql` |  `hg_buildings.last_assessment_id`, `result_status`; partial GIST index on damaged rows (`WHERE damage_class IS NOT NULL`). |  Makes the "current" damage view cheap for viewport queries. |  `current_damage._query_rows` filters on `damage_class IS NOT NULL`; `publish_current_results` updates the columns. |  
| `0004_building_attributes.sql` |  `hg_buildings.attributes JSONB`. |  Per-building source attributes (height, occupancy, footprint metadata). |  Footprint loaders write; building detail/history read. |  
| `0005_dtf_run_results.sql` |  `hg_dtf_run_results` PK `(model_id, building_id)`, `probabilities JSONB`, `crop_url`, `mask_url`, index `(image_layer_id, building_id)`. |  Per-model DTF-AOI output history, independent of the assessment state machine. |  Written by `postgis_query.record_run_results` (from `processors/damage_triage_aoi.py`); read by `get_building_history`; table also created on demand by `ensure_run_results_table`. |  
| `0006_imagery_updates.sql` |  `hg_imagery.active BOOLEAN`, `retired_at`; `hg_buildings.imagery_updated_at`; partial index on active imagery. |  Supports replacing imagery on an existing layer without losing history: old rows are retired, new rows become active, buildings are marked dirty. |  `imagery_catalog.deactivate_layer_imagery` / `register_imagery_record`; `postgis_load.mark_imagery_updated`; `imagery_update` processor. |  
  
   
**Structural observations**  
- The schema is wide but coherent: `hg_buildings` is the join hub; assessments reference buildings by `(assessment_id, building_id)`; batches and units carry lease/attempt state directly in the row.  
- There is **no foreign key** from `hg_buildings` to `hg_imagery`; an imagery replacement leaves building rows whose `imagery_updated_at` is the only reconciliation signal.  
- Lease expiry is implemented three times (`assessment_store.expire_leases`, batch requeue logic, unit requeue logic) with slightly different predicates.  
- Status transitions are plain `UPDATE`s without a compare-and-set guard in several paths, so a late worker can overwrite `cancelled` with `completed`.  
- Runtime DDL in request paths means a role without `CREATE TABLE`/`ALTER TABLE` will fail on the first DTF-AOI or assessment request, even though migrations succeeded at bootstrap.  
  
---  
**16. Docker, environments and setup**  
**16.1 Compose topology (**`docker/docker-compose.yml`**, project** `haste`**)**  
  
|  |  |  |  |  |  |  
| - | - | - | - | - | - |  
| **Service** |  **Image / build** |  **Port** |  **Volumes** |  **Depends on** |  **Role** |  
| `azurite` |  `docker/emulators/Dockerfile`, pinned 3.37.0 |  10000/10001/10002 |  `azurite-data` |  — |  Blob/queue/table emulator. |  
| `data-init` |  `docker/data-init/Dockerfile` |  — |  `azurite-data`, `../data`, `../../../models` (ro) |  `azurite` |  One-shot: creates containers/queues/CORS, uploads admin settings, copies the DTF checkpoint + DINOv3 backbone into the volume, `chmod 777`. `restart: "no"`. |  
| `api-proxy` |  nginx 1.21.6 |  **7071→80** |  `./nginx.conf` (ro) |  `hastefuncapi` |  Single origin for `/api/` and `/api/titiler/`; wildcard CORS; 20 MB body cap. |  
| `titiler` |  `docker/titiler/Dockerfile` (titiler 0.18.6) |  **8000** |  — |  `azurite` |  Imagery tile server (COG). |  
| `postgis` |  `postgis/postgis:16-3.4` |  **127.0.0.1:5433→5432** |  `postgis-data`, `../db/migrations` (ro) |  — (healthcheck) |  Catalog, buildings, assessments. |  
| `hastefuncapi` |  `docker/api/Dockerfile` |  8080 internal |  `../data`, docker socket, DTF checkpoint (ro) |  azurite, data-init (completed), titiler, postgis (healthy) |  Azure Functions HTTP API (all user-facing operations). |  
| `hastefuncqueues` |  same Dockerfile |  — |  `../data`, `azurite-data`, docker socket, DTF checkpoint (ro), HF cache |  same four |  Nine queue triggers; `RUNNER_TYPE=local` by default. |  
| `ui` |  `build: ../ui` |  **4280** |  — |  api + queues + titiler |  Vite dev server. |  
| `training_image` |  `docker/training/Dockerfile`, `profiles: ["tasks"]` |  — |  — |  — |  Build-only task image used by `LocalRunner`/Batch. |  
| `imageryprep_image` |  `docker/imageryprep/Dockerfile`, `profiles: ["tasks"]` |  — |  — |  — |  Build-only imagery-prep task image. |  
  
   
Named network `haste_default` and volumes `haste_azurite-data`/`haste_postgis-data` are pinned because `LocalRunner` attaches spawned containers by those exact names (`HASTE_DOCKER_NETWORK`, `HASTE_DOCKER_AZURITE_VOLUME`). A plain `docker compose up` **does not build the task images** (they are in the `tasks` profile); the worker will then fail to start containers unless the images were built explicitly.  
**16.2 File-by-file**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Position** |  **Data flow / notes** |  
| `docker/docker-compose.yml` |  Base stack (above). |  Deployment entry. |  Builds images, wires env, mounts socket and volumes. |  
| `docker/docker-compose.gpu.yml` |  Overlay: `HASTE_ENABLE_GPU=1`, `HASTE_GPU_DEVICES`, `TORCH_VARIANT=cu124`, CUDA `BASE_IMAGE`. |  Dev GPU deployment. |  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`. |  
| `docker/docker-compose.hostcuda.yml` |  Overlay: host CUDA mount (`HASTE_HOST_CUDA_DIR` required), `TORCH_VARIANT=host`. |  Dev on hosts with matching driver. |  Images built this way cannot import torch without the mount. |  
| `docker/.env.example` |  Documents every knob: `HF_TOKEN`, `DOCKER_GID`, PostGIS port, Large-AOI limits, `DTF_DEVICE`, imagery-catalog mode. |  Operator reference. |  Copied to `.env`. In-tree `.env` only sets `DOCKER_GID=0`. |  
| `docker/nginx.conf` |  Reverse proxy + CORS + body limit. |  Edge. |  Browser → nginx → Functions/TiTiler. Explicitly local-dev-only. |  
| `docker/api/Dockerfile` |  Azure Functions Python 3.11 base, installs `requirements.txt`, copies `api/hastefuncapi`, `CMD func host start`. |  API + queue images. |  `EXPOSE 7071` disagrees with compose `ASPNETCORE_URLS=:8080`; cosmetic but misleading. |  
| `docker/emulators/Dockerfile` |  Azurite 3.37.0 with `--skipApiVersionCheck`. |  Emulator image. |  Persists to `azurite-data`. |  
| `docker/data-init/Dockerfile` |  Python 3.11 slim + blob/queue SDKs; bakes admin settings and empty ACL/stats JSON; runs `upload_data.py`. |  Init container. |  See `upload_data.py` below. |  
| `docker/data-init/upload_data.py` |  Waits for Azurite, creates 12 queues (8 work + 4 poison), creates containers with **public blob access** and wildcard CORS, uploads `config_admin_settings.json`, `users_acl.json`, `project_stats.json`. |  Bootstrap. |  Idempotent-ish; re-applies CORS/public access on existing containers. |  
| `docker/titiler/Dockerfile` |  Titiler 0.18.6 image, uvicorn 8000. |  Tile service. |  Reads COGs from Azurite blob URLs. |  
| `docker/imageryprep/Dockerfile` |  Functions base + GDAL/rasterio/geopandas; non-root `appuser`; optional torch layer; CLI shims `prepare-imagery` and `zip-artifacts`; `GDAL_SKIP`. |  Task image. |  Runs `hastegeo.workflows.prepare_imagery` and `zip_artifacts` inside Batch/Local containers. |  
| `docker/imageryprep/requirements.txt` |  Pins pydantic 2.9.2, Azure SDKs, GDAL 3.9.2 wheel from a lab blob store, rasterio 1.3.11, geopandas 1.1.2, pyarrow 23.0.1. |  Build input. |  — |  
| `docker/imageryprep/scripts/entrypoint.sh` |  Sets dirs, runs the requested CLI, fixes permissions. |  Container entry. |  — |  
| `docker/imageryprep/scripts/set_dirs.sh` |  Creates per-task scratch dirs, substitutes the task working dir into the experiment YAML. |  Container boot. |  — |  
| `docker/training/Dockerfile` |  Two-stage AzureML CUDA base; micromamba env from `env.yml`; `TORCH_VARIANT` rewrites (`cu124`/`cpu`/`host`); host-CUDA build validates that no `nvidia-*`/`triton`/non-`+cu` torch remains; installs five CLI shims (`embed-buildings`, `damage-triage-infer`, `damage-triage-aoi-infer`, `damage-triage-region-infer`); copies `hastelib/src/hastegeo` → `/app/hastegeo` and the whole DTF source → `/opt/damage-triage-former` (code only, no checkpoint); `GDAL_SKIP`. |  Task image for embeddings, imagery prep, DTF/assessment units. |  `hastegeo` reads `DTF_REPO_ROOT=/opt/damage-triage-former`; checkpoint is downloaded from HF/Azurite per run. |  
| `docker/training/env/env.yml` |  Conda env: python 3.10, GDAL<3.9, tippecanoe; torch 2.5.1+cu124, lightning 2.6.1, torchgeo 0.6.2, torchmetrics 1.9.0, smp, osgeo, OpenCV-headless, rasterio 1.4.3, `transformers>=4.56,<5`, timm, einops, huggingface_hub. |  Build input. |  The `<5` bound is load-bearing for `DinoV3Backbone`. |  
| `docker/training/code/run_workflow.py` |  Legacy segmentation workflow dispatcher (`--step inference` only): inference → require footprints → merge → visualise → COG. |  Legacy task entry. |  **Not used by DTF**; DTF uses the hastegeo CLI shims. |  
| `docker/training/code/inference.py` |  torchgeo segmentation inference over a grid; snapped patch/stride; no overlap blending (last write wins); 3-class palette. |  Legacy step 1. |  Raster → `*_predictions.tif`. |  
| `docker/training/code/merge_with_building_footprints.py` |  Buffers footprints (0/10/20 m, UTM-correct for geographic CRS), computes damaged fraction, writes GeoPackage. |  Legacy step 2. |  Predictions + footprints → `*_predictions.gpkg`. `unknown_pct` reads class 4, which `inference.py` never emits. |  
| `docker/training/code/output2visualizer.py` |  Thresholds `damage_pct_0m` into 5 classes, writes RGBA visualisation GeoTIFF. |  Legacy step 3. |  GPKG → `*_temp_vis.tif`. Uses `assert` for CRS equality (vanishes under `-O`). |  
| `docker/training/scripts/entrypoint.sh` |  Diagnostics, `PYTHONPATH=/app`, runs command, makes task dir world-readable for the queue worker. |  Container entry. |  `set -e` makes the trailing `EXIT_CODE` capture dead code. |  
| `docker/training/scripts/set_dirs.sh` |  Per-task `TORCH_HOME`/`MPLCONFIGDIR`, creates `masks/`/`images/`, substitutes task dir in YAML. |  Container boot. |  — |  
| `docker/training/scripts/print_gpu_info.py` |  CUDA/cuDNN diagnostics. |  Diagnostics. |  — |  
| `docker/ui/Dockerfile` |  Multi-stage Node 24 build → nginx 1.25, SPA fallback, port 5000. |  **Unused** by compose. |  Only relevant to the ACR/production path. |  
| `docker/ui/Dockerfile.dockerignore` |  Ignore list for the above. |  Build hygiene. |  — |  
| `docker/README.md` |  Operator runbook. |  Documentation. |  Stale in places (documents `docker_default`/`docker_azurite-data`, UI Node 20). |  
| `setup/create_conda_env.sh` |  Create/update `haste_env` from an env file. |  Dev setup. |  — |  
| `setup/create_invitations.sh` |  Bulk SWA invitations via `az rest`; placeholders for every identifier. |  Deployment helper. |  `emails.txt` → `email,url` output. |  
| `setup/start-task.sh` |  Azure Batch node start task: finds an unpartitioned disk, formats/mounts it under `$AZ_BATCH_NODE_ROOT_DIR/haste`, appends fstab, permission test. |  Batch node bootstrap. |  Runs as root; no `set -e`; destructive if the "unpartitioned disk" heuristic is wrong. |  
| `setup/install_python.ps1` |  Windows/macOS Python 3.11 + GDAL bootstrap. |  Dev setup. |  Downloads installers without checksums. |  
| `setup/config_admin_settings.json` |  Double-encoded JSON string with `sourceTypes`, `baseModels`, `labelingToolSettings`, `tileServerSettings: "/api/titiler"`. |  Seed data. |  Uploaded verbatim by data-init; parsed by the API. |  
  
   
**16.3 Repository root files**  
  
|  |  |  |  |  
| - | - | - | - |  
| **File** |  **Functionality** |  **Position** |  **Data flow / notes** |  
| `README.md` |  Project overview, architecture summary, quick start, licences. |  Human entry. |  References `assets/banner.png` and `LICENSE.txt`, neither of which exists in the tree. |  
| `QUICKSTART.md` |  Phased, verify-gated local bring-up runbook written for coding agents. |  Human/agent entry. |  Correctly defers to compose as ground truth, but describes the UI as “Vite + React via SWA CLI” while the compose `ui` service runs the Vite dev server directly. |  
| `package.json`, `package-lock.json` |  Vestigial Node metadata (`name: hastegeo`, `version: 1.4.1`, only `vite` as a dependency). |  Unused by the compose build and by `ui/`. |  Misleading name/version; not referenced by any Dockerfile or script. |  
| `.dockerignore`, `.gitignore` |  Build/repo hygiene. |  Build. |  Exclusions for data, caches, node_modules, logs. |  
| `data/` |  Runtime working data: chunk uploads (`<uuid>/file-chunks/*`), Function logs (`data/logs/<date>/*.log`), generated task artifacts. |  Runtime storage mounted at `/app/data` into API and worker. |  2.7 GB in this working copy; not source, must not be shipped in an image or committed. |  
  
   
**16.4 Entry topology**  
```  
flowchart LR  
   B[Browser :4280] -->|/api/*| N[nginx :7071]  
   B -->|tiles| T[TiTiler :8000]  
   N -->|/api/| F[hastefuncapi :8080]  
   N -->|/api/titiler/| T  
   F --> Q[(Azure Queue / Azurite)]  
   F --> P[(PostGIS :5432)]  
   F --> S[(Blob / Azurite)]  
   Q --> W[hastefuncqueues]  
   W -->|RUNNER_TYPE=local| D[Docker socket]  
   D --> I[haste-training / haste-imageryprep task containers]  
   I --> S  
   I --> P  
   W --> P  
   W --> S  
   
```  
**17. Tests**  
The test inventory and gaps are described in §12. In summary: the suite is strong on **pure geometry/SQL-shape** behaviour and real-raster imagery prep, and weak on **execution boundaries** — there is no test that actually starts a `LocalRunner` container, talks to Azure Batch, exercises a data-layer or artifact-storage implementation end-to-end, downloads the HF checkpoint, runs a real model forward, or drives the React UI. Model-loading tests mock `torch.load`, so the crop-mode/mask-pooling mismatch is invisible to CI. No UI tests or CI workflow exist.  
  
---  
**18. Architectural flow diagrams —** `haste-fork`  
**18.0 Entry points and build path**  
  
|  |  |  |  
| - | - | - |  
| **Trigger** |  **First component** |  **Terminal artifact** |  
| `docker compose up` (build execution) |  Dockerfiles under `docker/`, `ui/` |  Running API/queue/TiTiler/UI services; task images available to the runner |  
| `PUT /api/PutLayer` (HTTP) |  `function_app.py` |  Layer record + `image-queue` message |  
| `PUT /api/PutRunDTF...` / `PutRunDTFAoi...` / `PutRunEmbedding...` |  `function_app.py` |  Model record + corresponding queue message |  
| `PUT /api/PutAoiAssessment` |  `function_app.py` |  `hg_assessments` row + assessment queue message |  
| Queue message |  `api/hastefuncqueues/function_app.py` |  Processor side effects; re-enqueue until terminal |  
| Task container command |  `hastegeo.workflows.*` console script |  Files under `outputs/`, uploaded by the runner |  
  
   
```  
flowchart TD  
   R[Repository: haste-fork] -->|docker compose build| IMG[API/queue images + UI]  
   R -->|profile tasks build| TASKS[haste-training + haste-imageryprep]  
   R -->|hatchling wheel via haste_build.py| WHL[hastegeo wheel + rewritten requirements]  
   IMG --> SVC[Running services]  
   WHL --> TASKS  
   TASKS --> RUN[UnifiedRunner task execution]  
   SVC --> API[hastefuncapi]  
   API --> Q[(Queues)]  
   Q --> W[hastefuncqueues]  
   W --> RUN  
   RUN --> OUT[(Task outputs: COGs, GPKGs, JSONL, manifests)]  
   OUT --> ST[(Blob / local artifacts)]  
   OUT --> DB[(PostGIS)]  
   ST --> API  
   DB --> API  
   
```  
**18.1 Imagery prep (entry:** `PUT /api/PutLayer`**, then** `prepare-imagery` **task)**  
```  
flowchart TD  
   U[UI CreateEditImageLayerForm] -->|PUT PutLayer| A[function_app.create_image_layer]  
   A -->|validate url allowlist, CRS, size| V{valid?}  
   V -- no --> E1[400 error to UI]  
   V -- yes --> L[(data layer: project/layer JSON)]  
   A -->|enqueue| Q1[(image-queue)]  
   Q1 --> W1[ImageryPreProcessor]  
   W1 -->|write imageryprep_config.yaml| CFG[(config artifact)]  
   W1 -->|submit task| R1[UnifiedRunner]  
   R1 -->|LocalRunner / AzureBatch| C1[haste-imageryprep container]  
   C1 --> P1[workflows/prepare_imagery.py]  
   P1 -->|COGs, previews, mosaic, footprints.gpkg, valid mask| OUT[(task outputs)]  
   P1 -->|imagery_manifest.json| OUT  
   W1 -->|poll task status| R1  
   R1 -->|success| W2[ImageryPostProcessor]  
   R1 -->|upload outputs| S2[(blob / artifact storage)]  
   OUT --> W2  
   W2 -->|read manifest| L2[update layer URLs + ready]  
   W2 -->|load footprints| PG1[(PostGIS hg_buildings)]  
   W2 -->|register raster metadata| PG2[(PostGIS hg_imagery)]  
   R1 -->|failure| F1[status failed + error]  
   W1 -->|poison after retries| PQ[(image-queue-poison)]  
   
```  
Key failure divergences: unsupported CRS, missing nodata, footprint/imagery misalignment, GDAL OOM on huge rasters, task timeout, and the queue's at-least-once redelivery causing duplicate prep.  
**18.2 Whole-layer DTF (**`dtf` **queue) — present but unreachable with the shipped checkpoint**  
```  
flowchart TD  
   UI[CreateEditDTFModal - no importer] -.->|dead UI path| API  
   API[function_app.run_dtf] -->|build config| CFG[(imagery, footprints, predictions.gpkg, checkpoint, config)]  
   API -->|enqueue| Q[(dtf-queue)]  
   Q --> W[processors/damage_triage.py]  
   W -->|download HF checkpoint| HF[(best_model.pth + config.json)]  
   W -->|config + checkpoint_cache| R[UnifiedRunner]  
   R --> C[haste-training container]  
   C --> CLI[damage-triage-infer CLI]  
   CLI --> B[workflows/damage_triage_infer.py]  
   B --> M{checkpoint has gate.weight?}  
   M -- yes, crop mode --> ERR[RuntimeError: crop-mode checkpoint needs per-building path]  
   M -- no --> BUILD[build_dtf_model → mask-pooling Model]  
   BUILD --> TILE[1024 px tiles, rasterize per-footprint masks]  
   TILE --> SCORE[score_window, normalization=unit]  
   SCORE --> DEC[decode gate+leaf → 5 classes]  
   DEC --> GPKG[(predictions.gpkg)]  
   GPKG --> UP[upload artifacts]  
   UP --> MODEL[(model.gpkgUrl / predictedDamageLayerUrl)]  
   UP --> S[(blob artifacts)]  
   ERR --> FAIL[task failed → poison]  
   
```  
This diagram is the clearest expression of the central defect: the only shipped checkpoint takes the left branch, and the whole-layer pipeline is wired to the right branch only.  
**18.3 AOI DTF (**`dtf-aoi` **queue)**  
```  
flowchart TD  
   UI[LabelingTool / RunDtfAoiButton] -->|PUT RunDTFAoiInferenceQueueMessage| API[function_app]  
   API -->|resolve AOI buildings| PG[(PostGIS hg_buildings)]  
   API -->|enqueue| Q[(dtf-aoi-queue)]  
   Q --> W[processors/damage_triage_aoi.py]  
   W -->|shortlist building ids + geometry| PG  
   W -->|config: imagery + building list| R[UnifiedRunner]  
   R --> C[haste-training container]  
   C --> CLI[damage-triage-aoi-infer CLI]  
   CLI --> B[workflows/damage_triage_aoi_infer.py]  
   B -->|read COG, crop per building| S[(imagery COG)]  
   B --> M{checkpoint type}  
   M -- crop --> CROP[per-building 224 crop, imagenet]  
   M -- mask --> MASK[mask-pooling, unit norm]  
   CROP --> OUT[class + confidence + probs]  
   MASK --> OUT  
   OUT --> GPKG[(predictions.gpkg + crops)]  
   OUT --> RUN[(hg_dtf_run_results)]  
   OUT --> BLD[(hg_buildings.review_status=pending_review)]  
   BLD --> REV[BuildingValidation / DtfAoiReviewPanel]  
   REV -->|PutBuildingDTFReview| PG  
   W -->|failure| PQ[(dtf-aoi-queue-poison)]  
   
```  
Divergences: building outside every imagery footprint, crop smaller than the minimum pixel threshold, imagery CRS mismatch, and review-status updates racing a re-run.  
**18.4 Large-AOI assessment (**`hg_assessments` **state machine)**  
```  
flowchart TD  
   UI[AoiAssessmentModal] -->|PUT PutAoiAssessment| API[function_app]  
   API -->|area/policy validation| V{valid?}  
   V -- no --> REJ[status rejected]  
   V -- yes --> NEW[(hg_assessments submitted)]  
   NEW --> DISCOVER[discovering: eligible buildings + imagery]  
   DISCOVER --> PLAN[region_windows: grid + building-centred units]  
   PLAN --> TILE[tiling: hg_processing_units + hg_unit_buildings]  
   TILE --> BATCH[insert_batches → hg_batches pending]  
   BATCH --> SCHED[scheduler claims batch FOR UPDATE SKIP LOCKED]  
   SCHED -->|config per batch| R[UnifiedRunner]  
   R --> C[haste-training container]  
   C --> CLI[damage-triage-region-infer CLI]  
   CLI --> EV[(evidence.jsonl)]  
   EV --> UP[upsert_prediction_evidence]  
   UP --> AGG[region_aggregate: per-building winner]  
   AGG --> DA[(hg_damage_assessments)]  
   AGG --> CUR[current_damage.publish_current_results]  
   CUR --> BLD[(hg_buildings current damage)]  
   AGG --> CNT[update_assessment_counts]  
   CNT --> STATUS{all units done?}  
   STATUS -- yes --> DONE[completed / completed_with_errors]  
   STATUS -- no --> WAIT[poll next batch]  
   SCHED -->|lease expired| REQUEUE[requeue_or_fail_batch]  
   REQUEUE --> BATCH  
   C -->|task error| FAILB[fail_batch → retry or failed]  
   DONE --> UI2[AoiAssessmentDashboard / map layer / reports]  
   
```  
Divergences: assessment area above the configured cap, no imagery covering the AOI, zero eligible buildings, every unit failing, cancellation while batches are in flight, and lease expiry causing a unit to be computed twice.  
**18.5 Result hierarchy**  
```  
Layer (project_id, image_layer_id)  
 ├── imagery records            hg_imagery (active/retired)  
 ├── buildings                  hg_buildings (class, damage_class, confidence, review_status)  
 │   ├── attributes             hg_buildings.attributes (JSONB)  
 │   ├── DTF-AOI history        hg_dtf_run_results (per model)  
 │   └── assessment history     hg_damage_assessments (per assessment)  
 ├── embeddings                 artifacts + model records  
 ├── whole-layer DTF            predictions.gpkg + artifacts (unreachable with shipped weights)  
 ├── AOI DTF                    predictions.gpkg + crops + review status  
 └── assessments                hg_assessments  
     ├── batches                hg_batches  
     ├── units                  hg_processing_units + hg_unit_buildings  
     ├── evidence               hg_prediction_evidence  
     └── results                hg_damage_assessments  
 Artifacts (blob/local)  
 ├── prepared COGs, tiles, footprints.gpkg  
 ├── model checkpoints, configs, metrics, logs  
 ├── zipped layer exports  
 └── validation reports  
   
```  
**18.6 Where success diverges into failure**  
  
|  |  |  |  
| - | - | - |  
| **Stage** |  **Divergence** |  **Observable state** |  
| HTTP API |  Auth/validation/size |  4xx, no row created |  
| Queue enqueue |  Storage unavailable |  API 5xx after work may already be queued |  
| Worker pickup |  Poison after N deliveries |  `*-poison` queue, model status failed |  
| Runner start |  Image missing (tasks profile not built), Docker socket absent, GPU requested without device |  task `failed` or worker exception |  
| Task execution |  CRS mismatch, OOM, checkpoint incompatible, corrupt raster |  `status.json` failed, logs scraped |  
| Result upload |  Partial upload, permissions |  status failed, orphaned artifacts |  
| DB write |  Runtime DDL denied, constraint conflict |  task succeeds but DB stale |  
| Aggregation |  Missing evidence, lease expiry |  `completed_with_errors`, `insufficient_data` |  
| UI |  Error string swallowed, polling stopped early |  stale "processing" or generic "Error uploading chunk." |  
  
   
  
---  
**19. Gap analysis**  
**19.1 Logical bottlenecks**  
  
|  |  |  |  |  
| - | - | - | - |  
| **Hotspot** |  **Location** |  **Why it is a bottleneck** |  **Consequence** |  
| Per-tile full-footprint scan |  `rasterize_instance_masks` (`damage_triage_infer.py:315-330`) |  Every 1024 tile runs `footprints.geometry.intersects(tile_box)` over **all** buildings; there is no spatial index on the in-memory GeoDataFrame. |  O(tiles × buildings) predicate evaluations; a 100k-building layer over 1,000 tiles means 100M geometry tests before any inference. |  
| Whole-tile forward + unbounded mask batch |  `score_window` |  One backbone pass per tile with up to N masks resident on the GPU; AOI mask mode scores one building at a time. |  Memory spikes on dense tiles, low GPU utilisation on sparse AOIs. |  
| Serial local execution |  `LocalRunner.add_task` (`container.wait()`) |  The worker blocks for the entire task; concurrency defaults to 1. |  One long assessment batch stalls every other queue item handled by that worker. |  
| Checkpoint re-download/re-load per task |  `_download_hf_checkpoint`, region CLI |  Each batch/container re-resolves or re-downloads weights and rebuilds DINOv3. |  Minutes of cold start per batch; egress amplification; GPU idle. |  
| Poll-by-re-enqueue |  queue handlers |  Progress is driven by re-enqueuing the entity; each hop pays visibility timeout and storage latency. |  Coarse status latency and duplicate work under at-least-once delivery. |  
| UI polling |  `Project.jsx` (20 s), `AoiAssessmentDashboard` (3 s) |  Full-detail refetch on a timer with no abort/backoff. |  Load scales with open tabs, not with work; stale responses race. |  
| Embedded scope payloads |  `dtfAoi` config, region batch config |  The full building list/geometry is serialised into the task message/config. |  Large queue messages, memory blow-ups, slow JSON parsing. |  
| Aggregation in memory |  `region_aggregate.aggregate` |  Loads all evidence rows for an assessment to pick winners. |  Memory grows linearly with AOI size; no streaming/chunked aggregation. |  
| Map read caps |  `layer_buildings_feature_collection` (20k), assessment paging |  Hard caps avoid OOM but silently truncate. |  Analysts can mistake a truncated view for complete coverage. |  
  
   
Severity: **P0** = wrong results or an unusable primary path; **P1** = production reliability/security defect; **P2** = maintainability/scale risk; **P3** = polish.  
**P0 — correctness and primary-path failures**  
  
|  |  |  |  |  
| - | - | - | - |  
| **#** |  **Finding** |  **Evidence** |  **Impact** |  
| P0-1 |  **The shipped checkpoint cannot run the whole-layer pipeline.** `build_dtf_model` is called without `allow_crop_mode`; a checkpoint containing `gate.weight` raises `RuntimeError` (line 260). The UI confirms the incompatibility in source comments: “the old whole-layer tile workflow (CreateEditDTFModal) can't load the crop-mode checkpoint this deployment ships”. |  `workflows/damage_triage_infer.py:733-735,260`; `LayerRow.jsx:54-55`; `ImageLayerInfoMobile.jsx:29-30` |  Whole-layer DTF is unavailable; operators may believe it ran because the queue/API still exists. |  
| P0-2 |  **i-dtf loads the same checkpoint with the wrong architecture.** Its loader reads flat keys from the repository-root model card `models/damage_triage/config.json`, which has nested `architecture` and no `use_gated_head`, constructs a non-gated `Model`, uses `strict=False` and logs only counts, then raises `NotImplementedError`. Crop-mode keys (`vit/gate/leaf/aux`) do not match `backbone/seg_head/damage_head`. Because the load happens in the FastAPI lifespan, the service dies during startup and never serves `/health`. |  `i-dtf/model_loader.py`; `i-dtf/api.py:72-85`; `models/damage_triage/config.json` |  The standalone service cannot start with the shipped weights. |  
| P0-3 |  **Three incompatible configuration schemas** describe the same model: the repository-root model card (`models/damage_triage/config.json`: `input_shape`, `mask_pooling`, nested architecture), the training dump (`use_gated_head`, `sfp_stride`, `no_multiscale`), and i-dtf's flat interpretation. `strict=False` hides drift in two of three loaders. |  §13.1 `checkpoints/config.json`; `models/damage_triage/config.json`; HASTE `build_dtf_model`; i-dtf `build_model` |  Silent random initialisation of whole heads; predictions become noise rather than an error. |  
| P0-4 |  **Normalisation is inconsistent and partly uncontrollable.** Compose advertises `HASTE_DTF_NORMALIZATION=imagenet`, but only the region workflow reads it. Whole-layer hard-codes `unit` (`damage_triage_infer.py:774`); AOI mask mode hard-codes `crop/255` (`damage_triage_aoi_infer.py:298`); AOI crop mode hard-codes ImageNet; i-dtf uses ImageNet for crops but unit for tiles. |  `damage_triage_region_infer.py:108`; `damage_triage_aoi_infer.py:298`; `damage_triage_infer.py:774`; i-dtf `inference.py` |  The env var is misleading, and the same weights produce different class distributions depending on entry point; whole-layer vs AOI comparisons are invalid. |  
| P0-5 |  **Class-3 label drift**: model card says "Partial Structural Damage", code and UI say "Partial Structural Collapse". |  model card vs `dtf_constants.py` |  Reports, training labels and exports disagree; any label-mapping by name breaks. |  
| P0-6 |  **Local-dev stack is unauthenticated and grants host root by design.** `DEVELOPMENT_MODE=true` → anonymous + admin, unsigned `x-ms-client-principal`, API bound to `7071:80` on all interfaces, Docker socket mounted in API **and** worker, blob container public, wildcard CORS, fixed dev credentials. |  `function_app.py:72-77,136-159`; `docker-compose.yml:155,204,213,233,334`; `nginx.conf`; `upload_data.py` |  Anyone who can reach port 7071 can drive the API and queue jobs that spawn task containers; compromise of the API process yields root-equivalent Docker control on the host. Acceptable only on a trusted laptop; bind compose to `127.0.0.1` and state the trust boundary in the docs. |  
  
   
**P1 — reliability, security and state integrity**  
  
|  |  |  |  |  
| - | - | - | - |  
| **#** |  **Finding** |  **Evidence** |  **Impact** |  
| P1-1 |  **Poll-by-re-enqueue with at-least-once delivery** means duplicate task execution; most processors are not idempotent (footprint reload, artifact upload, evidence upsert are, but layer status and counters are not). |  queue workers; `assessment_store.update_assessment_counts` |  Double counts, duplicate artifacts, lost updates. |  
| P1-2 |  **No compare-and-set on status transitions.** A late worker can overwrite `cancelled`/`failed` with `completed`; counters are read-modify-write. |  `assessment_store.py`, `processors/aoi_assessment.py` |  Assessments report success after cancellation or failure. |  
| P1-3 |  **Lease expiry is implemented three times** with different predicates; requeue can run a unit whose first attempt is still alive. |  `assessment_store.expire_leases`, batch/unit requeue paths |  Duplicate GPU work and evidence races. |  
| P1-4 |  **Runtime DDL and grants.** `ensure_run_results_table`, `_ensure_columns`, `ensure_schema`, `mark_imagery_updated` issue DDL from request/task paths behind process-global flags. |  `hastelib/core/utils/*` |  First request fails under a least-privilege DB role; multi-process workers each attempt the DDL. |  
| P1-5 |  **Migrations only run on a fresh volume**; no schema-version table and no migration runner. Existing deployments never receive `0003-0006` unless runtime DDL happens to add the columns. |  compose init mount |  Divergent schemas between fresh and upgraded installs. |  
| P1-6 |  **`LocalRunner` treats a missing status file as COMPLETED**, `cancel_task` only writes a status file (does not stop the container), hard-codes Azurite/DTF paths, and passes the whole environment into task containers. |  `core/runners/local.py` |  Failed tasks are reported as successes, cancellation leaves GPU work running, secrets leak into task env. |  
| P1-7 |  **Azure Batch path**: output upload uses root containers, 5xx retries without idempotency keys, aggressive pool resize, and `delete_all_resources` deletes more than the job's own resources. |  `core/runners/azure_batch.py` |  Shared pools/jobs can be destroyed; partial uploads leave corrupt outputs. |  
| P1-8 |  **Concurrency defaults to 1** and each task re-downloads the HF checkpoint / re-loads the model, so throughput scales with containers, not with GPUs. |  compose `HASTE_REGION_*`, workers |  Large AOIs are serialised; checkpoint download amplifies egress and cold start. |  
| P1-9 |  **Data-layer `update()` argument order differs** between implementations (local/blob/cosmos/datalake/postgres), so the same call updates different keys depending on backend. Artifact base URL is hard-coded to Azure Blob. |  `core/data_layer/*`, `artifact_storage/*` |  Local runs write wrong objects; portability claim is false. |  
| P1-10 |  **5,292-line `function_app.py`** mixes routing, auth, validation, orchestration and SQL; no shared request schema; duplicated AOI resolution. |  `api/hastefuncapi/function_app.py` |  High regression risk, hard to test, inconsistent errors. |  
| P1-11 |  **Assessment batch/unit planning has no per-imagery cap**, and the unit payload embeds the full building list in task config. |  `region_windows.group_batches`, `processors/aoi_assessment.py` |  Very large AOIs produce megabyte configs, memory spikes and slow queue messages. |  
| P1-12 |  **`_aoi_max_area_km2` code default conflicts with documentation.** |  `function_app.py` vs `docker/README.md`/`.env.example` |  Operators cannot predict rejection thresholds; validation rejects assessments the docs say are allowed. |  
| P1-13 |  **UI destroys error information** and polls without cancellation; maps leak; dead paths remain. |  §14.7 |  Operators cannot diagnose failures; long sessions degrade and duplicate requests pile up. |  
| P1-14 |  **Checkpoint loaded with `torch.load` without `weights_only=True`** from an env-controlled path/HF. |  `damage_triage_infer.py:253` |  Pickle deserialisation risk if the source is compromised. |  
| P1-15 |  **No upload/pixel/time caps on the HTTP API** beyond nginx's 20 MB; chunk count, raster size and AOI vertex count are unbounded. |  `function_app.py`, `nginx.conf` |  Memory/disk exhaustion from a single request. |  
  
   
**P2 — scaling, testability and operability**  
  
|  |  |  |  
| - | - | - |  
| **#** |  **Finding** |  **Impact** |  
| P2-1 |  No execution-level tests: `LocalRunner`, Azure Batch, data layers, artifact storage, HF download, real model forward, UI. CI cannot catch P0-1/P0-2. |    |  
| P2-2 |  Whole-tile 1024×1024 inference with `(N,1024,1024)` masks is memory-heavy; AOI/region crop mode avoids it, but the mask path is still available and unbounded. |    |  
| P2-3 |  Non-overlapping tiling splits boundary buildings; i-dtf additionally pads with `BORDER_REPLICATE`, which fabricates context at edges. |    |  
| P2-4 |  Queue retry/poison policy is per-queue env configuration with no dead-letter dashboard; failures are only visible by polling JSON status. |    |  
| P2-5 |  Logging is scraped from task files (`workflow_progress.log`) and stdout; no correlation id across API → queue → task → DB. |    |  
| P2-6 |  `haste_build.py` mutates the source tree's `__about__.py`, uploads to Azure, rewrites three `requirements.txt` files and `env.yml` as a side effect of `pip wheel`; concurrent builds race, and an offline build fails or silently skips. |    |  
| P2-7 |  UI Docker image runs the Vite **dev** server and ships `node_modules`; no production build/nginx is used by compose. |    |  
| P2-8 |  `docker/api/Dockerfile` exposes 7071 while compose and nginx use 8080; task images are not built by a plain `compose up` because they are profile-gated. |    |  
| P2-9 |  Public blob container means every crop/artifact URL is world-readable; a leaked URL is permanent access. |    |  
| P2-10 |  Runtime configuration is split across compose env, `.env`, `config_admin_settings.json` and hard-coded defaults, with no single schema or startup validation. |    |  
  
   
**P3 — polish and consistency**  
- Version/build metadata drift (`__about__.py` rewritten at build; wheel name regex assumes `hastegeo-`).  
- `docker/README.md` is stale (network/volume names, Node version), and the root `README.md` references a missing `assets/banner.png` and `LICENSE.txt`.  
- `QUICKSTART.md` describes the UI as SWA-CLI-hosted while compose runs the Vite dev server; the root `package.json` is a vestigial second Node project.  
- `data/` contains 2.7 GB of runtime chunks/logs inside the working tree; ensure it is excluded from images and commits.  
- `setup/install_python.ps1` downloads installers without checksums; `setup/start-task.sh` runs without `set -e`.  
- Dead code in the UI (`ImageLayer.jsx`, `CreateEditDTFModal`, unimplemented admin Save, missing tour key).  
- Unvalidated event types and direct state mutation in map components.  
- `output2visualizer.py` uses `assert` for CRS equality (disabled under `-O`).  
- `merge_with_building_footprints.py` computes `unknown_pct` from a class the legacy inference never emits.  
- External DTF README references a training script that is not in the tree.  
- Dataset/weights are CC-BY-NC-4.0 while the code is MIT; the DINOv3 backbone licence is undocumented in-repo.  
**Cross-directory summary**  
`haste-fork` and `i-dtf` are two independent shells over the same broken model contract:  
- **HASTE** is a full orchestration platform (API → queue → runner → container → PostGIS/blob) whose primary DTF path is unusable with the shipped crop-mode weights, while its AOI/region paths silently switch architecture and normalisation.  
- **i-dtf** is a single-process classifier that assumes a mask-pooling model and the same crop-mode weights cannot satisfy that assumption.  
Fixing P0-1…P0-4 at the **model contract** level (one authoritative config schema, explicit `crop_mode`/`mask_pooling` flag, one normalisation policy per checkpoint, and a startup self-test that runs one crop and one masked instance) would repair both directories at once. Everything else in the gap list is independent hardening.  
