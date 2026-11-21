# Damage Assessment ML — HASTE ↔ DamageTriageFormer

Automated, map-based building damage assessment from post-disaster satellite
imagery. This repo integrates Microsoft's **HASTE** platform (ingestion, GIS
workspace, labeling UI, map/report) with **DamageTriageFormer**, a pretrained
5-class damage-typology model, so an analyst can go from raw imagery to a
reviewed, per-building damage map without training a model per disaster.

---

## 1. What this is

Post-disaster response teams need building-by-building damage assessment
fast. This project evaluated three candidate systems — DAHiTra, HASTE, and
DamageTriageFormer — and landed on a hybrid: **HASTE's map/UI/ingestion
shell, driven by DamageTriageFormer's inference** instead of HASTE's own
per-scene trained model. That combination is referred to throughout the docs
as **Option B**.

| | |
|---|---|
| **UI / map / ingestion / footprints / reports** | HASTE (`Haste-DTF/haste-fork`) — Azure Functions API, React UI, Azure Batch training/inference workers |
| **Damage model** | DamageTriageFormer (`Haste-DTF/haste-fork/external/damage-triage-former`) — DINOv3 ViT-L/16 backbone, gated damage head, 5-class typology, pretrained checkpoint on Hugging Face |
| **Output** | Undamaged / Partial Roof Damage / Total Roof Damage / Partial Structural Damage / Total Structural Collapse, per building, with confidence |

---

## 2. What's actually built here

Two parallel inference paths exist in `Haste-DTF/haste-fork`, both driven by
the same pretrained checkpoint:

### Whole-layer (`modelType="dtf"`) — fully automated
Classifies every building footprint in an image layer in one Azure Batch
job. Tiles the mosaic into 1024×1024 windows, batches all intersecting
building masks per tile, writes a full-layer predictions GeoPackage +
map raster. Triggered from the layer list (`CreateEditDTFModal.jsx`), no
analyst interaction required. Good for "classify everything, review later."

### AOI-scoped (`modelType="dtf-aoi"`) — interactive, reviewed
Lets an analyst draw an AOI (or click-select specific buildings) on the map,
scopes inference to just that area via a live PostGIS `ST_Intersects`
query, crops and scores **one building at a time**, and writes results back
to individual building rows — never touching the whole layer. Every new
prediction lands as `review_status="pending_review"`; the analyst clicks a
building on the map to see its crop, damage mask, class, and confidence, and
Accepts or Rejects it. Exports exclude rejected predictions by default.

```
Analyst draws AOI on the map (LabelingTool.jsx)
        │
        ▼
PUT PutRunDTFAoiInferenceQueueMessage  (aoi polygon | buildingIds)
        │
        ▼
DTFAoiInferencePreprocessor → queue (dtf_aoi_queue_name)
        │
        ▼
DTFAoiInferencePostprocessor
  ├─ postgis_query.buildings_in_aoi()      ST_Intersects against hg_buildings
  ├─ Azure Batch task: damage-triage-aoi-infer
  │     for each building: crop_service.crop_building() → 1 forward pass
  │     (checkpoint loaded once for the whole run, not per building)
  └─ postgis_query.update_building_predictions()   PATCH, never touches geometry
        │
        ▼
GetHgBuildingsGeoJSON → Visualizer.jsx vector layer (click a building)
        │
        ▼
DtfAoiReviewPanel.jsx → PutBuildingDTFReview (Accept / Reject)
        │
        ▼
GetAoiPredictionsGpkg (excludes rejected, on demand)
```

---

## 3. Repo structure

```
.
├── README.md                              this file
├── docs/haste-advancement/                large-AOI spec, implementation plan, repo audit
└── Haste-DTF/
    ├── haste-fork/                         HASTE platform (Microsoft, MIT) + this project's changes
    │   ├── hastelib/src/hastegeo/          core Python lib: models, processors, workflows, utils
    │   │   ├── core/processors/            damage_triage.py (whole-layer), damage_triage_aoi.py (AOI)
    │   │   ├── core/utils/                 postgis_load.py, postgis_query.py, crop_service.py
    │   │   └── workflows/                  damage_triage_infer.py, damage_triage_aoi_infer.py
    │   ├── api/hastefuncapi/               Azure Functions HTTP API
    │   ├── api/hastefuncqueues/            Azure Functions queue triggers (job workers)
    │   ├── ui/src/Components/              React UI (LabelingTool, Visualizer, BuildingValidation)
    │   ├── db/migrations/                  PostGIS schema (hg_buildings)
    │   ├── docker/                         docker-compose.yml + per-service Dockerfiles
    │   └── external/damage-triage-former/  DTF model: training + inference code
    │       └── src/model/                  DINOv3 backbone, gated damage head, net.py
```

---

## 4. Running it locally

```bash
cd Haste-DTF/haste-fork
docker compose -f docker/docker-compose.yml up
```

Brings up: Azurite (blob/queue emulator), a PostGIS database (`hg_buildings`
schema auto-applied from `db/migrations/` on first start), TiTiler, the two
Azure Functions apps, and the React UI at `localhost:4280`.

Local `hastelib/` source changes take effect on rebuild without publishing a
new package — the API/queue Dockerfiles install the pinned `hastegeo` wheel
first, then reinstall from local source over it.

**First run, in order:**
1. Create a project, upload post-event imagery + building footprints.
2. Either run the whole-layer button ("Run DamageTriageFormer") from the
   layer list, **or** open the Labeling Tool, draw an AOI, and run the
   AOI-scoped path.
3. For the AOI path: click a classified building on the map to review it.

---

## 5. Status

- Whole-layer (`dtf`) path: implemented, was already working prior to this
  project's changes.
- AOI-scoped (`dtf-aoi`) path: implemented end-to-end — PostGIS schema,
  backend processor/queue/API, per-building crop inference, AOI-draw and
  building-click-select UI, map review panel with Accept/Reject, filtered
  export.
- Backend unit tests (PostGIS query/load layer, AOI processor) pass against
  a real Python environment: `cd Haste-DTF/haste-fork/hastelib && PYTHONPATH=src pytest tests/`.
- Frontend changes pass ESLint clean of new issues.
- **Not yet verified**: the full live loop (`docker compose up` → draw AOI →
  run → click a building → Accept/Reject) has not been exercised in a
  running Docker environment.

---

## License

HASTE (`Haste-DTF/haste-fork`) is Microsoft-licensed, MIT. DamageTriageFormer
code (`Haste-DTF/haste-fork/external/damage-triage-former`) is MIT; its
pretrained weights and benchmark dataset are CC-BY-NC-4.0 (non-commercial use
only) — see `Haste-DTF/haste-fork/external/damage-triage-former/CITATION.cff`
for citations.
