# 1 — File Structure, Why It Looks Like This, and Component State

This document explains **what each part of the repository is, why it exists,
how the parts connect, and what state each one is in**. Read this first if you
are trying to understand the codebase. If you only want to get the system
running, skip to [`03-setup-and-run.md`](03-setup-and-run.md).

---

## 1.1 The one-paragraph version

This repository welds two separate projects together. **HASTE** (Microsoft, MIT)
supplies everything around the model — imagery ingestion, the GIS workspace, the
map UI, building footprints, job queueing, and reporting. **DamageTriageFormer
(DTF)** supplies the model itself — a DINOv3 ViT-L/16 backbone with a gated
damage head that classifies a building crop into 5 damage classes. HASTE
normally trains its own per-scene segmentation model; here that is replaced by
DTF's pretrained checkpoint. That substitution is referred to throughout the
project as **Option B**.

The consequence of this design — and the reason the tree looks the way it does —
is that **HASTE is vendored as a fork, DTF is vendored as source, and the glue
lives in a third place**: `hastelib/src/hastegeo/workflows/` and
`hastelib/src/hastegeo/core/processors/`.

---

## 1.2 Top-level layout

```
damage-assesment-ml/
├── README.md                      Project overview + architecture (start here)
├── requirements.txt               Deps for the LOCAL standalone scripts only
│                                    (NOT for the Docker stack — see §1.6)
├── download_models.py             Fetches both models into models/
├── run_texas_assessment.py        Standalone whole-image run (no Docker)
├── run_local_aoi_assessment.py    Standalone AOI run (no Docker, no PostGIS)
├── local_api_server.py            Minimal HTTP bridge: browser UI -> local pipeline
│
├── data/            [GITIGNORED — you will not receive this]
│   ├── imagery/                   Post-event GeoTIFF rasters
│   ├── footprint/                 Building footprint GeoJSON / GPKG
│   └── outputs/                   Results written by the standalone scripts
│
├── models/          [GITIGNORED — created by download_models.py]
│   ├── backbone/dinov3-vitl16/    DINOv3 ViT-L/16 backbone
│   └── damage_triage/             best_model.pth + config.json
│
├── logs/                          Local run logs
│
└── Haste-DTF/
    ├── haste-fork/                The HASTE platform (see §1.3)
    └── Damage-TriageFormer/       The DTF model source (see §1.4)
```

### Why `data/` and `models/` are gitignored

They are large and/or licensed separately. `data/footprint/` alone is ~5.3 GB
locally and `data/imagery/` ~829 MB; model weights are ~1.3 GB. `.gitignore`
excludes `data/`, `models/`, and the extensions `*.tif`, `*.tiff`, `*.gpkg`,
`*.geojson`.

> **This is the single most common source of confusion for a new user.**
> A fresh clone has **no imagery, no footprints, and no model weights**. The
> code will not run until you supply them. See
> [`03-setup-and-run.md` §3.4](03-setup-and-run.md) for where to get each.

---

## 1.3 `Haste-DTF/haste-fork/` — the platform

This is a fork of Microsoft's HASTE. Most of it is upstream code; the
project-specific work is concentrated in a handful of places.

```
haste-fork/
├── hastelib/src/hastegeo/         ← THE CORE LIBRARY. Shared by every container.
│   ├── workflows/                   Entry points executed INSIDE task containers
│   │   ├── damage_triage_aoi_infer.py   AOI-scoped DTF inference   [Option B]
│   │   ├── damage_triage_infer.py       Whole-layer DTF inference  [Option B]
│   │   ├── embed_buildings.py           Building embeddings (MOSAIKS/DINOv2)
│   │   ├── prepare_imagery.py           Imagery tiling / COG prep
│   │   └── zip_artifacts.py             Bundles outputs for download
│   │
│   └── core/
│       ├── processors/            Server-side orchestration (runs in the queue app)
│       │   ├── damage_triage_aoi.py     DTFAoiInferencePre/Postprocessor
│       │   ├── damage_triage.py         Whole-layer equivalent
│       │   ├── imagery.py, embedding.py, inference.py, train.py
│       │   └── artifacts.py, labels.py, metadata.py, stats.py, uploader.py
│       │
│       ├── runners/               HOW a job is executed
│       │   ├── local.py                 LocalRunner — spawns sibling containers
│       │   ├── azure_batch.py           AzureBatchRunner — production
│       │   ├── unified_runner.py        Picks local vs. batch
│       │   └── docker_utils.py          Docker SDK wrapper
│       │
│       ├── utils/                 postgis_query.py, postgis_load.py,
│       │                          crop_service.py, gdal_security.py, imagery.py
│       ├── models/                Data models
│       ├── data_layer/            Storage abstraction
│       └── artifact_storage/      Blob artifact handling
│
├── api/
│   ├── hastefuncapi/              Azure Functions HTTP API (the REST surface)
│   └── hastefuncqueues/           Azure Functions queue triggers (the job workers)
│
├── ui/                            React + Vite app (map, labeling, review)
├── db/migrations/                 PostGIS schema, incl. hg_buildings
├── docker/                        Compose file + all Dockerfiles  → see file 02
├── external/damage-triage-former/ DTF model code, VENDORED (code only, no weights)
├── setup/                         Seed config for the local stack
└── build_and_push_images.sh       Builds + pushes training/imageryprep to ACR
```

### The connection that matters most

`hastelib/src/hastegeo/` is **copied into three different images** —
`hastefuncapi`, `hastefuncqueues`, and `haste-training`. That is deliberate:
the same library is both the orchestrator (in the queue app) and the worker
(inside the task container). It explains a rule you must remember:

> Editing anything under `hastelib/` requires **rebuilding the images that
> embed it**, not just restarting containers.

### Why `external/damage-triage-former/` exists separately from `Damage-TriageFormer/`

There are two copies of the DTF model code on purpose:

| Path | Purpose |
|---|---|
| `Haste-DTF/Damage-TriageFormer/` | The upstream model repo — training scripts, the research code, used by the **standalone local scripts** |
| `Haste-DTF/haste-fork/external/damage-triage-former/` | A vendored **code-only** copy baked into the training image at `/opt/damage-triage-former` |

The vendored copy carries **no checkpoint**. The ~1.3 GB weights are fetched at
runtime from Hugging Face so that pushing/pulling the training image does not
drag the model with it.

---

## 1.4 `Haste-DTF/Damage-TriageFormer/` — the model

```
Damage-TriageFormer/
├── src/
│   ├── config.py         DEVICE selection, class names, thresholds
│   ├── model/            DINOv3 backbone + gated damage head (net.py)
│   ├── data/             Dataset / transform code
│   ├── training/         Training loops
│   └── utils/            visualization.py and helpers
├── checkpoints/config.json
├── main.py
└── requirements.txt      Model-only pins (torch 2.2.0) — see §1.6
```

**Output classes (index order is load-bearing — do not reorder):**

```
0  Undamaged
1  Partial Roof Damage
2  Total Roof Damage
3  Partial Structural Collapse
4  Total Structural Collapse
```

---

## 1.5 How the pieces connect at runtime

Two execution paths exist. They share the model and the crop logic, but nothing
else.

### Path A — the full HASTE stack (Docker Compose)

```
 Browser UI (:4280)
     │  draws an AOI / clicks buildings
     ▼
 hastefuncapi          HTTP API — validates, writes a queue message
     │
     ▼
 Azurite queue         local-dtf-aoi-queue
     │
     ▼
 hastefuncqueues       GetRunDTFAoiInferenceQueueTrigger
     │                 └─ DTFAoiInferencePostprocessor
     │                     ├─ postgis_query.buildings_in_aoi()   ST_Intersects
     │                     ├─ LocalRunner.run()  ─────────────┐
     │                     └─ update_building_predictions()   │
     │                                                        ▼
     │                                     ┌──────────────────────────────┐
     │                                     │ SIBLING CONTAINER            │
     │                                     │ image: haste-training        │
     │                                     │ cmd:   damage-triage-aoi-infer│
     │                                     │  • loads checkpoint ONCE     │
     │                                     │  • crops each building       │
     │                                     │  • 1 forward pass per bldg   │
     │                                     │  • writes GPKG + PNG + JSON  │
     │                                     └──────────────────────────────┘
     ▼
 PostGIS hg_buildings   review_status = "pending_review"
     ▼
 UI review panel        Accept / Reject → export excludes rejected
```

The key architectural point: **`hastefuncqueues` does not run the model.** It
spawns a *sibling container* from the `haste-training` image via the Docker
socket and waits for it. That is why the queue container mounts the Docker
socket, and why `haste-training` is a "build-only" service (see file 02).

### Path B — standalone local scripts (no Docker at all)

```
download_models.py  →  models/
                          │
      data/imagery/ ──────┼──────→ run_local_aoi_assessment.py   (AOI)
      data/footprint/ ────┘         run_texas_assessment.py      (whole image)
                                              │
                                              ▼
                                       data/outputs/
```

`local_api_server.py` is a thin HTTP bridge so the browser UI's "Classify
selected" button can drive Path B synchronously, without the Azure stack.

Use Path B for quick model experiments. Use Path A for the real product.

---

## 1.6 Three requirements files — which is which

This trips people up constantly. They are **not** interchangeable.

| File | Applies to | torch |
|---|---|---|
| `requirements.txt` (root) | Standalone local scripts (Path B) | `2.2.0` |
| `Haste-DTF/Damage-TriageFormer/requirements.txt` | The model repo standalone | `2.2.0` |
| `Haste-DTF/haste-fork/docker/training/env/env.yml` | **The training container** (Path A) | `2.5.1`, conda + pip |
| `Haste-DTF/haste-fork/docker/imageryprep/requirements.txt` | Imagery prep container | none |
| `Haste-DTF/haste-fork/api/hastefunc*/requirements.txt` | The two Functions apps | none |

The Docker stack reads **only** the `env.yml` / per-image requirements. Editing
the root `requirements.txt` has no effect on any container.

---

## 1.7 Component state

Honest status of each piece, as of 2026-08-12.

| Component | State | Notes |
|---|---|---|
| AOI-scoped DTF inference (`dtf-aoi`) | ✅ **Verified working** | Ran end-to-end on 2026-08-12; `exit_code 0`, 22.6 s, correct 5-class output with 0.9875 confidence |
| Docker Compose stack (10 services) | ✅ Working | All containers healthy; UI, API, queues, PostGIS, Azurite, TiTiler |
| Optimized images | ✅ Verified | Training image 13.93 → 6.66 GB; see file 02 |
| CPU / GPU dual setup | ⚠️ CPU verified, GPU **unbuilt** | CPU path fully tested. `cu124` produces a byte-identical `env.yml`, so it *should* build, but has not been built |
| Whole-layer DTF inference (`dtf`) | ⚠️ Present, unverified here | Code exists and is wired; not exercised in recent testing |
| Standalone scripts (Path B) | ⚠️ Present, unverified here | `download_models.py`, `run_*_assessment.py` not run in recent testing |
| Azure Batch runner | ⚠️ Not exercised locally | `LocalRunner` is what runs locally; `azure_batch.py` needs real Azure |
| `docs/` folder | ❌ **Missing** | `README.md` links to `docs/PROJECT_OVERVIEW.md`, `docs/OptionB_Quick_Start.md`, `docs/OptionB_AOI_Integration_Plan.md` — none exist in the tree. **These links are broken.** |
| HF checkpoint caching | ❌ **Known issue** | Every task re-downloads the ~1.3 GB checkpoint. Log shows it resolving `HOME` to `/home` rather than the shared cache dir |
| Friendly-log retrieval | ❌ Minor bug | `get_filecontent_from_task` reports `damage_triage_aoi_friendly.log not found` although the file exists in the task's `logs/`. Cosmetic — results unaffected |

---

**Next:** [`02-docker-images-and-containers.md`](02-docker-images-and-containers.md)
