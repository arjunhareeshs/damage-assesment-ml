# 3 — Setup and Run Guide

**Start here if you just cloned this repository.**

This guide takes you from a fresh clone to a working damage assessment. It
assumes no prior knowledge of the codebase.

Expect **60–90 minutes** end to end, most of it unattended downloading.

---

## 3.1 What you are setting up

A system that takes post-disaster satellite imagery plus building footprints and
returns a per-building damage classification:

```
Undamaged  ·  Partial Roof Damage  ·  Total Roof Damage
           ·  Partial Structural Collapse  ·  Total Structural Collapse
```

Two ways to run it. **Pick one.**

| | Path A — Docker stack | Path B — standalone scripts |
|---|---|---|
| What you get | Full product: map UI, AOI drawing, review workflow, database | Just the model, on files |
| Setup effort | Higher | Lower |
| Disk needed | ~25 GB | ~5 GB |
| Needs Docker | Yes | No |
| Use it for | Real work, demos, the actual product | Quick model experiments |

Path A is the real system. **This guide covers Path A in full**, with Path B in
§3.9.

---

## 3.2 Prerequisites

| Requirement | Notes |
|---|---|
| **Docker Desktop** | Running, with WSL2 backend on Windows |
| **Disk: ~25 GB free** | ~11 GB images + build cache + data |
| **RAM: 16 GB minimum** | 32 GB recommended; task containers default to a 32 GB limit |
| **Git** | To clone |
| **Hugging Face account** | Free — needed for a token to fetch model weights |
| GPU | **Optional.** The system runs on CPU by default and is fully functional without one |

Verify Docker:

```powershell
docker --version
docker compose version
docker info
```

---

## 3.3 Architecture at a glance

```
                        ┌──────────────────────────┐
    Browser  ──────────►│  ui           :4280      │
                        └────────────┬─────────────┘
                                     │ HTTP
                        ┌────────────▼─────────────┐
                        │  api-proxy    :7071      │  nginx
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  hastefuncapi            │  REST API
                        └────────────┬─────────────┘
                                     │ enqueue
                        ┌────────────▼─────────────┐
                        │  azurite   :10000-10002  │  blob + queue emulator
                        └────────────┬─────────────┘
                                     │ dequeue
                        ┌────────────▼─────────────┐
                        │  hastefuncqueues         │  orchestrator
                        └───┬──────────────────┬───┘
              ST_Intersects │                  │ docker run (per job)
                        ┌───▼──────────┐   ┌───▼──────────────────────┐
                        │ postgis :5432│   │ haste-training container │
                        │ hg_buildings │   │ DamageTriageFormer       │
                        └──────────────┘   │ crop → forward → GPKG    │
                                           └──────────────────────────┘
                        ┌──────────────────────────┐
                        │  titiler      :8000      │  map raster tiles
                        └──────────────────────────┘
```

The orchestrator does **not** run the model. It launches a separate container
per job. Details in [`02-docker-images-and-containers.md`](02-docker-images-and-containers.md).

---

## 3.4 Get the data and the models

> **A fresh clone contains no imagery, no footprints, and no model weights.**
> `.gitignore` excludes `data/`, `models/`, `*.tif`, `*.gpkg`, `*.geojson`.
> Nothing will run until you supply these. This is the step people skip.

### a) Model weights

Both models come from Hugging Face:

| Model | Hugging Face ID | Lands in |
|---|---|---|
| DamageTriageFormer checkpoint | `Ymx1025/DamageTriageFormer-model` | `models/damage_triage/` |
| DINOv3 ViT-L/16 backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `models/backbone/dinov3-vitl16/` |

Get a token at <https://huggingface.co/settings/tokens> (read scope is enough).
The DINOv3 backbone is a gated model — **you must accept its licence on the
model page while signed in**, or the download returns 403.

For Path A the training container downloads the checkpoint itself, using the
`HF_TOKEN` you put in `docker/.env` (next step). For Path B, run:

```powershell
python download_models.py     # idempotent; skips files already present
```

### b) Imagery

Post-event georeferenced rasters (GeoTIFF) → `data/imagery/`.

The imagery used in development is **NAIP aerial imagery** (e.g.
`m_3510133_se_14_060_20200915.tif`), available free from the USDA/USGS. Any
georeferenced GeoTIFF covering your area of interest works. For Path A you
upload imagery through the UI rather than placing it in `data/`.

### c) Building footprints

Building polygons (GeoJSON or GeoPackage) → `data/footprint/`.

Common free sources:
* **Microsoft Building Footprints** — <https://github.com/microsoft/GlobalMLBuildingFootprints>
* **OpenStreetMap** buildings, via Overpass or a regional extract

Footprints must cover the same area as your imagery and be in a CRS
`geopandas` can read. These files get large — the development set is ~5.3 GB.

---

## 3.5 Configure

```powershell
cd Haste-DTF\haste-fork\docker
copy .env.example .env
```

Edit `.env` and set your token:

```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
DOCKER_GID=0                  # 0 is correct on Docker Desktop
PRESERVE_LOCAL_TASK_DIRS=1    # keep task dirs — very useful for debugging
```

`docker/.env` is gitignored, so it is yours alone and never committed.

> **Do not set `HASTE_ENABLE_GPU=1` here.** GPU mode has two halves that must
> move together and this file can only set one. Use the overlay in §3.8.

---

## 3.6 Build and start

```powershell
cd Haste-DTF\haste-fork\docker
docker compose up -d --build
```

First build takes **30–60 minutes** — it resolves a large conda environment for
the training image. Later builds are much faster thanks to a pip cache mount.

Check everything came up:

```powershell
docker compose ps
```

Expected: 7 services `Up`, and `data-init`, `training_image`, `imageryprep_image`
showing `Exited (0)`.

> `Exited (0)` on those three is **correct, not a failure.** Two are build-only
> and one is a seeding job. See [file 02 §2.3](02-docker-images-and-containers.md).

Open <http://localhost:4280>.

| Service | URL |
|---|---|
| UI | http://localhost:4280 |
| API | http://localhost:7071 |
| TiTiler | http://localhost:8000 |
| PostGIS | `localhost:5432` |

---

## 3.7 Run your first assessment

1. **Create a project** in the UI.
2. **Upload imagery** — a post-event GeoTIFF. Wait for imagery prep to finish
   (a `haste-imageryprep` container runs and exits).
3. **Upload building footprints** — your GeoJSON/GPKG. These load into PostGIS
   as `hg_buildings`.
4. **Choose a path:**
   * *AOI-scoped (recommended, and the verified path)* — open the Labeling Tool,
     draw an AOI or click specific buildings, run DamageTriageFormer.
   * *Whole-layer* — from the layer list, "Run DamageTriageFormer", which
     classifies every footprint in the layer.
5. **Watch it run:**
   ```powershell
   docker compose logs -f hastefuncqueues
   ```
   You should see the trigger fire, a task ID appear, and a container spawn.
6. **Review** — click a classified building on the map to see its crop, damage
   mask, class and confidence, then Accept or Reject. New predictions arrive as
   `pending_review`; exports exclude rejected ones by default.

### What a successful run looks like

```
Running AOI-scoped DamageTriageFormer inference
1 buildings in scope for this AOI run
Loading DamageTriageFormer checkpoint on cpu...
Model architecture: gated head
Scored building 1/1
Classified 1/1 buildings
Writing 1 buildings -> ./outputs/dtf_aoi_predictions_2258.gpkg
AOI-scoped DamageTriageFormer inference completed successfully.
```

with `status.json` reporting `"state": "completed", "exit_code": 0`.

A verified reference run: **1 building, 22.6 s on CPU**, predicted `Undamaged`
at `0.9875` confidence. Model load is ~7 s of that and is paid once per run, not
per building.

---

## 3.8 Running on a GPU (optional)

Only if the host really has an NVIDIA GPU. Verify first:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that prints a GPU table:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The overlay flips `TORCH_VARIANT=cu124` and `HASTE_ENABLE_GPU=1` together so
they cannot drift apart. `--build` is required — the CPU and CUDA variants are
different images sharing one tag.

> Status: the CPU path is verified end to end. The `cu124` build is expected to
> work (its environment file is byte-identical to the original) but **has not
> been built and run** — treat the first GPU build as unproven.

---

## 3.9 Path B — standalone, no Docker

For model experiments without the platform.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python download_models.py

# whole image
python run_texas_assessment.py

# AOI-scoped
python run_local_aoi_assessment.py
```

Put imagery in `data/imagery/` and footprints in `data/footprint/`; results are
written to `data/outputs/`.

`local_api_server.py` bridges the browser UI's "Classify selected" button to
this local pipeline if you want the map without the Azure stack.

> Path B uses the **root** `requirements.txt` (torch 2.2.0). The Docker stack
> uses `docker/training/env/env.yml` (torch 2.5.1). They are independent —
> editing one has no effect on the other.

---

## 3.10 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Code change has no effect | Rebuilt images do nothing for running containers. `docker compose up -d --force-recreate`. Changing `hastelib/` requires rebuilding **four** images. |
| `training_image` shows `Exited (0)` | Normal — it is build-only. Not an error. |
| Job stuck at "Queued" | Check `docker compose logs -f hastefuncqueues`. Also refresh the UI — status can display stale. |
| Container fails: `could not select device driver ... [[gpu]]` | GPU requested on a machine without one. Ensure `HASTE_ENABLE_GPU=0` (the default) and do not use the GPU overlay. |
| 403 downloading the backbone | Accept the DINOv3 licence on its Hugging Face page while signed in, and check `HF_TOKEN` in `docker/.env`. |
| Disk filling up | Build cache, usually. `docker builder prune -af`. **Never** `docker system prune -a --volumes` — it deletes your data. |
| Every run re-downloads ~1.3 GB | Known issue — the checkpoint cache is not shared between task containers. Costs time and disk; does not affect results. |
| `damage_triage_aoi_friendly.log not found` warning | Known cosmetic bug. The file exists in the task's `logs/`. Results unaffected. |
| Job failed, need detail | Read `/shared/azurite/task_work/training-pool/<task-id>/output.log` and `status.json` inside the `hastefuncqueues` container. |

### Reading a failed job

```powershell
docker exec docker-hastefuncqueues-1 bash -lc "ls -t /shared/azurite/task_work/training-pool | head -5"
docker exec docker-hastefuncqueues-1 bash -lc "cat /shared/azurite/task_work/training-pool/<TASK_ID>/status.json"
docker exec docker-hastefuncqueues-1 bash -lc "tail -60 /shared/azurite/task_work/training-pool/<TASK_ID>/output.log"
```

---

## 3.11 Everyday commands

```powershell
cd Haste-DTF\haste-fork\docker

docker compose up -d                    # start
docker compose down                     # stop (data preserved)
docker compose ps                       # status
docker compose logs -f hastefuncqueues  # follow the orchestrator
docker compose up -d --force-recreate   # apply rebuilt images
docker system df                        # disk usage
```

---

## 3.12 Known gaps

Be aware before relying on this:

* The root `README.md` links to `docs/PROJECT_OVERVIEW.md`,
  `docs/OptionB_Quick_Start.md` and `docs/OptionB_AOI_Integration_Plan.md`.
  **That `docs/` folder is not present in the tree — those links are broken.**
* The whole-layer (`dtf`) path and the Path B standalone scripts exist and are
  wired, but were not exercised in the most recent verification pass. Only the
  AOI-scoped path is confirmed working end to end.
* Every user still **builds** the images locally. Building generates tens of GB
  of build cache regardless of image size. Publishing prebuilt images to a
  registry and switching compose from `build:` to `image:` would remove that
  cost for everyone pulling the repo — not yet done.

---

## 3.13 Where to go next

| To understand | Read |
|---|---|
| The codebase and how components connect | [`01-file-structure.md`](01-file-structure.md) |
| Images, containers, rebuild rules, disk | [`02-docker-images-and-containers.md`](02-docker-images-and-containers.md) |
| Project background and the Option B decision | `README.md` at the repo root |
