# 2 — Docker: Images, Containers, and What Each One Does

This document explains the Docker layer: **how many images get built, how many
containers run, what each is responsible for, and which source files in
[`01-file-structure.md`](01-file-structure.md) end up inside which image**.

Everything here lives under `Haste-DTF/haste-fork/docker/`.

---

## 2.1 The short answer

| | Count | Total size |
|---|---|---|
| Images **built** from this repo | 8 | ~10.8 GB |
| Images **pulled** from registries | 2 | ~0.27 GB |
| **Total images** | **10** | **~11.1 GB** |
| Containers **running** continuously | 7 | |
| Containers that **run once and exit** | 3 | |
| Containers spawned **per job**, then removed | 1 per task | |

Plus one optional service (`training`) behind the `standalone` profile that is
off unless you explicitly ask for it.

---

## 2.2 Every image, what it is, and what it's for

### Built from this repo

| # | Image | Size | Dockerfile | Purpose |
|---|---|---|---|---|
| 1 | `haste-training` | **6.66 GB** | `docker/training/Dockerfile` | **The model container.** Holds the conda `bda` env (torch, torchgeo, transformers, timm), the vendored DTF code at `/opt/damage-triage-former`, and a pre-baked DINOv2 hub cache. Runs all DTF inference and embedding workflows. |
| 2 | `docker-hastefuncapi` | 1.12 GB | `api/hastefuncapi/Dockerfile` | Azure Functions **HTTP API** — the REST surface the UI calls. |
| 3 | `docker-hastefuncqueues` | 1.12 GB | `api/hastefuncqueues/Dockerfile` | Azure Functions **queue workers** — the orchestrator. Reads queue messages, queries PostGIS, spawns task containers. |
| 4 | `haste-imageryprep` | 0.96 GB | `docker/imageryprep/Dockerfile` | Imagery tiling / COG preparation. Spawned per task like the training image. |
| 5 | `haste-titiler` | 0.46 GB | `docker/titiler/Dockerfile` | Raster tile server — serves imagery to the map. |
| 6 | `docker-ui` | 0.35 GB | `ui/Dockerfile` | React + Vite dev server behind the SWA CLI. |
| 7 | `docker-azurite` | 0.11 GB | `docker/emulators/Dockerfile` | Azure Storage emulator — blob + queue + table. |
| 8 | `docker-data-init` | 0.06 GB | `docker/data-init/Dockerfile` | One-shot seeding of Azurite with default config JSON. |

### Pulled from registries

| # | Image | Size | Purpose |
|---|---|---|---|
| 9 | `postgis/postgis:16-3.4` | 0.21 GB | The spatial database. Holds `hg_buildings`; schema auto-applied from `db/migrations/` on first start. |
| 10 | `mcr.microsoft.com/oss/nginx/nginx:1.21.6` | 0.06 GB | `api-proxy` — routes `:7071` to the Functions apps. |

> **Why `haste-training` is so much larger than everything else:** it is the only
> image carrying a deep-learning stack. torch alone is ~1.7 GB, and the conda
> environment totals ~3.1 GB. Everything else is a web service.

---

## 2.3 Compose services and their runtime state

`docker compose config --services` reports 10. They fall into three groups.

### Group A — long-running services (7 containers, stay `Up`)

| Service | Port(s) | What it does |
|---|---|---|
| `azurite` | 10000 / 10001 / 10002 | Blob, queue, table emulator. All job messages flow through here. |
| `postgis` | 5432 | `hg_buildings` table; AOI queries run `ST_Intersects` against it. |
| `titiler` | 8000 | Serves map raster tiles. |
| `hastefuncapi` | (via proxy) | HTTP API. |
| `hastefuncqueues` | — | Queue triggers. **Mounts the Docker socket** so it can spawn task containers. |
| `api-proxy` | 7071 → 80 | nginx in front of the Functions apps. |
| `ui` | 4280 | The web app you open in a browser. |

### Group B — one-shot containers (run, exit `0`, stay as exited records)

| Service | Why it exits immediately |
|---|---|
| `data-init` | Uploads seed config into Azurite, then finishes. Exiting is success. |
| `training_image` | **Build-only.** Exists so compose builds and tags `haste-training`. It is never meant to stay running. |
| `imageryprep_image` | **Build-only.** Same, for `haste-imageryprep`. |

> Seeing `docker-training_image-1  Exited (0)` is **normal and correct**. It is
> not a crash. Its only job is to produce the image tag.

### Group C — optional

| Service | How to start |
|---|---|
| `training` | `docker compose --profile standalone up training` — direct fine-tuning runs, off by default. |

---

## 2.4 The per-job container (the part that surprises people)

When you trigger an assessment, **a new container is created on the fly**, does
the work, and is removed:

```
hastefuncqueues  ──(Docker socket)──►  docker run haste-training:latest \
                                          damage-triage-aoi-infer --config <task>.json
```

`LocalRunner` (`hastelib/src/hastegeo/core/runners/local.py`) creates it,
streams its logs, waits for the exit code, and reads the outputs back. This is
the local stand-in for Azure Batch, which does the same thing on a real pool.

A completed task leaves a directory like:

```
/shared/azurite/task_work/training-pool/dtfaoi-<uuid>/
├── inputs/       dtf_aoi_config_<modelId>.json
├── outputs/      *.gpkg, building_<id>.png, mask_<id>.png, manifest.json
├── logs/         damage_triage_aoi_friendly.log
├── output.log    full container stdout/stderr
├── status.json   { state, exit_code, start_time, end_time }
└── resolved_command.txt
```

These are kept only because `PRESERVE_LOCAL_TASK_DIRS=1`. They are your primary
debugging tool — **read `output.log` and `status.json` first when a job fails.**

---

## 2.5 Which source files land in which image

| Source | → Image(s) |
|---|---|
| `hastelib/src/hastegeo/` | `hastefuncapi`, `hastefuncqueues`, `haste-training`, `haste-imageryprep` |
| `external/damage-triage-former/` | `haste-training` only (at `/opt/damage-triage-former`) |
| `docker/training/code/` | `haste-training` (at `/app`) |
| `api/hastefuncapi/` | `hastefuncapi` |
| `api/hastefuncqueues/` | `hastefuncqueues` |
| `ui/` | `docker-ui` |
| `db/migrations/` | mounted into `postgis` |
| `setup/config_admin_settings.json` | `data-init` |

**Rebuild rule:** change `hastelib/` → rebuild **four** images. Change only the
UI → rebuild one. Restarting a container does *not* pick up source changes.

---

## 2.6 CPU and GPU builds

The training image builds in two variants, controlled by the `TORCH_VARIANT`
build argument.

| | CPU (default local) | GPU |
|---|---|---|
| `TORCH_VARIANT` | `cpu` | `cu124` |
| `HASTE_ENABLE_GPU` | `0` | `1` |
| Image size | 6.66 GB | ~9.9 GB (est.) |
| Command | `docker compose up --build` | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build` |

**These two flags must move together**, which is exactly why the GPU settings
live in a single overlay file rather than as two independent env vars:

* `TORCH_VARIANT=cu124` alone → you ship 3.2 GB of CUDA libraries that never
  touch a device.
* `HASTE_ENABLE_GPU=1` alone → every task container requests a GPU that torch
  cannot use, and on a machine with no NVIDIA device the container fails to
  start at all.

Before using the GPU overlay, confirm the host can actually do passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that fails, use the base file — it is correct for your machine.

> The published ACR image keeps `cu124` (the Dockerfile default), because Azure
> Batch nodes are GPU nodes. `build_and_push_images.sh` takes `-v cu124|cpu` and
> defaults to `cu124`, so the pushed image is unaffected by the local CPU build.

The code selects the device safely everywhere — every call site uses
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`. A CPU-only image
therefore **degrades to CPU rather than failing**. Results are identical; only
speed differs.

---

## 2.7 Disk footprint and cleanup

Docker consumes far more than the images themselves. On a working laptop:

| | Typical |
|---|---|
| Images | ~42 GB (incl. unrelated images) |
| **Build cache** | **can exceed 60 GB** |
| Volumes (Azurite blobs + PostGIS) | ~36 GB — **this is your data** |

Build cache is usually the biggest single consumer and is safe to delete:

```powershell
docker builder prune -af          # clears ALL build cache
docker image prune -f             # removes dangling images
```

> ⚠️ **Never run `docker system prune -a --volumes`.** The `--volumes` flag
> destroys the Azurite blob store and the PostGIS database — all local project
> state. There is no undo.

To keep cache from creeping back:

```powershell
docker builder prune -f --filter until=168h
```

---

## 2.8 Quick reference

```powershell
cd Haste-DTF\haste-fork\docker

docker compose up -d --build            # start everything (CPU)
docker compose ps                       # what is running
docker compose logs -f hastefuncqueues  # follow the orchestrator
docker compose up -d --force-recreate   # apply rebuilt images
docker compose down                     # stop (volumes/data preserved)
docker system df                        # disk usage
```

**`--force-recreate` matters.** Rebuilding an image does nothing for a container
that is already running — it keeps using the old image until recreated. If a
change "isn't taking effect", this is almost always why.

---

**Next:** [`03-setup-and-run.md`](03-setup-and-run.md)
