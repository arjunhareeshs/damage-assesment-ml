# syntax=docker/dockerfile:1
#
# Per-building AOI damage assessment (run_local_aoi_assessment.py) on a GPU.
# No Azure, PostGIS or queue. Model weights and data are bind-mounted at run
# time (see docker-compose.yml), never copied into the image.
#
# Build with ./build.sh (or: docker compose build). Run with ./run.sh

FROM python:3.10-slim

# GDAL, GEOS and PROJ come bundled inside the rasterio, pyogrio, shapely and
# pyproj binary wheels, so no apt geo packages are needed. libglib is for
# OpenCV, libgomp for the CPU-side math libraries, libexpat for rasterio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Python packages come from ./localpkgs, a copy of this machine's own tested
# install (torch 2.8.0+cu126 and its CUDA libraries, transformers, geopandas,
# rasterio, OpenCV, ...), made by stage_local_packages.py. Nothing is
# downloaded. The CUDA runtime ships inside the torch packages, so the host
# only needs an NVIDIA driver supporting CUDA >= 12.6 (build.sh/run.sh check it).
COPY --from=pylibs --chmod=a+rX / /usr/local/lib/python3.10/site-packages/

# Fail the build here, not at run time, if a copied package is incomplete.
RUN python -c "import torch, torch.nn, transformers, einops, cv2, numpy, pandas, \
geopandas, rasterio, pyogrio, shapely, pyproj, pyarrow; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); \
from transformers import AutoModel, AutoConfig"

WORKDIR /app

COPY Haste-DTF/haste-fork/hastelib/src /app/Haste-DTF/haste-fork/hastelib/src
COPY Haste-DTF/Damage-TriageFormer/src /app/Haste-DTF/Damage-TriageFormer/src
COPY run_local_aoi_assessment.py /app/run_local_aoi_assessment.py

# WORKDIR here is the env var the hastegeo workflows write logs/ under.
ENV PYTHONUNBUFFERED=1 \
    WORKDIR=/app/run \
    HOME=/tmp \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DTF_DEVICE=cuda \
    DTF_BACKBONE_REPO=/app/models/backbone

RUN useradd --uid 1000 --no-create-home app \
    && mkdir -p /app/run/logs /app/outputs \
    && chown -R app:app /app/run /app/outputs
USER app

ENTRYPOINT ["python", "/app/run_local_aoi_assessment.py"]
