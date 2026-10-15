# syntax=docker/dockerfile:1.7
# ---------- Stage 1: Build Conda Environment ----------
FROM mcr.microsoft.com/azureml/curated/minimal-py312-cuda12.4-inference AS conda-env

USER root

# Pin micromamba's root prefix to the base image's conda root rather than relying
# on auto-detection from the base image's conda activation. Without this,
# `micromamba create -f env.yml` (name-based, no -p) can't resolve a root prefix
# and aborts at startup ("tl::bad_expected_access ... Bad expected access").
# Rooting it at /opt/miniconda also places the `bda` env at
# /opt/miniconda/envs/bda, where the later `conda-pack -n bda` looks for it.
ENV MAMBA_ROOT_PREFIX=/opt/miniconda

# Install micromamba and conda-pack (micromamba avoids Python version conflicts).
# Install conda-pack with micromamba rather than the base image's `conda`: a
# plain `conda install conda-pack` also drags in a conda self-upgrade (26.5.0 ->
# 26.5.3) whose conda-forge build bricks the base `conda` on link
# (ModuleNotFoundError: No module named 'conda'). micromamba installs conda-pack
# into the same /opt/miniconda prefix without touching/upgrading conda.
RUN for i in 1 2 3 4 5; do \
		apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx curl && break || sleep 15; \
	done \
	&& rm -rf /var/lib/apt/lists/* \
	&& curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba \
	&& micromamba install -p /opt/miniconda -c conda-forge conda-pack --yes \
	&& micromamba clean -afy

# Copy only env.yml first (maximize build cache)
COPY docker/training/env/env.yml /tmp/env.yml

# Torch build variant. `cu124` (the default) is what Azure Batch GPU nodes need,
# so build_and_push_images.sh keeps the default and the ACR image is unchanged.
# Passing `cpu` rewrites the pinned wheels to the CPU builds, which drops the
# transitively-pulled `nvidia/*` CUDA runtime (2.7 GB) and `triton` (555 MB)
# from site-packages. docker-compose.yml passes `cpu` for the local stack, where
# no GPU is exposed to the containers anyway.
ARG TORCH_VARIANT=cu124
RUN sed -i "s/+cu124/+${TORCH_VARIANT}/g; s#/whl/cu124#/whl/${TORCH_VARIANT}#g" /tmp/env.yml

# Create environment using micromamba and pack it for reuse.
# Force-reinstall setuptools after pip to restore conda-managed files pip may clobber.
# No BuildKit cache mount here: `az acr build` (ACR Tasks quick build) runs the
# classic Docker builder, which rejects `RUN --mount=type=cache` outright
# ("the --mount option requires BuildKit"). A local `docker build`/`docker
# compose build` with BuildKit enabled would benefit from one, but ACR is the
# build path this image actually ships through, so plain RUN is what works
# everywhere.
RUN micromamba create -f /tmp/env.yml -y \
	&& micromamba install -n bda -c conda-forge setuptools --force-reinstall -y \
	&& micromamba clean -afy \
	&& conda-pack -n bda -o /tmp/bda.tar.gz

# conda-pack writes the archive 0600 root. Stage 2 reads it through a bind mount
# as the unprivileged `dockeruser`, and a bind mount preserves the source mode
# (unlike the `COPY --chown` this replaced), so without this it fails with
# "tar: /tmp/bda.tar.gz: Cannot open: Permission denied". Kept as its own layer
# so the expensive env-create layer above stays cache-valid.
RUN chmod 0644 /tmp/bda.tar.gz


# ---------- Stage 2: Final Runtime Image ----------
FROM mcr.microsoft.com/azureml/curated/minimal-py312-cuda12.4-inference

USER root

# Install runtime system dependencies and CA certificates (with retry for flaky network)
RUN for i in 1 2 3 4 5; do \
		apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx ca-certificates && break || sleep 15; \
	done \
	&& update-ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# CLI shim for the building-embedding workflow (mirrors imageryprep's
# prepare-imagery shim). `python` resolves to the conda env at runtime via PATH.
RUN printf '#!/bin/bash\nexec python -m hastegeo.workflows.embed_buildings "$@"\n' > /usr/local/bin/embed-buildings \
	&& chmod +x /usr/local/bin/embed-buildings

# CLI shim for the DamageTriageFormer inference workflow (Option B).
RUN printf '#!/bin/bash\nexec python -m hastegeo.workflows.damage_triage_infer "$@"\n' > /usr/local/bin/damage-triage-infer \
	&& chmod +x /usr/local/bin/damage-triage-infer

# CLI shim for the AOI-scoped DamageTriageFormer inference workflow
# (Option B, per-building-crop path — docs/OptionB_AOI_Integration_Plan.md).
RUN printf '#!/bin/bash\nexec python -m hastegeo.workflows.damage_triage_aoi_infer "$@"\n' > /usr/local/bin/damage-triage-aoi-infer \
	&& chmod +x /usr/local/bin/damage-triage-aoi-infer

# CLI shim for the batched Large-AOI DamageTriageFormer region inference workflow.
RUN printf '#!/bin/bash\nexec python -m hastegeo.workflows.damage_triage_region_infer "$@"\n' > /usr/local/bin/damage-triage-region-infer \
	&& chmod +x /usr/local/bin/damage-triage-region-infer

# Prepare conda env directory owned by the target user, then extract the packed env
# as that user — non-root tar uses the running user's UID/GID, so no chown -R needed.
#
# A bind mount would let the 4.1 GB tarball exist only for the duration of this
# RUN instead of landing in a permanent layer, but `az acr build` (ACR Tasks
# quick build) runs the classic Docker builder, which rejects `RUN
# --mount=type=bind` the same way it rejects `--mount=type=cache` above
# ("the --mount option requires BuildKit"). So this uses a plain COPY --from
# instead: it does cost one extra ~4 GB layer in the pushed image (the
# tarball's bytes stay in that layer even after the `rm` below deletes it from
# the final filesystem view), but it's what actually builds on the path this
# image ships through.
RUN mkdir -p /opt/conda/envs/bda && chown dockeruser:dockeruser /opt/conda/envs/bda
COPY --from=conda-env --chown=dockeruser:dockeruser /tmp/bda.tar.gz /tmp/bda.tar.gz
USER dockeruser
RUN tar -xzf /tmp/bda.tar.gz -C /opt/conda/envs/bda \
	&& /opt/conda/envs/bda/bin/conda-unpack \
	&& rm -f /tmp/bda.tar.gz

# Set working directory (created as dockeruser since USER is already set)
WORKDIR /app

# Copy application code and scripts with correct ownership at copy time
COPY --chown=dockeruser:dockeruser docker/training/code /app
COPY --chown=dockeruser:dockeruser docker/training/scripts/ /app/scripts/

# Copy haste source so training can import the package just like imagery prep
COPY --chown=dockeruser:dockeruser hastelib/src/hastegeo /app/hastegeo

# Vendor DamageTriageFormer's model code (Option B). Code only (MIT licensed,
# see external/damage-triage-former/LICENSE) — the ~500MB checkpoint is NOT
# baked into the image; it's downloaded per-task by the Azure Batch runner
# like imagery/footprints (see DTFInferencePostprocessor._create_dtf_config),
# so pushing/pulling this image doesn't carry model weights.
COPY --chown=dockeruser:dockeruser external/damage-triage-former /opt/damage-triage-former

# Convert line endings and ensure scripts are executable inside the container
USER root
RUN sed -i 's/\r$//' /app/scripts/*.sh && chmod +x /app/scripts/*.sh

# Activate the environment by default and expose shared data directories
ENV PATH="/opt/conda/envs/bda/bin:${PATH}"
ENV PROJ_LIB=/opt/conda/envs/bda/share/proj
ENV GDAL_DATA=/opt/conda/envs/bda/share/gdal
ENV PYTHONPATH=/app
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Refuse the GDAL drivers carrying the deferred CVEs (HDF4/HDF-EOS heap
# overflow CVE-2026-8087 et al.) at the GDAL level, so subprocess CLI tools
# (gdalwarp/gdal_translate) and the runtime never register them. In-process
# training step entrypoints additionally call hastegeo.core.utils.
# gdal_security.harden_gdal(). See docs/known-vulnerabilities.md Root Cause C.
ENV GDAL_SKIP="HDF4 HDF4Image HDF5 HDF5Image netCDF"


# Pre-bake DINOv2 source + weights into an image-level torch.hub cache so
# Azure Batch nodes don't need internet access (or the GitHub API) at job
# runtime. The workflow code calls ``torch.hub.set_dir(TORCH_HUB_CACHE_DIR)``
# before loading so this cache is used regardless of the per-task TORCH_HOME
# the entrypoint script sets. Must run after the SSL_CERT_FILE env block so
# the conda Python trusts GitHub's TLS cert for the source-repo fetch.
RUN mkdir -p /opt/torch-hub-cache/hub \
	&& chown -R dockeruser:dockeruser /opt/torch-hub-cache
USER dockeruser
RUN python -c "import torch; torch.hub.set_dir('/opt/torch-hub-cache/hub'); \
	torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True, skip_validation=True); \
	torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True, skip_validation=True)"

# Use training-specific entrypoint wrapper to mirror imagery prep behavior
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
