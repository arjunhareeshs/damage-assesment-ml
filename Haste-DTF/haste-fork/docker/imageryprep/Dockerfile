# Pinned to the same stable tag docker/api/Dockerfile uses. The previous
# `4-nightly-...` tag silently changes the base under every rebuild, so two
# developers could never produce the same image.
FROM mcr.microsoft.com/azure-functions/python:4.30.0-python3.11-slim

# Retry loop: security.debian.org's CDN occasionally 404s a package version
# transiently across edge nodes (seen with libglib2.0-0 on bullseye-security);
# retrying a fresh apt-get update usually lands on a consistent mirror.
RUN for i in 1 2 3 4 5; do \
        apt-get update && apt-get install -y \
            vim \
            curl \
            libgl1 \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
        && break || sleep 15; \
    done \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

USER root

# Create non-root user before COPY so --chown can reference it
RUN groupadd -r appgroup && useradd -r -g appgroup -m -s /bin/bash appuser

# Only the requirements file before the installs: copying the source code
# here would invalidate the pip layers below on every code change and
# re-download ~500 MB of packages. The code is copied after them.
COPY --chown=appuser:appgroup docker/imageryprep/requirements.txt /app/

# torch (CUDA 12.4 wheel by default, own layer, own index), matching training's
# TORCH_VARIANT default. Nothing this image runs (prepare_imagery,
# zip_artifacts) imports torch -- the model workflows run in haste-training --
# so the host-CUDA build passes TORCH_VARIANT=none and skips the ~3 GB of
# torch + nvidia/* wheels entirely. `cpu` installs the small CPU wheel.
ARG TORCH_VARIANT=cu124
RUN if [ "$TORCH_VARIANT" = "none" ] || [ "$TORCH_VARIANT" = "host" ]; then \
        echo "TORCH_VARIANT=$TORCH_VARIANT: not installing torch"; \
    else \
        pip install --no-cache-dir --retries 10 --timeout 120 torch \
            --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"; \
    fi

# Install dependencies (no need to install haste package since we copy source directly)
# --no-cache-dir: without it the downloaded wheels (~500 MB, incl. the GDAL
# and opencv manylinux wheels) are baked into this layer and shipped.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Source code last, so editing it rebuilds in seconds (ownership set at copy
# time, which avoids a chown -R afterwards).
COPY --chown=appuser:appgroup hastelib/src/hastegeo /app/hastegeo
COPY --chown=appuser:appgroup docker/imageryprep/scripts /app/scripts/

# Provide CLI entrypoints expected by the workflows without requiring a separate package install
RUN printf '#!/bin/bash\nexec python -m hastegeo.workflows.prepare_imagery "$@"\n' > /usr/local/bin/prepare-imagery \
    && printf '#!/bin/bash\nexec python -m hastegeo.workflows.zip_artifacts "$@"\n' > /usr/local/bin/zip-artifacts \
    && chmod +x /usr/local/bin/prepare-imagery \
    && chmod +x /usr/local/bin/zip-artifacts

# Set PYTHONPATH so hastegeo modules can be found at runtime
ENV PYTHONPATH=/app

# Refuse the GDAL drivers carrying the deferred CVEs (HDF4/HDF-EOS heap
# overflow CVE-2026-8087 et al.) at the GDAL level, so subprocess CLI tools
# (gdalwarp/gdal_translate) and the runtime never register them. In-process
# code additionally enforces a stricter allowlist via hastegeo.core.utils.
# gdal_security.harden_gdal(). See docs/known-vulnerabilities.md Root Cause C.
ENV GDAL_SKIP="HDF4 HDF4Image HDF5 HDF5Image netCDF"

# Ensure /app itself is writable by appuser (for runtime-created dirs like ./logs)
RUN chown appuser:appgroup /app

# Convert line endings and make entrypoint script executable
RUN sed -i 's/\r$//' /app/scripts/*.sh && chmod +x /app/scripts/entrypoint.sh

USER appuser

# Use the custom entrypoint script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]