# syntax=docker/dockerfile:1
#
# i-dtf: standalone image-in / image-out damage classifier.
#
# Isolation contract with the rest of the repo:
#   - This image only ever COPYs two things: this i-dtf/ folder, and the
#     read-only Damage-TriageFormer/src architecture code it needs to
#     reconstruct the model class. It never touches Haste-DTF/haste-fork
#     or the docker/ compose stack, and nothing outside i-dtf/ is written.
#   - The model checkpoint + DINOv3 backbone (~2.5GB) are NEVER copied into
#     the image. They are bind-mounted read-only from the host's existing
#     models/ folder at container run time (see docker-compose.yml), so
#     this image stays small and the weights are never duplicated on disk.
#   - Deleting this i-dtf/ folder removes 100% of what this file adds to
#     the repo -- the image itself is a build artifact, not a modification
#     of anything else in the tree.
#
# Build from the REPO ROOT (needs both i-dtf/ and Haste-DTF/.../src in one
# context) with any plain Docker CLI or ACR/CI builder -- no BuildKit
# named-build-context / Compose "additional_contexts" support required:
#   docker build -f i-dtf/Dockerfile -t i-dtf:latest .
#   az acr build --registry <name> --image i-dtf:latest -f i-dtf/Dockerfile .
#
# docker compose build (from this directory) works the same way -- see
# docker-compose.yml, which points its build context at the repo root.

# Damage-TriageFormer's model source, imported in place as the `src`
# package (unmodified, read-only). Pulled into its own stage so it can be
# COPY --from=... below instead of copying the whole repo into the final
# image.
FROM scratch AS dtf_src
COPY Haste-DTF/Damage-TriageFormer/src /src

FROM python:3.11-slim

# opencv-python-headless needs libgl-adjacent runtime libs on some slim
# bases for a handful of codecs; keep this minimal and purge apt lists
# immediately to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA 12.1 torch wheel first (own layer, own index) -- see requirements.txt
# for why this is split out. Targets the CUDA 12.1 driver on the Batch GPU
# nodes; bump the cuXXX suffix here if the pool's driver changes.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY i-dtf/requirements.txt /app/i-dtf/requirements.txt
RUN pip install --no-cache-dir -r /app/i-dtf/requirements.txt

COPY --from=dtf_src /src /app/Haste-DTF/Damage-TriageFormer/src

# i-dtf application code.
COPY i-dtf/ /app/i-dtf

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# CMD (not ENTRYPOINT) so `docker compose run i-dtf python -m i-dtf.predict ...`
# cleanly replaces this instead of prepending to it.
CMD ["python", "-m", "i-dtf.api"]
