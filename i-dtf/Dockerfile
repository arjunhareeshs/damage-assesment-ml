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
# Build (from this directory, with a modern Docker Buildx that supports
# --build-context / Compose "additional_contexts"):
#   docker compose build
#
# Fallback for older Docker CLIs without additional-context support (run
# from the REPO ROOT instead, since it needs both folders in one context):
#   docker build -f i-dtf/Dockerfile \
#     --build-context dtf_src=Haste-DTF/Damage-TriageFormer/src \
#     -t i-dtf:latest .

FROM python:3.11-slim

# opencv-python-headless needs libgl-adjacent runtime libs on some slim
# bases for a handful of codecs; keep this minimal and purge apt lists
# immediately to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch wheel first (own layer, own index) -- see requirements.txt
# for why this is split out.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/i-dtf/requirements.txt
RUN pip install --no-cache-dir -r /app/i-dtf/requirements.txt

# Damage-TriageFormer's model source, imported in place as the `src`
# package (unmodified, read-only). Brought in from a secondary build
# context so the primary context stays this small i-dtf/ folder instead of
# the whole (multi-GB) repository.
COPY --from=dtf_src . /app/Haste-DTF/Damage-TriageFormer/src

# i-dtf application code.
COPY . /app/i-dtf

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# CMD (not ENTRYPOINT) so `docker compose run i-dtf python -m i-dtf.predict ...`
# cleanly replaces this instead of prepending to it.
CMD ["python", "-m", "i-dtf.api"]
