#!/usr/bin/env bash
# Build the HASTE Docker images. Builds only: it does not start any container
# or download model files. Outside Docker's image cache it only writes
# HASTE_HOST_CUDA_DIR into Haste-DTF/haste-fork/docker/.env (host-CUDA mode).
#
#   ./build.sh                 auto (default): GPU build when this machine has
#                              an NVIDIA GPU, Docker's nvidia runtime and CUDA 12
#                              libraries; otherwise the CPU build
#   ./build.sh --host-cuda     force the GPU build: only torch + torchvision
#                              are downloaded, CUDA is used from this machine
#   ./build.sh --cpu           force the CPU build (CPU torch, no CUDA at all)
#   ./build.sh --bundled-cuda  self-contained GPU images (CUDA 12.4 baked in,
#                              ~8 GB more to download; what Azure Batch needs)
#   ./build.sh --no-cache      rebuild every layer
#
# Only haste-training contains torch; no other image needs a GPU. In the GPU
# build the models still fall back to the CPU when no GPU can be attached.
#
# The GPU build needs a pip `site-packages/nvidia` folder with CUDA 12
# libraries (what `pip install torch` installs on the host). It is found from
# python3; override with HASTE_HOST_CUDA_DIR=/abs/path/to/site-packages/nvidia
# The path is saved to Haste-DTF/haste-fork/docker/.env for `docker compose up`.
#
# Images built, all in the Compose project `haste`
# (Haste-DTF/haste-fork/docker/docker-compose.yml):
#   haste-azurite, haste-data-init, haste-titiler, haste-api, haste-queues,
#   haste-ui, and the task images haste-training + haste-imageryprep
# postgis and nginx are pulled as ready-made images when you start the stack.
#
# Also writes DOCKER_GID (the Docker socket's group, as the current docker
# context's engine sees it) to docker/.env: the queue worker needs it to start
# task containers. CPU/bundled builds remove a stale HASTE_HOST_CUDA_DIR.
#
# Safe to re-run: unchanged layers come from the cache. A failed attempt (for
# example a dropped download) is retried, and finished layers are kept.
# Exits non-zero on failure. Log: ./logs/build_<time>.log
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
COMPOSE_DIR="$ROOT/Haste-DTF/haste-fork/docker"
RETRIES="${BUILD_RETRIES:-3}"

# Docker Desktop is a separate engine; the GPU works on the native one.
export DOCKER_CONTEXT=default

die() { echo "[build.sh] ERROR: $*" >&2; exit 1; }

mode_name=auto
extra=()
for arg in "$@"; do
    case "$arg" in
        --host-cuda) mode_name=host ;;
        --bundled-cuda|--gpu) mode_name=bundled ;;
        --cpu) mode_name=cpu ;;
        --no-cache) extra+=(--no-cache) ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $arg (see ./build.sh --help)" ;;
    esac
done

[ -f "$COMPOSE_DIR/docker-compose.yml" ] || die "not found: $COMPOSE_DIR/docker-compose.yml"
docker info >/dev/null 2>&1 || die "Docker engine not reachable (try: sudo systemctl start docker)"

# Locate a pip `nvidia` folder holding the CUDA 12 libraries torch loads.
find_host_cuda() {
    if [ -n "${HASTE_HOST_CUDA_DIR:-}" ]; then
        echo "$HASTE_HOST_CUDA_DIR"
        return
    fi
    python3 - <<'PY' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("nvidia.cudnn")
if spec and spec.submodule_search_locations:
    print(os.path.dirname(list(spec.submodule_search_locations)[0]))
PY
}

# Print why the host's CUDA libraries are unusable (nothing when they're fine).
host_cuda_libs_problem() {
    local dir="$1" missing=()
    [ -n "$dir" ] || { echo "no CUDA 12 libraries found (python3 cannot import nvidia.cudnn)"; return; }
    case "$dir" in /*) ;; *) echo "HASTE_HOST_CUDA_DIR must be an absolute path (got: '$dir')"; return ;; esac
    for lib in cudnn/lib/libcudnn.so.9 cublas/lib/libcublas.so.12 cublas/lib/libcublasLt.so.12 \
               cuda_runtime/lib/libcudart.so.12 cuda_nvrtc/lib/libnvrtc.so.12 \
               cufft/lib/libcufft.so.11 curand/lib/libcurand.so.10 cusolver/lib/libcusolver.so.11 \
               cusparse/lib/libcusparse.so.12 nccl/lib/libnccl.so.2 nvjitlink/lib/libnvJitLink.so.12; do
        [ -e "$dir/$lib" ] || missing+=("$lib")
    done
    [ "${#missing[@]}" -eq 0 ] || echo "CUDA folder $dir is missing: ${missing[*]}"
}

# Print why this machine can't run GPU containers (nothing when it can).
gpu_problem() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
        || { echo "nvidia-smi sees no NVIDIA GPU"; return; }
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
        || echo "Docker has no 'nvidia' runtime (sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker)"
}

# Write KEY=VALUE into docker/.env (compose reads it on every `up`), replacing
# an existing KEY line and leaving everything else untouched.
set_env_var() {
    local key="$1" value="$2" env_file="$COMPOSE_DIR/.env" tmp
    touch "$env_file"
    tmp="$(mktemp)"
    grep -v "^${key}=" "$env_file" > "$tmp" || true
    printf '%s=%s\n' "$key" "$value" >> "$tmp"
    cat "$tmp" > "$env_file"
    rm -f "$tmp"
}

# Remove KEY from docker/.env (no-op when absent).
unset_env_var() {
    local key="$1" env_file="$COMPOSE_DIR/.env" tmp
    [ -f "$env_file" ] || return 0
    tmp="$(mktemp)"
    grep -v "^${key}=" "$env_file" > "$tmp" || true
    cat "$tmp" > "$env_file"
    rm -f "$tmp"
}

files=(-f "$COMPOSE_DIR/docker-compose.yml")
cuda_dir=""
if [ "$mode_name" = auto ] || [ "$mode_name" = host ]; then
    cuda_dir="$(find_host_cuda)"
    libs_problem="$(host_cuda_libs_problem "$cuda_dir")"
    gpu_problem_msg="$(gpu_problem)"
    if [ "$mode_name" = auto ]; then
        if [ -z "$libs_problem" ] && [ -z "$gpu_problem_msg" ]; then
            mode_name=host
        else
            echo "[build.sh] No usable GPU: ${gpu_problem_msg:-$libs_problem}"
            echo "[build.sh] Building the CPU images (use --host-cuda to force the GPU build)."
            mode_name=cpu
        fi
    else
        # Forced GPU build: the libraries are required (torch can't load
        # without them); a missing GPU/runtime only means CPU at runtime.
        [ -z "$libs_problem" ] || die "$libs_problem
  Install CUDA 12 runtime wheels on the host (e.g. pip install --user torch), or
  point HASTE_HOST_CUDA_DIR at a site-packages/nvidia folder that has them,
  or build with --cpu / --bundled-cuda instead."
        [ -z "$gpu_problem_msg" ] \
            || echo "[build.sh] WARNING: $gpu_problem_msg; models will run on CPU until a GPU is available."
    fi
fi

case "$mode_name" in
    host)
        export HASTE_HOST_CUDA_DIR="$cuda_dir"
        set_env_var HASTE_HOST_CUDA_DIR "$cuda_dir"
        files+=(-f "$COMPOSE_DIR/docker-compose.hostcuda.yml")
        overlay=docker-compose.hostcuda.yml
        mode="GPU: torch + torchvision only, CUDA from this machine ($cuda_dir, saved to docker/.env)"
        ;;
    bundled)
        # CUDA ships inside the image: a host CUDA path left over from an
        # earlier --host-cuda build must not be mounted into tasks.
        unset_env_var HASTE_HOST_CUDA_DIR
        files+=(-f "$COMPOSE_DIR/docker-compose.gpu.yml")
        overlay=docker-compose.gpu.yml
        mode="GPU with CUDA 12.4 bundled in the training image"
        ;;
    cpu)
        unset_env_var HASTE_HOST_CUDA_DIR  # no host CUDA in a CPU build
        overlay=""
        mode="CPU only (CPU torch in haste-training)"
        ;;
esac

# The queue worker runs as a non-root user and reaches the Docker socket
# through this group; a wrong value means no task container can start.
# Ask the engine that will run HASTE (the current docker context) what group
# owns the socket it mounts into containers: Docker Desktop's VM (group 0)
# differs from a system engine's host socket (e.g. 145). Fall back to the
# host socket only if the engine cannot be asked.
sock_gid="$(docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    --entrypoint stat alpine -c %g /var/run/docker.sock 2>/dev/null || true)"
[ -n "$sock_gid" ] || sock_gid="$(stat -c %g /var/run/docker.sock 2>/dev/null || true)"
if [ -n "$sock_gid" ]; then
    set_env_var DOCKER_GID "$sock_gid"
    echo "[build.sh] DOCKER_GID=$sock_gid (Docker socket group) saved to docker/.env"
else
    echo "[build.sh] WARNING: /var/run/docker.sock not found; set DOCKER_GID in docker/.env yourself."
fi

mkdir -p "$ROOT/logs"
log_file="$ROOT/logs/build_$(date +%Y%m%d_%H%M%S).log"
echo "[build.sh] Mode: $mode"
echo "[build.sh] Log file: $log_file"

start=$(date +%s)
status=1
for attempt in $(seq 1 "$RETRIES"); do
    echo "[build.sh] Build attempt $attempt of $RETRIES ($(date +%T))"
    set +e
    (cd "$COMPOSE_DIR" && docker compose "${files[@]}" --profile tasks build --progress=plain "${extra[@]}") 2>&1 \
        | tee -a "$log_file"
    status="${PIPESTATUS[0]}"
    set -e
    [ "$status" -eq 0 ] && break
    echo "[build.sh] Attempt $attempt failed (exit $status)." | tee -a "$log_file"
    [ "$attempt" -lt "$RETRIES" ] && sleep 10
done

elapsed=$(( $(date +%s) - start ))
if [ "$status" -ne 0 ]; then
    echo "[build.sh] FAILED after ${elapsed}s. See $log_file" >&2
    exit "$status"
fi

echo "[build.sh] Build finished in ${elapsed}s."

# Ready-made images (postgis, nginx) now, so `up` downloads nothing.
echo "[build.sh] Pulling the ready-made images (postgis, nginx)..."
pulled=0
for attempt in $(seq 1 "$RETRIES"); do
    if (cd "$COMPOSE_DIR" && docker compose "${files[@]}" pull --ignore-buildable --quiet) 2>&1 | tee -a "$log_file"; then
        pulled=1; break
    fi
    echo "[build.sh] Pull attempt $attempt failed." | tee -a "$log_file"
    [ "$attempt" -lt "$RETRIES" ] && sleep 10
done
[ "$pulled" -eq 1 ] || echo "[build.sh] WARNING: could not pull postgis/nginx; \`up\` will try again."

# The queue worker bind-mounts this; if Docker had to create it, it would be
# root-owned inside your home and break Hugging Face downloads on the host.
mkdir -p "$HOME/.cache/huggingface/hub"

# Smoke tests: prove each image imports its code (no network, a few seconds).
# In GPU mode this is also the check that the image's torch loads this
# machine's CUDA libraries and sees the GPU.
smoke_failed=0
smoke() {
    local name="$1"; shift
    if out="$("$@" 2>&1)"; then
        echo "[build.sh]   ok   $name: $(printf '%s' "$out" | tail -1)"
    else
        echo "[build.sh]   FAIL $name:"; printf '%s\n' "$out" | tail -15 | sed 's/^/      /'
        smoke_failed=1
    fi
}
echo "[build.sh] Smoke-testing the images..."
smoke haste-imageryprep docker run --rm --network none -w /tmp -e WORKDIR=/tmp --entrypoint python haste-imageryprep -c \
    "import hastegeo.workflows.prepare_imagery, hastegeo.core.utils.footprint_sources; print('imports ok')"
smoke haste-api docker run --rm --network none -w /tmp -e WORKDIR=/tmp --entrypoint python haste-api -c \
    "import hastegeo.core.processors.uploader, hastegeo.core.processors.damage_triage; print('imports ok')"
smoke haste-queues docker run --rm --network none -w /tmp -e WORKDIR=/tmp --entrypoint python haste-queues -c \
    "import docker, hastegeo.core.runners.local; print('imports ok')"
train_py="import torch, torchvision, transformers, torchgeo, lightning, cv2, rasterio, geopandas
import hastegeo.workflows.damage_triage_infer, hastegeo.workflows.embed_buildings
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
if [ "$mode_name" = host ]; then
    smoke "haste-training (GPU via $cuda_dir)" docker run --rm --network none -w /tmp -e WORKDIR=/tmp --gpus all \
        -v "$cuda_dir:/opt/host-cuda/nvidia:ro" --entrypoint python haste-training -c \
        "$train_py
assert torch.cuda.is_available(), 'torch cannot use the GPU with the mounted CUDA libraries'
x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item()
torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 32, 32, device='cuda'))
print('GPU', torch.cuda.get_device_name(0), '| cuDNN', torch.backends.cudnn.version(), '| matmul + conv ok')"
elif [ "$mode_name" = bundled ]; then
    smoke haste-training docker run --rm --network none -w /tmp -e WORKDIR=/tmp --gpus all --entrypoint python haste-training -c "$train_py"
else
    smoke haste-training docker run --rm --network none -w /tmp -e WORKDIR=/tmp --entrypoint python haste-training -c "$train_py"
fi
if [ "$smoke_failed" -ne 0 ]; then
    echo "[build.sh] Smoke tests FAILED (details above). The images are built but not usable as-is." >&2
    [ "$mode_name" = host ] && echo "[build.sh] For a GPU failure, ./build.sh --cpu gives a working CPU stack." >&2
    exit 1
fi

echo "[build.sh] Images:"
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' \
    | grep -E '^REPOSITORY|^haste-' || true
echo "[build.sh] Start the stack with:"
echo "  cd $COMPOSE_DIR && docker compose -f docker-compose.yml${overlay:+ -f $overlay} up -d"
