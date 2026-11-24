# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared DamageTriageFormer model library.

Not a runnable workflow any more. The whole-layer tile pipeline
and the per-building AOI workflow were removed; this module keeps the pieces
the large-AOI workflow imports:

* ``build_dtf_model`` / ``GatedDinoCrop`` / checkpoint discovery
* ``score_window`` / ``score_crop`` / ``score_crops`` and the decoders
* ``read_tile``, ``crop_box_from_mask``, device / batch-size helpers

Used by ``damage_triage_region_infer.py``.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import torch
import torch.nn.functional as F
from hastegeo.core.utils.dtf_constants import (  # noqa: F401 (re-exported)
    DTF_CATEGORIES,
    DTF_IM_SIZE,
    UNCLASSIFIED,
    compute_model_digest,
)
from hastegeo.core.utils.dtf_preprocess import (
    DEFAULT_SPEC,
    PreprocessingSpec,
    load_preprocessing_spec,
    resolve_normalization,
)
from hastegeo.core.utils.logs import Logger as HasteLogger
from shapely.geometry import box

def _find_local_checkpoint():
    """Look for pre-downloaded best_model.pth and config.json in local directories."""
    env_ckpt = os.getenv("DTF_CHECKPOINT_PATH")
    env_cfg = os.getenv("DTF_CHECKPOINT_CONFIG_PATH")
    if env_ckpt and env_cfg and os.path.exists(env_ckpt) and os.path.exists(env_cfg):
        return env_ckpt, env_cfg

    candidate_dirs = []

    cwd = Path.cwd()
    candidate_dirs.append(cwd / "models" / "damage_triage")
    for parent in cwd.parents:
        candidate_dirs.append(parent / "models" / "damage_triage")

    file_dir = Path(__file__).resolve().parent
    for parent in file_dir.parents:
        candidate_dirs.append(parent / "models" / "damage_triage")
        candidate_dirs.append(parent / "Damage assesment" / "damage-assesment-ml" / "models" / "damage_triage")

    candidate_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    candidate_dirs.append(Path("/opt/dtf-checkpoint"))
    candidate_dirs.append(Path("/app/models/damage_triage"))

    for d in candidate_dirs:
        if not d.exists():
            continue
        ckpt = d / "best_model.pth"
        cfg = d / "config.json"
        if not cfg.exists():
            cfg = d / "dtf_config.json"
        if ckpt.exists() and cfg.exists():
            return str(ckpt), str(cfg)

    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.exists():
        found_ckpts = list(hf_hub.glob("**/best_model.pth"))
        found_cfgs = list(hf_hub.glob("**/config.json"))
        if found_ckpts and found_cfgs:
            return str(found_ckpts[0]), str(found_cfgs[0])

    return None, None

WORKDIR = os.getenv("WORKDIR", ".")
LOG_DIR = os.path.join(WORKDIR, "logs")
LOG_FILE = "damage_triage_verbose.log"
os.makedirs(LOG_DIR, exist_ok=True)

logger = HasteLogger.get_logger(
    "damage_triage_infer", log_dir=LOG_DIR, log_file=LOG_FILE
)
logger.info("Executing %s", __file__)

# DTF_CATEGORIES / DTF_IM_SIZE / UNCLASSIFIED (checkpoint class-index order,
# must match Damage-TriageFormer/src/config.py CATEGORIES exactly) live in
# hastegeo.core.utils.dtf_constants so torch-free code (the Functions apps)
# can import them; they are re-exported from this module unchanged.

# Where the training docker image bakes the cloned Damage-TriageFormer repo
# (its `src` package must be importable to reconstruct the model + load the
# checkpoint). Override for local dev runs outside the image.
DTF_REPO_ROOT = os.getenv("DTF_REPO_ROOT", "/opt/damage-triage-former")

# RGBA palette for the map visualizer raster, indexed by (dtf_class + 1) —
# index 0 is nodata/unclassified (transparent). Same 5-color damage ramp
# (white -> dark red) as the embedding sub-model's output2visualizer.py
# IDX_TO_COLOR, so both sub-models render with a visually consistent scale
# on the Visualizer map.
DTF_IDX_TO_COLOR = np.array(
    [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [252, 190, 165, 255],
        [251, 112, 80, 255],
        [211, 32, 32, 255],
        [103, 0, 13, 255],
    ],
    dtype=np.uint8,
)


def resolve_device() -> torch.device:
    """Inference device from ``DTF_DEVICE`` (default ``auto``).

    ``auto`` keeps the historical cuda-if-available behaviour. An explicit
    ``cuda``/``cuda:N`` refuses to fall back to CPU, so a container started
    without GPU access fails immediately instead of running for hours on CPU.
    """
    requested = os.getenv("DTF_DEVICE", "auto")
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"DTF_DEVICE={requested} but torch sees no CUDA device "
                f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})."
            )
    if device.type == "cuda":
        logger.info(
            "Using %s: %s (torch %s, CUDA %s)",
            device,
            torch.cuda.get_device_name(device),
            torch.__version__,
            torch.version.cuda,
        )
    return device


class GatedDinoCrop(torch.nn.Module):
    """Crop-mode DamageTriageFormer: the architecture of the current released
    checkpoint (upstream ``scripts/train_crop.py``). DINOv3 CLS token -> linear
    gate + 4-way leaf. Takes a 224x224 building crop, no instance mask."""

    crop_mode = True
    use_gated_head = True

    def __init__(self):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if DTF_REPO_ROOT not in sys.path:
            sys.path.insert(0, DTF_REPO_ROOT)
        from src.config import BACKBONE  # noqa: E402 (see sys.path above)

        # Architecture only: every weight comes from the checkpoint.
        self.vit = AutoModel.from_config(AutoConfig.from_pretrained(BACKBONE))
        hidden = self.vit.config.hidden_size
        self.gate = torch.nn.Linear(hidden, 1)
        self.leaf = torch.nn.Linear(hidden, len(DTF_CATEGORIES) - 1)
        self.aux = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        cls = self.vit(pixel_values=x).last_hidden_state[:, 0]
        p_damaged = torch.sigmoid(self.gate(cls))
        return torch.cat(
            [1 - p_damaged, p_damaged * torch.softmax(self.leaf(cls), dim=-1)],
            dim=1,
        )


def score_crop(model, crop: np.ndarray, device: torch.device) -> np.ndarray:
    """(1, 5) class probabilities for one ``(3, H, W)`` uint8 building crop."""
    import cv2

    spec = model_spec(model)
    resized = cv2.resize(
        np.transpose(crop, (1, 2, 0)),
        (spec.input_size, spec.input_size),
        interpolation=cv2.INTER_AREA,
    )
    x = np.transpose(spec.normalize(resized), (2, 0, 1))
    with torch.inference_mode():
        return model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()


def model_spec(model) -> PreprocessingSpec:
    """The preprocessing contract a loaded model carries (ImageNet default)."""
    spec = getattr(model, "preprocessing", None)
    return spec if isinstance(spec, PreprocessingSpec) else DEFAULT_SPEC


def read_checkpoint_config(config_path: str) -> dict:
    """Normalise the checkpoint config into one schema.

    Two formats exist for the same model and this is the only place that
    understands them:

    * model card (``architecture`` block): crop-mode when
      ``architecture.mask_pooling`` is false.
    * training argparse dump (``use_gated_head`` ...): a mask-pooling run.

    Returns ``{"schema", "crop_mode", "class_names", "train"}``. ``crop_mode``
    is ``None`` when the file does not say. Raises on unreadable or
    unrecognised files rather than guessing.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: expected a JSON object.")
    arch = cfg.get("architecture")
    if isinstance(arch, dict):
        mask_pooling = arch.get("mask_pooling")
        crop_mode = None if mask_pooling is None else (not bool(mask_pooling))
        return {
            "schema": "model_card",
            "crop_mode": crop_mode,
            "class_names": list(arch.get("class_names") or []),
            "train": {},
        }
    if "use_gated_head" in cfg or "sfp_stride" in cfg or "no_multiscale" in cfg:
        return {"schema": "training_dump", "crop_mode": False, "class_names": [], "train": cfg}
    raise ValueError(
        f"{config_path}: neither a model card ('architecture') nor a training "
        "config ('use_gated_head'); refusing to guess the architecture."
    )


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """Tensors only: ``weights_only=True`` never unpickles arbitrary code."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot safely load checkpoint {checkpoint_path}: {exc}. Only "
            "plain tensor state_dicts are accepted."
        ) from exc
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"{checkpoint_path}: no state_dict found.")
    return state_dict


def build_dtf_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    strict_region_check: bool = False,
    allow_crop_mode: bool = False,
    expected_digest: Optional[str] = None,
    selftest: bool = False,
):
    """Load the DamageTriageFormer checkpoint and return ``(model, use_gated)``.

    The single loading path. It
      * loads tensors only (``weights_only=True``);
      * reads the config through ``read_checkpoint_config`` and checks it
        against the weights: a model card that says crop-mode with
        mask-pooling weights (or the reverse) is an error;
      * checks the file's sha256 against ``expected_digest`` and the model
        card's ``weights_sha256``;
      * loads a crop-mode model with ``strict=True`` and a mask-pooling model
        refusing any missing ``backbone.*``/``damage_head.*`` weight;
      * attaches the checkpoint's ``PreprocessingSpec`` as
        ``model.preprocessing`` so every scorer normalises identically;
      * with ``selftest=True`` pushes one blank crop through and verifies a
        finite 5-class probability row.

    Crop-mode models are returned only when ``allow_crop_mode`` is set (the
    tile/mask scorers cannot use them).
    """
    state_dict = _load_state_dict(checkpoint_path, device)
    is_crop_weights = "gate.weight" in state_dict
    info = read_checkpoint_config(config_path) if os.path.isfile(config_path) else None

    if info is not None and info["crop_mode"] is not None:
        if info["crop_mode"] != is_crop_weights:
            raise RuntimeError(
                f"Checkpoint/config mismatch: {config_path} ({info['schema']}) "
                f"describes a {'crop' if info['crop_mode'] else 'mask-pooling'} "
                f"model but {checkpoint_path} holds "
                f"{'crop' if is_crop_weights else 'mask-pooling'} weights."
            )

    if info is not None and info["class_names"]:
        names = info["class_names"]
        if len(names) != len(DTF_CATEGORIES):
            raise RuntimeError(
                f"{config_path} lists {len(names)} classes; the pipeline has "
                f"{len(DTF_CATEGORIES)}."
            )
        if names != DTF_CATEGORIES:
            logger.warning(
                "Checkpoint class names differ from DTF_CATEGORIES (index order "
                "is what counts): %s vs %s",
                names,
                DTF_CATEGORIES,
            )

    if expected_digest or (info is not None and info["schema"] == "model_card"):
        actual = compute_model_digest(checkpoint_path)
        if expected_digest and actual != expected_digest:
            raise RuntimeError(
                f"Checkpoint digest {actual} != expected {expected_digest}."
            )
        try:
            with open(config_path, encoding="utf-8") as f:
                card_sha = json.load(f).get("weights_sha256")
        except (OSError, ValueError):
            card_sha = None
        if card_sha and card_sha != actual:
            raise RuntimeError(
                f"{checkpoint_path} sha256 {actual} does not match the model "
                f"card's weights_sha256 {card_sha}: wrong or corrupt weights."
            )

    spec = load_preprocessing_spec(config_path)

    if is_crop_weights:
        if not allow_crop_mode:
            raise RuntimeError(
                f"{checkpoint_path} is a crop-mode checkpoint; this caller "
                "needs a mask-pooling one."
            )
        model = GatedDinoCrop()
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        model.preprocessing = spec
        use_gated = True
    else:
        if info is None:
            raise RuntimeError(
                f"{config_path} not found: a mask-pooling checkpoint needs its "
                "training config to rebuild the architecture."
            )
        if DTF_REPO_ROOT not in sys.path:
            sys.path.insert(0, DTF_REPO_ROOT)
        from src.model.net import Model as DTFModel  # noqa: E402 (see sys.path above)

        train_cfg = info["train"]
        if not train_cfg:
            raise RuntimeError(
                f"{config_path} is a model card; building a mask-pooling model "
                "needs the training config."
            )
        model = DTFModel(
            im_size=DTF_IM_SIZE,
            num_classes=len(DTF_CATEGORIES),
            use_hierarchical=bool(train_cfg.get("use_hierarchical", False)),
            use_gated_head=bool(train_cfg.get("use_gated_head", False)),
            use_aux_severity=bool(train_cfg.get("use_aux_severity", False)),
            use_sfp=bool(train_cfg.get("use_sfp", False)),
            sfp_stride=int(train_cfg.get("sfp_stride", 4)),
            use_multiscale=not train_cfg.get("no_multiscale", False),
            use_attention=not train_cfg.get("no_attention_pool", False),
            drop_path_rate=float(train_cfg.get("drop_path", 0.0)),
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                "Checkpoint load: %d missing, %d unexpected keys (architecture "
                "from %s)", len(missing), len(unexpected), config_path,
            )
        critical = [
            k for k in missing if k.startswith(("backbone.", "damage_head."))
        ]
        if critical:
            raise RuntimeError(
                f"Checkpoint failed validation: {len(critical)} essential "
                "backbone./damage_head. weights missing (e.g. "
                f"{critical[:3]}); predictions would be noise."
            )
        model.to(device)
        model.eval()
        model.preprocessing = spec
        use_gated = model.use_gated_head

    logger.info(
        "Loaded DTF %s model; preprocessing %s (%s)",
        "crop" if is_crop_weights else "mask-pooling",
        spec.fingerprint,
        spec.source,
    )
    if selftest:
        selftest_model(model, device)
    return model, use_gated


def selftest_model(model, device: torch.device) -> None:
    """One blank crop through a crop-mode model; fail loudly on a broken load.

    Checks the output is ``(1, 5)``, finite, non-negative and sums to 1. Only
    crop mode: a mask-pooling model needs a full 1024 px tile, too heavy for
    a startup probe.
    """
    if getattr(model, "crop_mode", False) is not True:
        return
    size = model_spec(model).input_size
    probs = score_crop(model, np.zeros((3, size, size), dtype=np.uint8), device)
    if probs.shape != (1, len(DTF_CATEGORIES)):
        raise RuntimeError(f"Model self-test: output shape {probs.shape}.")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise RuntimeError("Model self-test: non-finite or negative probabilities.")
    if abs(float(probs.sum()) - 1.0) > 1e-3:
        raise RuntimeError(f"Model self-test: probabilities sum to {probs.sum():.4f}.")


def decode_gated_predictions(damage_out) -> np.ndarray:
    """Turn a GatedDamageHead output for one image into a (N, 5) prob matrix.

    ``damage_out`` is ``(gate_list, leaf_list)`` or, when the checkpoint was
    trained with ``use_aux_severity`` (true for the released checkpoint),
    ``(gate_list, leaf_list, aux_list)`` — see
    ``Damage-TriageFormer/src/model/gated_head.py:GatedDamageHead.forward``.
    Each element is a per-image list; this decodes image index 0 (this
    workflow always runs one tile per forward call).

    The gate is a binary damaged/undamaged classifier (``sigmoid``); the
    leaf is a 4-way classifier over damage sub-types, conditional on
    ``damaged``. Decomposed into the same 5-class probability space as
    ``DTF_CATEGORIES`` (sums to 1 per row): ``P(0) = 1 - P(damaged)``,
    ``P(i+1) = P(damaged) * P(leaf=i)`` for leaf class i in 0..3.
    """
    gate_list, leaf_list = damage_out[0], damage_out[1]
    gate_logits = gate_list[0]  # (N,)
    leaf_logits = leaf_list[0]  # (N, 4)
    n = gate_logits.shape[0]
    if n == 0:
        return np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)

    p_damaged = torch.sigmoid(gate_logits).unsqueeze(-1)  # (N, 1)
    leaf_probs = F.softmax(leaf_logits, dim=-1)  # (N, 4)
    probs = torch.cat([1.0 - p_damaged, p_damaged * leaf_probs], dim=-1)  # (N, 5)
    return probs.cpu().numpy()


def decode_severity_predictions(damage_out) -> np.ndarray:
    """Turn a plain SeverityHead output for one image into a (N, 5) prob matrix.

    ``damage_out`` is ``[sev_logits_list]`` (see ``Model.forward``'s
    non-gated, non-hierarchical branch); ``sev_logits_list[0]`` is ``(N, 5)``.
    Only used if a future checkpoint is trained without the gated head.
    """
    logits = damage_out[0][0]
    if logits.shape[0] == 0:
        return np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)
    return F.softmax(logits, dim=-1).cpu().numpy()


from hastegeo.core.utils.raster_bands import (  # noqa: E402  re-exported
    rgb_band_indexes,
    valid_pixel_mask,
)


def read_tile(raster, window: rasterio.windows.Window, tile_size: int) -> np.ndarray:
    """Read an RGB tile, zero-padding edge tiles up to (3, tile_size, tile_size)."""
    data = raster.read(rgb_band_indexes(raster), window=window)
    _, h, w = data.shape
    if h == tile_size and w == tile_size:
        return data
    padded = np.zeros((3, tile_size, tile_size), dtype=data.dtype)
    padded[:, :h, :w] = data
    return padded


def score_window(
    model: torch.nn.Module,
    use_gated: bool,
    tile: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    normalization: str = "imagenet",
    masks_per_forward: Optional[int] = None,
) -> np.ndarray:
    """Score building masks on a single tile window with DamageTriageFormer.

    Args:
        model: Evaluated PyTorch DTF model.
        use_gated: True if model uses gated head, False if severity head.
        tile: (3, H, W) numpy array, uint8 [0..255] or float32.
        masks: (N, H, W) numpy array of binary float masks.
        device: torch.device ('cuda' or 'cpu').
        normalization: must be 'imagenet' (kept for config compatibility).
        masks_per_forward: Optional maximum masks per forward pass.

    Returns:
        np.ndarray: Probabilities array of shape (N, 5).
    """
    if len(masks) == 0:
        return np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)

    resolve_normalization(normalization)  # only 'imagenet' is valid
    if tile.dtype != np.uint8:
        tile = np.clip(
            tile * 255.0 if tile.max() <= 1.0 else tile, 0, 255
        ).astype(np.uint8)
    tile_norm = model_spec(model).normalize(tile)

    tile_t = torch.from_numpy(tile_norm).unsqueeze(0).to(device)

    chunk_size = (
        masks_per_forward
        if (masks_per_forward and masks_per_forward > 0)
        else len(masks)
    )
    all_probs = []
    idx = 0

    while idx < len(masks):
        current_chunk_size = min(chunk_size, len(masks) - idx)
        chunk_masks = masks[idx : idx + current_chunk_size]
        masks_t = torch.from_numpy(chunk_masks).to(device)

        try:
            with torch.inference_mode():
                _, damage_out = model(tile_t, [masks_t])
                p = (
                    decode_gated_predictions(damage_out)
                    if use_gated
                    else decode_severity_predictions(damage_out)
                )
                all_probs.append(p)
                idx += current_chunk_size
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if current_chunk_size > 1:
                chunk_size = max(1, current_chunk_size // 2)
                logger.warning(
                    "CUDA OOM on tile; reducing mask chunk size to %d and retrying",
                    chunk_size,
                )
                continue
            else:
                logger.warning(
                    "CUDA OOM at mask chunk size 1; retrying on CPU"
                )
                orig_device = next(model.parameters()).device
                model.to("cpu")
                try:
                    with torch.inference_mode():
                        _, damage_out = model(tile_t.cpu(), [masks_t.cpu()])
                        p = (
                            decode_gated_predictions(damage_out)
                            if use_gated
                            else decode_severity_predictions(damage_out)
                        )
                        all_probs.append(p)
                        idx += current_chunk_size
                finally:
                    model.to(orig_device)

    return (
        np.vstack(all_probs)
        if all_probs
        else np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)
    )


# Crop-mode input contract defaults (the released checkpoint's model card,
# "preprocessing"): footprint bbox + 32 px margin clipped to the tile, resized
# to 224, ImageNet-normalised; footprints under 30 px are not scored. The
# loaded model's PreprocessingSpec overrides these when its config says so.
CROP_MARGIN_PX = 32
CROP_INPUT_SIZE = 224
MIN_CROP_FOOTPRINT_PX = 30
# GPU memory kept free for activations when sizing a forward pass, as a share
# of what is free when the batch size is chosen.
CROP_VRAM_BUDGET = 0.6
# Measured on an RTX 3060 (ViT-L, 224 px, fp32): 9.5 MiB per crop at batch 32-128,
# 17 MiB at batch 1. 12 MiB leaves a margin. Throughput plateaus at ~42 crops/s
# from batch 16 up, so a larger batch only costs memory.
CROP_BYTES_PER_ITEM = 12 * 1024 * 1024
CROP_MAX_PER_FORWARD = 64
CROP_CPU_PER_FORWARD = 4


def crop_box_from_mask(
    mask: np.ndarray, width: int, height: int, margin: int = CROP_MARGIN_PX
):
    """Pixel box ``(x0, y0, x1, y1)`` around a boolean footprint mask, grown by
    ``margin`` and clipped to the real ``width`` x ``height`` window (never into
    zero padding). ``None`` if the mask is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return (
        max(0, int(cols[0]) - margin),
        max(0, int(rows[0]) - margin),
        min(width, int(cols[-1]) + 1 + margin),
        min(height, int(rows[-1]) + 1 + margin),
    )


def resolve_crops_per_forward(
    device: torch.device,
    upper: Optional[int] = None,
    free_bytes: Optional[int] = None,
) -> int:
    """Crops per forward pass, chosen from the hardware that is actually there.

    * CPU: a small fixed batch (a large one only adds latency).
    * CUDA: as many as fit in ``CROP_VRAM_BUDGET`` of the memory free right
      now, rounded down to a power of two, between 1 and ``upper``
      (``HASTE_REGION_MASKS_PER_FORWARD``, default ``CROP_MAX_PER_FORWARD``).
    """
    cap = upper if upper and upper > 0 else CROP_MAX_PER_FORWARD
    if device.type != "cuda":
        return max(1, min(cap, CROP_CPU_PER_FORWARD))
    if free_bytes is None:
        free_bytes = torch.cuda.mem_get_info(device)[0]
    fit = int(free_bytes * CROP_VRAM_BUDGET // CROP_BYTES_PER_ITEM)
    fit = max(1, min(cap, fit))
    return 1 << (fit.bit_length() - 1)


def score_crops(
    model: torch.nn.Module,
    tile: np.ndarray,
    boxes: list,
    device: torch.device,
    crops_per_forward: Optional[int] = None,
) -> np.ndarray:
    """Score one crop per box from a ``(3, H, W)`` uint8 tile with a crop-mode
    model. Returns ``(len(boxes), 5)`` probabilities in ``boxes`` order.

    Crops are run ``crops_per_forward`` at a time. On CUDA out-of-memory the
    chunk is halved and retried, and a single crop that still does not fit is
    scored on the CPU, so a window is never lost to memory pressure.
    """
    import cv2

    if not boxes:
        return np.empty((0, len(DTF_CATEGORIES)), dtype=np.float32)
    if tile.dtype != np.uint8:
        tile = np.clip(
            tile * 255.0 if tile.max() <= 1.0 else tile, 0, 255
        ).astype(np.uint8)
    spec = model_spec(model)
    size = spec.input_size
    hwc = np.transpose(tile, (1, 2, 0))
    batch = np.empty((len(boxes), 3, size, size), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        crop = cv2.resize(
            np.ascontiguousarray(hwc[y0:y1, x0:x1]),
            (size, size),
            interpolation=cv2.INTER_AREA,
        )
        batch[i] = np.transpose(spec.normalize(crop), (2, 0, 1))

    chunk = crops_per_forward if crops_per_forward and crops_per_forward > 0 else len(batch)
    out = []
    idx = 0
    while idx < len(batch):
        n = min(chunk, len(batch) - idx)
        x = torch.from_numpy(batch[idx : idx + n])
        try:
            with torch.inference_mode():
                out.append(model(x.to(device)).cpu().numpy())
            idx += n
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if n > 1:
                chunk = max(1, n // 2)
                logger.warning(
                    "CUDA OOM on %d crops; reducing to %d per forward", n, chunk
                )
                continue
            logger.warning("CUDA OOM at 1 crop; scoring it on the CPU")
            orig_device = next(model.parameters()).device
            model.to("cpu")
            try:
                with torch.inference_mode():
                    out.append(model(x).numpy())
                idx += n
            finally:
                model.to(orig_device)
    return np.vstack(out)
