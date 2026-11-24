# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The single preprocessing contract for DamageTriageFormer inference.

Every scoring path (crop mode and mask mode) normalizes through
``PreprocessingSpec``. The spec comes from the checkpoint's own config
(``preprocessing`` block of the model card: mean, std, crop margin, input
size, minimum footprint area) and falls back to ImageNet statistics, which is
what the released checkpoint was trained with. There is no other policy:
``HASTE_DTF_NORMALIZATION`` is validated by ``resolve_normalization`` and
anything but ``imagenet`` is an error, not a silent override.

Torch-free so the Functions apps can validate configuration too.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

PREPROCESSING_VERSION = "1.0.0"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PreprocessingSpec:
    """How a uint8 RGB image becomes model input."""

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_size: int = 224
    crop_margin_px: int = 32
    min_footprint_px: int = 30
    color_space: str = "RGB"
    source: str = "default"  # "config" when read from the checkpoint config

    def __post_init__(self):
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must have 3 values (RGB).")
        if any(s <= 0 for s in self.std):
            raise ValueError(f"std must be positive, got {self.std}.")
        if self.input_size <= 0 or self.crop_margin_px < 0 or self.min_footprint_px < 0:
            raise ValueError("input_size must be > 0; margin and min area >= 0.")
        if self.color_space.upper() != "RGB":
            raise ValueError(f"Only RGB input is supported, got {self.color_space!r}.")

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """uint8 ``(3, H, W)`` or ``(H, W, 3)`` -> float32 of the same shape."""
        return normalize_tile(image, mode="imagenet", spec=self)

    @property
    def fingerprint(self) -> str:
        """Short stable hash of every value that changes the model input."""
        payload = json.dumps(
            [self.mean, self.std, self.input_size, self.crop_margin_px,
             self.min_footprint_px, self.color_space.upper()]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {
            "mean": list(self.mean), "std": list(self.std),
            "input_size": self.input_size, "crop_margin_px": self.crop_margin_px,
            "min_footprint_px": self.min_footprint_px,
            "color_space": self.color_space, "source": self.source,
            "fingerprint": self.fingerprint,
        }


DEFAULT_SPEC = PreprocessingSpec()


def load_preprocessing_spec(config_path: Optional[str]) -> PreprocessingSpec:
    """Read the ``preprocessing`` block of a checkpoint's model-card config.

    No file or no such block -> ``DEFAULT_SPEC`` (ImageNet, the released
    checkpoint's training statistics). A block that is present but malformed
    raises, so a bad config can never quietly change model input.
    """
    if not config_path or not os.path.isfile(config_path):
        return DEFAULT_SPEC
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    block = cfg.get("preprocessing") if isinstance(cfg, dict) else None
    if not isinstance(block, dict):
        return DEFAULT_SPEC
    try:
        resize = block.get("resize", [DEFAULT_SPEC.input_size] * 2)
        if len(resize) != 2 or resize[0] != resize[1]:
            raise ValueError(f"resize must be square, got {resize}.")
        return PreprocessingSpec(
            mean=tuple(float(v) for v in block.get("image_mean", DEFAULT_SPEC.mean)),
            std=tuple(float(v) for v in block.get("image_std", DEFAULT_SPEC.std)),
            input_size=int(resize[0]),
            crop_margin_px=int(block.get("crop_margin_pixels", DEFAULT_SPEC.crop_margin_px)),
            min_footprint_px=int(
                block.get("minimum_footprint_area_pixels", DEFAULT_SPEC.min_footprint_px)
            ),
            color_space=str(block.get("color_space", "RGB")),
            source="config",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid preprocessing block in {config_path}: {exc}") from exc


def resolve_normalization(requested: Optional[str]) -> str:
    """The only supported policy is ``imagenet``; reject anything else."""
    value = (requested or "imagenet").strip().lower()
    if value != "imagenet":
        raise ValueError(
            f"Unsupported DTF normalization {requested!r}: the checkpoint is "
            "trained with ImageNet statistics, so 'imagenet' is the only "
            "valid value."
        )
    return value


def normalize_tile(
    tile: np.ndarray, mode: str = "imagenet", spec: Optional[PreprocessingSpec] = None
) -> np.ndarray:
    """Normalize a uint8 RGB image tile according to the specified mode.

    Args:
        tile: RGB image tile as uint8 numpy array. Expected shape is either
            (3, H, W) as produced by rasterio windowed reads, or (H, W, 3).
        mode: Normalization strategy, either 'imagenet' or 'unit'.

    Returns:
        np.ndarray: Float32 normalized array with the same shape as input.

    Raises:
        ValueError: If mode is not 'imagenet' or 'unit', or if input is not 3-channel.
    """
    if mode not in ("imagenet", "unit"):
        raise ValueError(
            f"Unknown normalization mode: {mode!r}. Must be 'imagenet' or 'unit'."
        )

    img = tile.astype(np.float32) / 255.0

    if mode == "unit":
        return img

    # Mean/std normalization (ImageNet unless a spec says otherwise)
    mean_v = np.asarray((spec or DEFAULT_SPEC).mean, dtype=np.float32)
    std_v = np.asarray((spec or DEFAULT_SPEC).std, dtype=np.float32)
    if img.ndim == 3:
        if img.shape[0] == 3:
            # (3, H, W)
            mean = mean_v[:, None, None]
            std = std_v[:, None, None]
        elif img.shape[2] == 3:
            # (H, W, 3)
            mean = mean_v[None, None, :]
            std = std_v[None, None, :]
        else:
            raise ValueError(
                f"Expected 3 channels (either shape (3, H, W) or (H, W, 3)), got shape {tile.shape}"
            )
    else:
        raise ValueError(f"Expected 3D array (channels, H, W), got shape {tile.shape}")

    return (img - mean) / std
