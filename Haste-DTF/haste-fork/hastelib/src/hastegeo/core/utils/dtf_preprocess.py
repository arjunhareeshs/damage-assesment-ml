# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tile normalization utilities for DamageTriageFormer inference.

Provides standardized preprocessing matching DamageTriageFormer training
(ImageNet mean/std) or legacy HASTE pipelines (unit [0, 1]).
"""

import numpy as np

PREPROCESSING_VERSION = "1.0.0"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_tile(tile: np.ndarray, mode: str = "imagenet") -> np.ndarray:
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

    # ImageNet normalization
    if img.ndim == 3:
        if img.shape[0] == 3:
            # (3, H, W)
            mean = IMAGENET_MEAN[:, None, None]
            std = IMAGENET_STD[:, None, None]
        elif img.shape[2] == 3:
            # (H, W, 3)
            mean = IMAGENET_MEAN[None, None, :]
            std = IMAGENET_STD[None, None, :]
        else:
            raise ValueError(
                f"Expected 3 channels (either shape (3, H, W) or (H, W, 3)), got shape {tile.shape}"
            )
    else:
        raise ValueError(f"Expected 3D array (channels, H, W), got shape {tile.shape}")

    return (img - mean) / std
