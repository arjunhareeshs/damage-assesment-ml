# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Which bands are RGB, and which pixels hold image data.

Torch-free, so the API (map crops), imagery prep (footprint alignment) and
the model workflows read imagery the same way.
"""

from __future__ import annotations

import numpy as np


def rgb_band_indexes(raster) -> list:
    """Band indexes holding red, green, blue, in that order.

    Uses the file's colour interpretation when it names all three (so a
    blue-green-red-NIR file is read as RGB, not BGR); otherwise bands 1-3,
    the layout of NOAA, NAIP and most visual products.
    """
    from rasterio.enums import ColorInterp

    ci = list(raster.colorinterp)
    wanted = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(c in ci for c in wanted):
        return [ci.index(c) + 1 for c in wanted]
    return [1, 2, 3]


def valid_pixel_mask(raster, window, data: np.ndarray, out_shape=None) -> np.ndarray:
    """(h, w) bool: pixels holding image data in ``window``.

    Honours the file's nodata value, alpha band or internal mask
    (``dataset_mask``), so a transparent collar is never scored. A file that
    declares none has all-zero RGB treated as no-data (black collar), the
    rule ``utils/aoi.py`` uses for the valid-area polygon.
    """
    from rasterio.enums import MaskFlags, Resampling

    declared = any(
        MaskFlags.all_valid not in flags for flags in raster.mask_flag_enums[:3]
    )
    if declared:
        return raster.dataset_mask(
            window=window, out_shape=out_shape, resampling=Resampling.nearest
        ) > 0
    if raster.nodata is not None:
        return ~np.all(data == raster.nodata, axis=0)
    return ~np.all(data == 0, axis=0)
