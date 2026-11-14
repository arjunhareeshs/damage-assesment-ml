# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Helpers for the TiTiler XYZ tile URLs the UI draws imagery from."""

from __future__ import annotations

from typing import Optional


def mask_nodata(tile_url: Optional[str]) -> Optional[str]:
    """Make no-data pixels transparent in an imagery tile URL.

    The processed imagery COGs carry no nodata value, so TiTiler painted the
    empty parts of a mosaic -- the gap between imagery tiles that don't touch,
    and the collar around each tile -- as solid black boxes over the map.
    ``nodata=0`` masks pixels that are 0 in every band (GDAL dataset-mask
    semantics), so genuinely dark pixels still render.

    Only for imagery: class/prediction rasters, where 0 can be a real value,
    must not be passed through this. Empty strings, non-TiTiler URLs and URLs
    that already set ``nodata`` are returned unchanged.
    """
    if not tile_url or "/tiles/" not in tile_url or "nodata=" in tile_url:
        return tile_url
    separator = "&" if "?" in tile_url else "?"
    return f"{tile_url}{separator}nodata=0"
