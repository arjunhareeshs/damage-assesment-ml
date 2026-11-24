# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Stable building ids for footprints that carry none (shapely only, so
every image can import it)."""

from __future__ import annotations

import hashlib

import shapely


def canonical_building_id(geom, source: str) -> str:
    """sha1 of the source name and the geometry rounded to 1e-7 degrees.

    The same building always gets the same id, whatever row it sits in or
    which AOI the file was clipped to.
    """
    try:
        geom = shapely.set_precision(geom, grid_size=1e-7)
    except Exception:  # pragma: no cover - invalid geometry
        pass
    h = hashlib.sha1(f"{source}|".encode("utf-8"))
    h.update(shapely.to_wkb(geom))
    return h.hexdigest()
