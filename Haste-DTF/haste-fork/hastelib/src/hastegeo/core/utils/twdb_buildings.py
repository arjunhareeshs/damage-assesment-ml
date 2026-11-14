# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Load TWDB Cycle 2 (2025) flood-planning building footprints into ``hg_buildings``.

Input: the ``FPRxxBldgs_SVI_Pop_Feb2025.gdb.zip`` files fetched by
``data/download_twdb_buildings.py``. Each holds one layer of MultiPolygon-Z
footprints in EPSG:3857 with ``Bldg_ID``, use type, population and SVI.

Footprints are reprojected to EPSG:4326, flattened to 2D, and single-part
MultiPolygons become Polygons. ``Bldg_ID`` is kept as the building id (it is
NOT unique: 31,533 ids repeat, mostly across regions, with different
outlines), so each id is qualified with its planning region: ``FPR03-3968020``,
and a repeat inside one region gets ``-2``, ``-3``... in file order. The
original number is kept as ``bldg_id`` in ``hg_buildings.attributes``, which
also holds use type, population and SVI. Ids are stable between loads.

    python -m hastegeo.core.utils.twdb_buildings --layer-id texas-twdb-2025 \\
        --project-id twdb data/data/footprints/twdb_cycle2_2025
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Iterable, List, Optional

import geopandas as gpd  # type: ignore
import pyogrio  # type: ignore
import shapely  # type: ignore
from psycopg2.extras import execute_values  # type: ignore

from .postgis_connect import get_connection

logger = logging.getLogger(__name__)

SOURCE = "twdb_cycle2_2025"
ATTRIBUTE_FIELDS = {
    "Simp_type": "simp_type",
    "Est_floors": "est_floors",
    "Est_Day_Pop": "est_day_pop",
    "Est_Night_Pop": "est_night_pop",
    "TXF_SVI": "txf_svi",
    "SVI": "svi",
}
_INSERT = """
    INSERT INTO hg_buildings
        (building_id, image_layer_id, project_id, geometry, subtype,
         footprint_source, footprint_version, attributes)
    VALUES %s
    ON CONFLICT (image_layer_id, building_id) DO UPDATE SET
        geometry = EXCLUDED.geometry,
        subtype = EXCLUDED.subtype,
        footprint_source = EXCLUDED.footprint_source,
        footprint_version = EXCLUDED.footprint_version,
        attributes = EXCLUDED.attributes
"""
_TEMPLATE = (
    "(%s, %s, %s, ST_SetSRID(ST_GeomFromWKB(%s), 4326), %s, %s, %s, %s::jsonb)"
)


def _clean(value):
    """JSON-safe scalar: NaN/None -> None, numpy -> python."""
    if value is None or value != value:
        return None
    return value.item() if hasattr(value, "item") else value


def to_storage_geometry(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    """EPSG:3857/Z MultiPolygons -> 2D EPSG:4326 Polygon/MultiPolygon."""
    if geoms.crs is None:
        raise ValueError("Source has no CRS.")
    if geoms.crs.to_epsg() != 4326:
        geoms = geoms.to_crs(epsg=4326)
    flat = shapely.force_2d(geoms.values)
    out = [
        g.geoms[0] if g.geom_type == "MultiPolygon" and len(g.geoms) == 1 else g
        for g in flat
    ]
    return gpd.GeoSeries(out, index=geoms.index, crs=4326)


def unique_id(region: str, raw: str, seen: set) -> str:
    base = f"{region}-{raw}"
    candidate, n = base, 1
    while candidate in seen:
        n += 1
        candidate = f"{base}-{n}"
    seen.add(candidate)
    return candidate


def rows_from_frame(
    frame: gpd.GeoDataFrame,
    image_layer_id: str,
    project_id: str,
    region: str = "FPR00",
    seen: Optional[set] = None,
):
    """Build insert rows for one chunk; drops empty geometry.

    ``seen`` carries the ids already issued for this region across chunks.
    """
    seen = set() if seen is None else seen
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame["geometry"] = to_storage_geometry(frame.geometry)
    frame = frame[~frame.geometry.is_empty]
    raw_ids = frame["Bldg_ID"].astype(str).str.strip()
    ids = [unique_id(region, r, seen) for r in raw_ids]
    cols = [c for c in ATTRIBUTE_FIELDS if c in frame.columns]
    rows = []
    for bid, raw, geom, rec in zip(
        ids, raw_ids, frame.geometry.values, frame[cols].to_dict("records")
    ):
        attrs = {"bldg_id": raw, "region": region}
        attrs.update({ATTRIBUTE_FIELDS[c]: _clean(rec[c]) for c in cols})
        rows.append(
            (
                bid, image_layer_id, project_id, shapely.to_wkb(geom),
                attrs.get("simp_type"), SOURCE, "2025-02",
                json.dumps(attrs),
            )
        )
    return rows


def load_twdb_buildings(
    paths: Iterable[str],
    image_layer_id: str,
    project_id: str,
    chunk_size: int = 50000,
    replace: bool = True,
    conn=None,
) -> int:
    """Stream the given ``.gdb.zip`` files into ``hg_buildings``.

    Memory stays bounded by ``chunk_size``. With ``replace`` the layer's
    existing rows are deleted first (once, before all files), in the same
    transaction, so a failed load leaves the previous data intact.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    total = 0
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE hg_buildings ADD COLUMN IF NOT EXISTS attributes JSONB"
            )
            if replace:
                cursor.execute(
                    "DELETE FROM hg_buildings WHERE image_layer_id = %s",
                    (image_layer_id,),
                )
            for path in paths:
                layer = pyogrio.list_layers(path)[0][0]
                n = pyogrio.read_info(path, layer=layer)["features"]
                before = total
                region = layer.split("_")[0]
                seen: set = set()
                for skip in range(0, n, chunk_size):
                    frame = pyogrio.read_dataframe(
                        path, layer=layer, skip_features=skip, max_features=chunk_size
                    )
                    rows = rows_from_frame(
                        frame, image_layer_id, project_id, region, seen
                    )
                    if rows:
                        execute_values(
                            cursor, _INSERT, rows, template=_TEMPLATE, page_size=5000
                        )
                    total += len(rows)
                logger.info(
                    "%s: %d of %d features loaded (empty geometry skipped)",
                    os.path.basename(path), total - before, n,
                )
        connection.commit()
        return total
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_conn:
            connection.close()


def find_files(source: str) -> List[str]:
    if os.path.isdir(source):
        return sorted(glob.glob(os.path.join(source, "*.gdb.zip")))
    return [source]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="a .gdb.zip file or the folder holding them")
    ap.add_argument("--layer-id", required=True, help="image_layer_id to load into")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--append", action="store_true", help="do not clear the layer first")
    ap.add_argument("--chunk-size", type=int, default=50000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = find_files(a.source)
    if not files:
        raise SystemExit(f"No .gdb.zip files found in {a.source}")
    n = load_twdb_buildings(
        files, a.layer_id, a.project_id, a.chunk_size, replace=not a.append
    )
    print(f"Loaded {n:,} buildings into layer {a.layer_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
