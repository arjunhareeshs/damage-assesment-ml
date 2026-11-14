#!/usr/bin/env python3
"""Convert TWDB Cycle 2 building ``.gdb.zip`` files to GeoParquet.

Reads each file geodatabase straight from the zip (no extraction) and writes
one ``.parquet`` per region, so the attributes can be opened in QGIS, DuckDB,
pandas/geopandas, etc.

Geometry is reprojected to EPSG:4326 and flattened to 2D by default. Pass
``--keep-crs`` to keep the source EPSG:3857 coordinates.

Requires: pip install pyogrio geopandas pyarrow

Usage:
    python convert_twdb_to_geoparquet.py
    python convert_twdb_to_geoparquet.py --regions 02 03 --out my_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pyogrio
from shapely import force_2d

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "data" / "footprints" / "twdb_cycle2_2025"


def convert(src: Path, dst: Path, keep_crs: bool) -> int:
    layer = pyogrio.list_layers(src)[0][0]
    frame = gpd.read_file(src, layer=layer, engine="pyogrio")
    if not keep_crs:
        frame = frame.to_crs(4326)
    frame["geometry"] = force_2d(frame.geometry.values)
    tmp = dst.with_suffix(".parquet.part")
    frame.to_parquet(tmp, compression="zstd", write_covering_bbox=True)
    tmp.replace(dst)
    return len(frame)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <src>/geoparquet)")
    ap.add_argument("--regions", nargs="*", help="e.g. 02 03 (default: all)")
    ap.add_argument("--keep-crs", action="store_true",
                    help="keep EPSG:3857 instead of reprojecting to 4326")
    args = ap.parse_args()

    out = args.out or args.src / "geoparquet"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src.glob("*.gdb.zip"))
    if args.regions:
        wanted = {f"FPR{r.zfill(2)}" for r in args.regions}
        files = [f for f in files if f.name[:5] in wanted]
    if not files:
        print("no .gdb.zip files found", file=sys.stderr)
        return 1

    total = 0
    for f in files:
        dst = out / (f.name.removesuffix(".gdb.zip") + ".parquet")
        n = convert(f, dst, args.keep_crs)
        total += n
        print(f"{f.name}: {n:,} buildings -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
    print(f"done: {total:,} buildings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
