#!/usr/bin/env python3
"""Download the TWDB Cycle 2 (2025) flood-planning building footprints.

Source: https://s3.amazonaws.com/twdb-gis-data/index.html
        Staged/office-of-planning/flood-planning/flood-data/property/cycle2_buildings_2025/

15 zipped file geodatabases (FPR01..FPR15, ~1.26 GB). Standard library only.
Resumable: finished files are skipped (size checked against the bucket
listing) and partial files continue where they stopped.

    python data/download_twdb_buildings.py                 # all regions
    python data/download_twdb_buildings.py --regions 2 9   # some regions
"""

import argparse
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "https://twdb-gis-data.s3.amazonaws.com"
PREFIX = (
    "Staged/office-of-planning/flood-planning/flood-data/property/"
    "cycle2_buildings_2025/"
)
DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "footprints", "twdb_cycle2_2025"
)


def list_files():
    url = f"{HOST}/?list-type=2&prefix={PREFIX}&max-keys=1000"
    with urllib.request.urlopen(url, timeout=60) as r:
        xml = r.read().decode()
    files = re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", xml, re.S)
    return [(k, int(s)) for k, s in files if k.endswith(".gdb.zip")]


def fetch_once(key, size, out_dir):
    name = key.rsplit("/", 1)[-1]
    dest = os.path.join(out_dir, name)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == size:
        return name, "already complete"
    if have > size:
        os.remove(dest)
        have = 0
    req = urllib.request.Request(f"{HOST}/{key}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r, open(
        dest, "ab" if have else "wb"
    ) as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if os.path.getsize(dest) != size:
        raise IOError(f"{name}: got {os.path.getsize(dest)} of {size} bytes")
    return name, "downloaded" if not have else "resumed"


def fetch(key, size, out_dir, attempts=10):
    """Retry dropped connections, resuming from the bytes already on disk."""
    for n in range(1, attempts + 1):
        try:
            return fetch_once(key, size, out_dir)
        except (OSError, IOError) as e:
            if n == attempts:
                raise
            wait = min(60, 2 ** n)
            print(f"  retry {n}/{attempts - 1} {key.rsplit('/', 1)[-1]}: {e} (in {wait}s)", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_DIR)
    ap.add_argument("--regions", type=int, nargs="*", help="e.g. 2 9 (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = list_files()
    if a.regions:
        want = {f"FPR{n:02d}Bldgs" for n in a.regions}
        files = [f for f in files if f[0].rsplit("/", 1)[-1].split("_")[0] in want]
    if not files:
        sys.exit("No matching files found in the bucket.")
    print(f"{len(files)} files, {sum(s for _, s in files)/1e9:.2f} GB -> {a.out}")

    failed = 0
    with ThreadPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(fetch, k, s, a.out): k for k, s in files}
        for j in as_completed(jobs):
            try:
                print(" ", *j.result(), flush=True)
            except Exception as e:
                failed += 1
                print("  FAILED", jobs[j].rsplit("/", 1)[-1], e)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
