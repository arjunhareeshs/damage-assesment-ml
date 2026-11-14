#!/usr/bin/env python3
"""Copy the Python packages this pipeline needs out of the host's own install
into ./localpkgs, laid out as a site-packages directory.

The Docker image copies that directory in (see Dockerfile) instead of
downloading torch (~3 GB) over the network. Each package is copied from its
own install record (RECORD), so the files are exactly what pip installed, and
the image runs the same versions this repo was tested with.

    python3 stage_local_packages.py            # rebuild ./localpkgs
"""
import importlib.metadata as md
import re
import shutil
import sys
from pathlib import Path

from packaging.requirements import Requirement

ROOTS = [
    "torch", "transformers", "einops", "huggingface_hub",
    "geopandas", "rasterio", "pyogrio", "shapely", "pyproj",
    "numpy", "pandas", "opencv-python-headless",
]
# Provided by the base image; copying them over would clobber it.
SKIP = {"pip", "setuptools", "wheel"}
OUT = Path(__file__).resolve().parent / "localpkgs"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def closure() -> dict:
    found, todo = {}, [norm(r) for r in ROOTS]
    while todo:
        name = todo.pop()
        if name in found or name in SKIP:
            continue
        dist = md.distribution(name)  # raises if a root is not installed
        found[name] = dist
        for req in dist.requires or []:
            r = Requirement(req)
            # Skip optional extras; evaluate markers for this interpreter.
            if r.marker is not None and not r.marker.evaluate({"extra": ""}):
                continue
            todo.append(norm(r.name))
    return found


def main() -> int:
    dists = closure()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    n_files = 0
    for name, dist in sorted(dists.items()):
        base = Path(dist.locate_file(""))
        for f in dist.files or []:
            parts = f.parts
            # Scripts installed outside site-packages, caches, and RECORD's
            # own self-references are not part of the importable package.
            if parts[0] == ".." or "__pycache__" in parts or f.suffix == ".pyc":
                continue
            src = base / f
            if not src.is_file():
                continue
            dst = OUT / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_files += 1
        print(f"  {name:28}{dist.version}")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(dists)} packages, {n_files} files, {size / 1e9:.2f} GB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
