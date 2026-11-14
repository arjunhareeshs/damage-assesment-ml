# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Read user-supplied building footprints from many source formats.

The imageryprep workflow used to accept exactly one ``.gpkg``. Real footprint
data arrives in other shapes: Esri file geodatabases (``.gdb`` folders or
``.gdb.zip`` — e.g. the TWDB Cycle 2 regional files), GeoJSON, GeoParquet, and
whole folders / zips holding several of those (one file per planning region).

:func:`read_footprints` takes any of those and returns a single
``GeoDataFrame`` that is

* EPSG:4326, 2D, Polygon / MultiPolygon only,
* limited to features that touch the AOI's bounding box (the filter is pushed
  down into the read, so a statewide file is never fully loaded), and
* normalised to the columns ``id`` (unique ``str``), ``subtype``, ``class``.

Supported inputs
----------------
===============  =========================================================
``.gpkg``        GeoPackage
``.geojson``     GeoJSON (``.json`` too). Missing CRS is taken as EPSG:4326,
                 per RFC 7946; every other format must carry its own CRS.
``.gdb``         Esri file geodatabase folder
``.gdb.zip``     zipped file geodatabase (read in place, never extracted)
``.parquet``     GeoParquet (``.geoparquet`` too)
``.zip``         a zip holding a ``.gdb`` folder, or ``.gpkg`` / ``.geojson``
folder           any of the above, searched recursively (depth <= 3)
===============  =========================================================

Several sources are merged. When the input holds more than one dataset, ids
are prefixed with the source name (whether or not each touches the AOI, so a
building keeps its id across AOIs), because ids such as TWDB ``Bldg_ID``
repeat across regions. Remaining duplicates get ``-2``, ``-3``... suffixes.

Debug a source from the shell::

    python -m hastegeo.core.utils.footprint_sources data/footprints/twdb \\
        --bbox=-97.9,30.1,-97.6,30.4 --out /tmp/austin.gpkg
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import shapely

from .gdal_security import harden_gdal, sniff_file_type

logger = logging.getLogger(__name__)

# OpenFileGDB must be allowed for .gdb sources; harden_gdal keeps it.
harden_gdal()

_EXT_KIND = {
    ".gpkg": "gpkg",
    ".geojson": "geojson",
    ".json": "geojson",
    ".parquet": "parquet",
    ".geoparquet": "parquet",
}
_RASTER_SNIFFS = {"tiff", "png", "jpeg"}
_MAX_DEPTH = 3
_ID_COLUMNS = (
    "id", "bldg_id", "building_id", "buildingid", "objectid", "fid", "uid",
    "uuid", "globalid",
)
_SUBTYPE_COLUMNS = ("subtype", "simp_type", "bldg_use", "building_type")
_CLASS_COLUMNS = ("class",)

SourcePaths = Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]


class FootprintSourceError(ValueError):
    """The input is not a usable footprint source (message is user-facing)."""


@dataclass(frozen=True)
class FootprintSource:
    """One readable dataset.

    ``path`` is what GDAL / pyarrow opens (may be a ``/vsizip/`` path);
    ``kind`` is ``gpkg`` | ``geojson`` | ``gdb`` | ``parquet``; ``label`` is a
    short human tag used in logs and to qualify ids.
    """

    path: str
    kind: str
    label: str


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _max_sources() -> int:
    return _int_env("HASTE_FOOTPRINT_MAX_SOURCES", 64)


def _max_uncompressed() -> int:
    return _int_env("HASTE_FOOTPRINT_MAX_UNCOMPRESSED_BYTES", 32 * 1024**3)


def _label(path: str) -> str:
    """``FPR03Bldgs_SVI_Pop_Feb2025.gdb.zip`` -> ``FPR03Bldgs_SVI_Pop_Feb2025``."""
    name = os.path.basename(path.rstrip("/\\"))
    low = name.lower()
    for suffix in (".gdb.zip", ".gdb", ".zip", ".gpkg", ".geojson", ".json",
                   ".parquet", ".geoparquet"):
        if low.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def _is_gdb_dir(path: str) -> bool:
    return os.path.isdir(path) and path.rstrip("/\\").lower().endswith(".gdb")


def _file_kind(path: str) -> Optional[str]:
    """Classify a file by content first, then by extension."""
    sniffed = sniff_file_type(path)
    if sniffed in _RASTER_SNIFFS:
        raise FootprintSourceError(
            f"{os.path.basename(path)!r} is a raster image, not building "
            "footprints (expected .gpkg, .geojson, .parquet, .gdb or a .zip "
            "of those)."
        )
    if sniffed == "zip":
        return "zip"
    if sniffed == "parquet":
        return "parquet"
    if sniffed == "gpkg":
        return "gpkg"
    if sniffed == "geojson":
        return "geojson"
    if sniffed == "sqlite":
        raise FootprintSourceError(
            f"{os.path.basename(path)!r} is a plain SQLite database, not a "
            "GeoPackage."
        )
    # Not recognised by magic bytes (BOM-prefixed JSON, empty file...):
    # trust the extension so the reader can produce a precise error.
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":
        return "zip"
    return _EXT_KIND.get(ext)


def _zip_sources(zip_path: str) -> List[FootprintSource]:
    """Find readable datasets inside a zip without extracting it."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise FootprintSourceError(
            f"{os.path.basename(zip_path)!r} is not a valid zip file."
        ) from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > 200_000:
            raise FootprintSourceError("Zip has too many entries.")
        if sum(i.file_size for i in infos) > _max_uncompressed():
            raise FootprintSourceError(
                "Zip expands beyond the allowed size "
                "(HASTE_FOOTPRINT_MAX_UNCOMPRESSED_BYTES)."
            )
        names = [i.filename for i in infos]

    for name in names:
        if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
            raise FootprintSourceError("Zip contains unsafe entry paths.")

    base = f"/vsizip/{zip_path}"
    out: List[FootprintSource] = []

    gdb_dirs = sorted(
        {n[: n.lower().index(".gdb/") + 4] for n in names if ".gdb/" in n.lower()}
    )
    for prefix in gdb_dirs:
        out.append(
            FootprintSource(f"{base}/{prefix}", "gdb", _label(prefix))
        )
    # A geodatabase zipped at the archive root (Esri's ``X.gdb.zip`` layout).
    if not gdb_dirs and any(n.lower().endswith(".gdbtable") for n in names):
        out.append(FootprintSource(base, "gdb", _label(zip_path)))

    if not out:
        for n in sorted(names):
            ext = os.path.splitext(n)[1].lower()
            if n.endswith("/") or "__MACOSX" in n:
                continue
            kind = _EXT_KIND.get(ext)
            if kind in ("gpkg", "geojson"):
                out.append(FootprintSource(f"{base}/{n}", kind, _label(n)))
    if not out:
        raise FootprintSourceError(
            f"{os.path.basename(zip_path)!r} contains no supported footprint "
            "data (need a .gdb folder, .gpkg or .geojson)."
        )
    return out


def discover_sources(paths: SourcePaths) -> List[FootprintSource]:
    """Expand files / folders / zips into a deterministic list of sources."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    found: List[FootprintSource] = []
    for raw in paths:
        found.extend(_discover_one(os.fspath(raw), depth=0))
    if not found:
        raise FootprintSourceError("No building-footprint data found.")
    found = _drop_format_duplicates(found)
    if len(found) > _max_sources():
        raise FootprintSourceError(
            f"Found {len(found)} footprint datasets; the limit is "
            f"{_max_sources()} (HASTE_FOOTPRINT_MAX_SOURCES)."
        )
    return found


_KIND_PRIORITY = {"parquet": 0, "gdb": 1, "gpkg": 2, "geojson": 3}


def _drop_format_duplicates(
    sources: List[FootprintSource],
) -> List[FootprintSource]:
    """Keep one copy when a folder holds the same dataset in several formats.

    ``X.gdb.zip`` next to its converted ``X.parquet`` share a label; loading
    both would double every building. The fastest format wins
    (parquet > gdb > gpkg > geojson). Same-label sources of the *same* kind
    are distinct files and are all kept.
    """
    best: dict = {}
    for src in sources:
        rank = _KIND_PRIORITY[src.kind]
        if src.label not in best or rank < best[src.label]:
            best[src.label] = rank
    kept = [s for s in sources if _KIND_PRIORITY[s.kind] == best[s.label]]
    if len(kept) != len(sources):
        logger.info(
            "Ignoring %d duplicate-format copies of the same dataset",
            len(sources) - len(kept),
        )
    return kept


def _discover_one(path: str, depth: int) -> List[FootprintSource]:
    if not os.path.exists(path):
        raise FootprintSourceError(f"Footprint source not found: {path}")
    if _is_gdb_dir(path):
        return [FootprintSource(path, "gdb", _label(path))]
    if os.path.isdir(path):
        if depth >= _MAX_DEPTH:
            return []
        out: List[FootprintSource] = []
        for name in sorted(os.listdir(path)):
            if name.startswith("."):
                continue
            child = os.path.join(path, name)
            if os.path.isdir(child) or _EXT_KIND.get(
                os.path.splitext(name)[1].lower()
            ) or name.lower().endswith(".zip"):
                try:
                    out.extend(_discover_one(child, depth + 1))
                except FootprintSourceError as exc:
                    # In a folder, ignore stray non-footprint files/zips
                    # (logs, rasters) rather than failing the whole set.
                    logger.info("Skipping %s: %s", child, exc)
        return out
    kind = _file_kind(path)
    if kind is None:
        raise FootprintSourceError(
            f"Unsupported footprint file type: {os.path.basename(path)!r}. "
            "Use .gpkg, .geojson, .parquet, .gdb or a .zip of those."
        )
    if kind == "zip":
        return _zip_sources(path)
    return [FootprintSource(path, kind, _label(path))]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _aoi_bounds_in(crs, aoi_polygon) -> Tuple[float, float, float, float]:
    return tuple(
        gpd.GeoSeries([aoi_polygon], crs="EPSG:4326").to_crs(crs).total_bounds
    )


def _overlaps(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _pick(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _choose_layer(src: FootprintSource, layer: Optional[str]) -> str:
    layers = pyogrio.list_layers(src.path)
    if len(layers) == 0:
        raise FootprintSourceError(f"{src.label}: dataset has no layers.")
    names = [str(n) for n, _ in layers]
    if layer:
        for n in names:
            if n.lower() == layer.lower():
                return n
        raise FootprintSourceError(
            f"{src.label}: layer {layer!r} not found (have {names})."
        )
    polygonal = [
        str(n) for n, gt in layers if gt and "polygon" in str(gt).lower()
    ]
    candidates = polygonal or names
    if len(candidates) == 1:
        return candidates[0]
    sized = [
        (pyogrio.read_info(src.path, layer=n).get("features", 0), n)
        for n in candidates
    ]
    chosen = max(sized)[1]
    logger.warning(
        "%s has %d candidate layers %s; using the largest, %r. Pass "
        "layer=... to override.", src.label, len(candidates), candidates, chosen,
    )
    return chosen


def _keep_columns(available: Iterable[str]) -> List[str]:
    keep = []
    for cands in (_ID_COLUMNS, _SUBTYPE_COLUMNS, _CLASS_COLUMNS):
        col = _pick(available, cands)
        if col and col not in keep:
            keep.append(col)
    return keep


@contextlib.contextmanager
def _ignore_gdb_spatial_index(enabled: bool):
    """Bypass the .spx spatial index while reading a file geodatabase.

    Esri geodatabases routinely ship a stale ``.spx`` (the TWDB regional
    files have indexes dated 2021 over 2025 tables). GDAL then answers a
    bbox query with *no* features, silently dropping every building. A
    sequential scan with the filter applied in C is correct and still
    fast (~1 s for 800k features), so never trust the index.
    """
    if not enabled:
        yield
        return
    key = "OPENFILEGDB_USE_SPATIAL_INDEX"
    pyogrio.set_gdal_config_options({key: "NO"})
    try:
        yield
    finally:
        pyogrio.set_gdal_config_options({key: None})


def _read_ogr(src: FootprintSource, aoi_polygon, layer) -> gpd.GeoDataFrame:
    with _ignore_gdb_spatial_index(src.kind == "gdb"):
        return _read_ogr_inner(src, aoi_polygon, layer)


def _read_ogr_inner(
    src: FootprintSource, aoi_polygon, layer
) -> gpd.GeoDataFrame:
    try:
        lyr = _choose_layer(src, layer)
        info = pyogrio.read_info(src.path, layer=lyr)
    except FootprintSourceError:
        raise
    except Exception:
        logger.warning(
            "%s: metadata read failed; falling back to a full read",
            src.label, exc_info=True,
        )
        return gpd.read_file(src.path)

    crs = info.get("crs")
    if not crs:
        if src.kind == "geojson":
            crs = "EPSG:4326"  # RFC 7946
        else:
            raise FootprintSourceError(
                f"{src.label}: dataset is missing a CRS; embed one "
                "(e.g. EPSG:4326) and retry."
            )
    bounds = _aoi_bounds_in(crs, aoi_polygon)

    total = info.get("total_bounds")
    if (
        total is not None
        and not any(isinstance(v, float) and math.isnan(v) for v in total)
        and not _overlaps(tuple(total), bounds)
    ):
        logger.info("%s: extent does not touch the AOI; skipped", src.label)
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    columns = _keep_columns(info.get("fields", []))
    try:
        gdf = gpd.read_file(
            src.path, layer=lyr, bbox=bounds, columns=columns,
            engine="pyogrio",
        )
    except Exception:
        logger.warning(
            "%s: bbox-filtered read failed; falling back to a full read "
            "(may use a lot of memory)", src.label, exc_info=True,
        )
        gdf = gpd.read_file(src.path, layer=lyr, columns=columns,
                            engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs(crs, allow_override=True)
    return gdf


def _read_parquet(src: FootprintSource, aoi_polygon) -> gpd.GeoDataFrame:
    schema = pq.read_schema(src.path)
    meta = (schema.metadata or {}).get(b"geo")
    if not meta:
        raise FootprintSourceError(
            f"{src.label}: parquet file has no GeoParquet 'geo' metadata."
        )
    geo = json.loads(meta)
    geom_col = geo["primary_column"]
    crs = geo["columns"][geom_col].get("crs") or "EPSG:4326"
    bounds = _aoi_bounds_in(crs, aoi_polygon)
    columns = _keep_columns(c for c in schema.names if c != geom_col)
    columns.append(geom_col)
    try:
        return gpd.read_parquet(src.path, bbox=bounds, columns=columns)
    except Exception:
        limit = _int_env("HASTE_FOOTPRINT_FULL_READ_MAX_BYTES", 1024**3)
        if os.path.getsize(src.path) > limit:
            raise FootprintSourceError(
                f"{src.label}: parquet has no usable bbox covering and is "
                "too large to read without one."
            )
        logger.warning("%s: parquet bbox read failed; full read", src.label,
                       exc_info=True)
        gdf = gpd.read_parquet(src.path, columns=columns)
        return gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]


def _normalise(gdf: gpd.GeoDataFrame, src: FootprintSource) -> gpd.GeoDataFrame:
    """Return ``id_raw/subtype/class/geometry`` in EPSG:4326, 2D polygons."""
    if gdf.crs is None:
        raise FootprintSourceError(
            f"{src.label}: dataset is missing a CRS; embed one "
            "(e.g. EPSG:4326) and retry."
        )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = shapely.force_2d(gdf.geometry.values)

    out = gpd.GeoDataFrame(geometry=gdf.geometry.values, crs=4326,
                           index=gdf.index)
    for name, cands in (
        ("id_raw", _ID_COLUMNS), ("subtype", _SUBTYPE_COLUMNS),
        ("class", _CLASS_COLUMNS),
    ):
        col = _pick([c for c in gdf.columns if c != "geometry"], cands)
        out[name] = gdf[col].values if col else None
    out["_src"] = src.label
    return out.reset_index(drop=True)


def _unique_ids(ids: pd.Series) -> pd.Series:
    """Suffix repeats with -2, -3... in order; the first keeps its id."""
    dup = ids.groupby(ids).cumcount()
    return ids.where(dup == 0, ids + "-" + (dup + 1).astype(str))


def read_footprints(
    paths: SourcePaths,
    aoi_polygon,
    *,
    layer: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Read, filter to the AOI bbox, and merge footprints from ``paths``.

    ``aoi_polygon`` is a shapely geometry in EPSG:4326. Returns a (possibly
    empty) GeoDataFrame with ``id``, ``subtype``, ``class``, ``geometry``.
    Raises :class:`FootprintSourceError` if the input is unusable or every
    source fails; in a multi-source set, one bad source is logged and
    skipped.
    """
    sources = discover_sources(paths)
    frames: List[gpd.GeoDataFrame] = []
    errors: List[str] = []
    for src in sources:
        try:
            if src.kind == "parquet":
                raw = _read_parquet(src, aoi_polygon)
            else:
                raw = _read_ogr(src, aoi_polygon, layer)
            frame = _normalise(raw, src)
        except Exception as exc:
            if len(sources) == 1:
                raise
            logger.warning("Skipping source %s: %s", src.label, exc)
            errors.append(f"{src.label}: {exc}")
            continue
        logger.info("%s (%s): %d footprints in AOI bbox", src.label,
                    src.kind, len(frame))
        if len(frame):
            frames.append(frame)

    if not frames:
        if errors and len(errors) == len(sources):
            raise FootprintSourceError(
                "Could not read any footprint source: " + "; ".join(errors)
            )
        return gpd.GeoDataFrame(
            {"id": [], "subtype": [], "class": []}, geometry=[], crs=4326
        )

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=4326
    )
    raw_ids = merged["id_raw"].astype("string").str.strip()
    missing = raw_ids.isna() | (raw_ids == "")
    raw_ids = raw_ids.mask(missing, pd.Series(merged.index, index=merged.index).astype(str))
    if len(sources) > 1:
        raw_ids = merged["_src"] + "-" + raw_ids
    merged["id"] = _unique_ids(raw_ids.astype(str))
    return merged[["id", "subtype", "class", "geometry"]]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("source", nargs="+", help="file, folder, .gdb or .zip")
    p.add_argument("--bbox", required=True,
                   help="AOI as xmin,ymin,xmax,ymax in EPSG:4326; use --bbox=-97.9,... (the = is required for negative values)")
    p.add_argument("--layer", default=None)
    p.add_argument("--out", default=None, help="optional .gpkg to write")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in args.bbox.split(","))
    except ValueError:
        p.error("--bbox must be 'xmin,ymin,xmax,ymax'")
    gdf = read_footprints(
        args.source, shapely.box(xmin, ymin, xmax, ymax), layer=args.layer
    )
    print(f"{len(gdf)} footprints")
    if args.out:
        gdf.to_file(args.out, driver="GPKG")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
