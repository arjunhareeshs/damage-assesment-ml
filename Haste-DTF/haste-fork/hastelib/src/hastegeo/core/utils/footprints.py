# Copyright (c) 2024 Overture Maps
# Licensed under the MIT License.
# Code from: https://github.com/OvertureMaps/overturemaps-py/blob/0fad53bceb955b14ac069ef321cbc2486996d5c7/overturemaps/core.py
# Modified to read from Azure Blob Storage instead of S3

"""Overture Maps client + a high-level building-footprint downloader.

This module is the single source of truth for fetching building footprints
inside HASTE. It is consumed by both the imageryprep workflow (where
footprints are cached per image layer) and historically by the inference
workflow (now slated to consume the cached gpkg).
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import fsspec
import geopandas as gpd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs
from geopandas import GeoDataFrame

from .building_ids import canonical_building_id  # noqa: F401 (re-exported)
from .gdal_security import harden_gdal

logger = logging.getLogger(__name__)

# Harden GDAL/OGR drivers before any geopandas/fiona read of a
# user-supplied vector file (GDAL CVE compensating control —
# docs/known-vulnerabilities.md Root Cause C).
harden_gdal()

OVERTURE_ACCOUNT_NAME = "overturemapswestus2"
FALLBACK_RELEASE = "2026-02-18.0"
# Matches Overture release names like "2026-02-18.0"
_RELEASE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")


type_theme_map = {
    "address": "addresses",
    "bathymetry": "base",
    "building": "buildings",
    "building_part": "buildings",
    "division": "divisions",
    "division_area": "divisions",
    "division_boundary": "divisions",
    "place": "places",
    "segment": "transportation",
    "connector": "transportation",
    "infrastructure": "base",
    "land": "base",
    "land_cover": "base",
    "land_use": "base",
    "water": "base",
}


# Expected output schema for a building-footprints GeoPackage. Downstream
# consumers (merge_with_building_footprints, GetBuildingFootprintsGeoJSON,
# the building-validation UI) tolerate missing ``subtype``/``class`` but
# always read by column name, so we synthesize a sentinel when the
# user-supplied input lacks them. Shared between the Overture path
# (download_building_footprints) and the user-supplied path
# (clip_and_normalize_user_footprints) so the two produce identical
# schemas.
_FOOTPRINT_OUTPUT_COLUMNS = ("id", "geometry", "subtype", "class")
# Which dataset each Overture building's geometry came from (OpenStreetMap,
# Microsoft ML Buildings, Google Open Buildings, Esri, ...). Overture merges
# them, and each was traced on different imagery, so each has its own offset
# from the roofs; footprint_alignment registers every source separately.
SOURCE_DATASET_COLUMN = "source_dataset"


def _geometry_source_dataset(sources) -> Optional[str]:
    """Dataset name of the entry describing the whole feature/geometry.

    Overture's ``sources`` is a list of ``{property, dataset, ...}``; the
    entry with an empty ``property`` covers the feature itself (its geometry).
    """
    if sources is None:
        return None
    try:
        entries = list(sources)
    except TypeError:
        return None
    if not entries:
        return None
    for entry in entries:
        if isinstance(entry, dict) and not entry.get("property"):
            return entry.get("dataset")
    first = entries[0]
    return first.get("dataset") if isinstance(first, dict) else None


def get_all_overture_types() -> List[str]:
    return list(type_theme_map.keys())


def record_batch_reader(
    overture_type, bbox=None, *, release: Optional[str] = None, columns=None
) -> Optional[pa.RecordBatchReader]:
    """Return a pyarrow RecordBatchReader for the desired bounding box(es)
    and Azure path.

    ``bbox`` is one ``(xmin, ymin, xmax, ymax)`` or a list of them; several
    boxes are OR-ed into ONE filter, so the dataset (every file footer of the
    release) is scanned once however many disjoint areas are asked for, and
    row groups outside all boxes are still skipped. ``release`` pins the
    Overture release (default: latest); ``columns`` limits what is read.
    """
    path = _dataset_path(overture_type, release)

    boxes = []
    if bbox:
        boxes = [bbox] if not isinstance(bbox[0], (list, tuple)) else list(bbox)
    filter = None
    for xmin, ymin, xmax, ymax in boxes:
        one = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )
        filter = one if filter is None else (filter | one)

    # Temporarily clear Azure storage env vars to prevent adlfs from using
    # local Azurite config when connecting to public Overture Maps blob storage
    saved_conn_str = os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
    saved_account = os.environ.pop("AZURE_STORAGE_ACCOUNT", None)

    try:
        t_fs = fsspec.filesystem(
            "az", account_name=OVERTURE_ACCOUNT_NAME, anon=True
        )
        pa_fs = fs.PyFileSystem(fs.FSSpecHandler(t_fs))

        dataset = ds.dataset(path, filesystem=pa_fs)
    finally:
        if saved_conn_str:
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = saved_conn_str
        if saved_account:
            os.environ["AZURE_STORAGE_ACCOUNT"] = saved_account

    batches = dataset.to_batches(filter=filter, columns=columns)

    # to_batches() can yield many empty batches; downstream consumers like
    # ParquetWriter emit a row group per batch which bloats output files.
    non_empty_batches = (b for b in batches if b.num_rows > 0)

    schema = dataset.schema
    if columns:
        schema = pa.schema([schema.field(c) for c in columns])
    geoarrow_schema = geoarrow_schema_adapter(schema)
    return pa.RecordBatchReader.from_batches(
        geoarrow_schema, non_empty_batches
    )


def geodataframe(
    overture_type: str, bbox: Tuple[float, float, float, float] = None
) -> GeoDataFrame:
    """Loads geoparquet for specified type into a geopandas dataframe.

    Args:
        overture_type: type to load (e.g. "building").
        bbox: optional bounding box (xmin, ymin, xmax, ymax) in EPSG:4326.

    Returns:
        GeoDataFrame with the optionally filtered theme data.
    """
    reader = record_batch_reader(overture_type, bbox)
    return gpd.GeoDataFrame.from_arrow(reader)


def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
    """Convert a geoarrow-compatible schema to a proper geoarrow schema.

    Assumes there is a single ``geometry`` column with WKB formatting.
    """
    geometry_field_index = schema.get_field_index("geometry")
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )
    return schema.set(geometry_field_index, geoarrow_geometry_field)


@lru_cache(maxsize=1)
def get_latest_release() -> str:
    """Discover the latest Overture Maps release from Azure Blob Storage.

    Lists the ``release/`` prefixes in the overturemapswestus2 container and
    returns the most recent version string (lexicographic sort works because
    release names follow the ``YYYY-MM-DD.N`` convention).

    Falls back to ``FALLBACK_RELEASE`` if the listing fails or returns no
    valid release names.
    """
    saved_conn_str = os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
    saved_account = os.environ.pop("AZURE_STORAGE_ACCOUNT", None)

    try:
        t_fs = fsspec.filesystem(
            "az", account_name=OVERTURE_ACCOUNT_NAME, anon=True
        )
        entries = t_fs.ls("release/")
        release_names = [
            entry.rstrip("/").split("/")[-1]
            for entry in entries
            if _RELEASE_PATTERN.match(entry.rstrip("/").split("/")[-1])
        ]
        if not release_names:
            logger.warning(
                "No valid Overture releases found, falling back to %s",
                FALLBACK_RELEASE,
            )
            return FALLBACK_RELEASE

        release_names.sort(reverse=True)
        latest = release_names[0]
        logger.info("Resolved latest Overture Maps release: %s", latest)
        return latest
    except Exception:
        logger.warning(
            "Failed to list Overture releases, falling back to %s",
            FALLBACK_RELEASE,
            exc_info=True,
        )
        return FALLBACK_RELEASE
    finally:
        if saved_conn_str:
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = saved_conn_str
        if saved_account:
            os.environ["AZURE_STORAGE_ACCOUNT"] = saved_account


def _dataset_path(overture_type: str, release: str = None) -> str:
    """Returns the Azure blob path of the Overture dataset to use."""
    if release is None:
        release = get_latest_release()
    theme = type_theme_map[overture_type]
    return f"release/{release}/theme={theme}/type={overture_type}/"


# Parallel footer reads when pruning Overture files. Each read is one or two
# small range requests; latency, not bandwidth, is what they wait on.
FOOTER_READ_THREADS = int(os.getenv("HASTE_OVERTURE_FOOTER_THREADS", "32"))


def _box_overlap(stats: dict, boxes) -> bool:
    """True if a row group's bbox statistics overlap any query box."""
    return any(
        stats["xmin"][0] < x1 and stats["xmax"][1] > x0
        and stats["ymin"][0] < y1 and stats["ymax"][1] > y0
        for x0, y0, x1, y1 in boxes
    )


def _row_filter(boxes):
    expr = None
    for xmin, ymin, xmax, ymax in boxes:
        one = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )
        expr = one if expr is None else (expr | one)
    return expr


def _overture_batches(overture_type, boxes, *, release=None, columns=None):
    """Batches of Overture rows inside any of ``boxes``, downloading only the
    row groups whose ``bbox`` statistics overlap one.

    The pyarrow dataset scan this replaces opened every file of the release
    (512 for buildings) and pulled far more than the footers through the
    fsspec handler, so a small AOI took minutes at laptop bandwidth and
    buffered GBs. Here footers are read in parallel, row groups are pruned
    on their statistics, and only matching ones are fetched, one file at a
    time (memory: one file's matching row groups). Rows are then filtered
    exactly on ``bbox``.
    """
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    path = _dataset_path(overture_type, release)
    saved_conn_str = os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
    saved_account = os.environ.pop("AZURE_STORAGE_ACCOUNT", None)
    try:
        az = fsspec.filesystem("az", account_name=OVERTURE_ACCOUNT_NAME, anon=True)
        files = sorted(f for f in az.ls(path, detail=False) if f.endswith(".parquet"))

        # cache_type="none": fetch exactly the ranges parquet asks for. The
        # default read-ahead cache doubled the time of each footer read.
        def matching(f):
            with az.open(f, "rb", cache_type="none") as fh:
                md = pq.ParquetFile(fh).metadata
            if md.num_row_groups == 0:
                return f, []
            names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
            ix = {k: names.index("bbox." + k) for k in ("xmin", "xmax", "ymin", "ymax")}
            keep = []
            for r in range(md.num_row_groups):
                rg = md.row_group(r)
                st = {k: rg.column(i).statistics for k, i in ix.items()}
                if any(s is None or not s.has_min_max for s in st.values()):
                    keep.append(r)  # no statistics: cannot rule it out
                    continue
                if _box_overlap({k: (s.min, s.max) for k, s in st.items()}, boxes):
                    keep.append(r)
            return f, keep

        with ThreadPoolExecutor(max(1, FOOTER_READ_THREADS)) as pool:
            plan = [(f, rgs) for f, rgs in pool.map(matching, files) if rgs]
        logger.info(
            "Overture %s: %d of %d files have row groups in the AOI (%d row groups)",
            overture_type, len(plan), len(files), sum(len(r) for _, r in plan),
        )
        row_filter = _row_filter(boxes)
        for f, rgs in plan:
            with az.open(f, "rb", cache_type="none") as fh:
                table = pq.ParquetFile(fh).read_row_groups(
                    rgs, columns=sorted(set(columns or []) | {"bbox"}) if columns else None
                )
            table = table.filter(row_filter)
            if columns:
                table = table.select(columns)
            schema = geoarrow_schema_adapter(table.schema)
            for batch in table.to_batches():
                if batch.num_rows:
                    yield pa.RecordBatch.from_arrays(batch.columns, schema=schema)
    finally:
        if saved_conn_str:
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = saved_conn_str
        if saved_account:
            os.environ["AZURE_STORAGE_ACCOUNT"] = saved_account


def keep_intersecting_whole(footprints, aoi_polygon):
    """Footprints that touch ``aoi_polygon`` (EPSG:4326), geometry untouched.

    A building that crosses the boundary, even by a corner, is kept as the
    whole building: slicing it (what ``gpd.clip`` did) stored a fragment,
    so the building was later cropped, masked and scored as part of
    itself. Invalid geometries are repaired first; only polygonal, non-empty
    features are returned.
    """
    import geopandas as _gpd

    footprints = footprints.copy()
    try:
        footprints["geometry"] = footprints.geometry.make_valid()
    except AttributeError:  # pragma: no cover - shapely < 2.0
        footprints["geometry"] = footprints.geometry.buffer(0)
    footprints = footprints[
        footprints.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        & ~footprints.geometry.is_empty
    ]
    aoi = _gpd.GeoSeries([aoi_polygon], crs="EPSG:4326")
    if footprints.crs is not None and footprints.crs != aoi.crs:
        aoi = aoi.to_crs(footprints.crs)
    return footprints[footprints.intersects(aoi.iloc[0])]


def _region_bboxes(bbox, aoi_polygon) -> List[Tuple[float, float, float, float]]:
    """One query box per disjoint AOI region, so imagery tiles kilometres
    apart do not pull every building in between."""
    if aoi_polygon is None or aoi_polygon.is_empty:
        return [tuple(bbox)]
    parts = list(getattr(aoi_polygon, "geoms", [aoi_polygon]))
    return [tuple(p.bounds) for p in parts if not p.is_empty]


def download_building_footprints(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    *,
    overwrite: bool = False,
    aoi_polygon=None,
    release: Optional[str] = None,
) -> int:
    """Download Overture Maps building footprints for an AOI to a GeoPackage.

    The output gpkg contains only Polygon/MultiPolygon features in EPSG:4326
    with columns ``id``, ``geometry``, ``subtype``, ``class``.

    Memory is bounded: Overture is read batch by batch, each batch is
    filtered to the AOI polygon and appended to the GeoPackage, so only one
    file's matching row groups are ever held. A multi-part AOI is read in ONE
    pass whose filter is the union of its parts' bounding boxes (imagery tiles
    kilometres apart do not pull every building in between), and only row
    groups whose statistics overlap a box are downloaded
    (``_overture_batches``).

    Args:
        bbox: AOI bounding box (xmin, ymin, xmax, ymax) in EPSG:4326, used
            when there is no ``aoi_polygon``.
        output_path: Destination ``.gpkg`` filename.
        overwrite: If False and the file already exists, raise FileExistsError.
        aoi_polygon: Optional shapely geometry in EPSG:4326. Footprints
            entirely outside it are dropped; every footprint that touches it
            is kept WHOLE (never sliced at the edge).
        release: Overture release to read (default: the latest). Pass it
            explicitly to record which dataset version was used.

    Returns:
        Number of features written.
    """
    import pyogrio

    if not output_path.endswith(".gpkg"):
        raise ValueError("output_path must end with .gpkg")
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Output file '{output_path}' already exists "
                "(pass overwrite=True to replace)."
            )
        os.remove(output_path)

    release = release or get_latest_release()
    columns = list(_FOOTPRINT_OUTPUT_COLUMNS)
    out_columns = columns + [SOURCE_DATASET_COLUMN]
    seen: set = set()
    written = 0
    scanned = 0
    batches = _overture_batches(
        "building", _region_bboxes(bbox, aoi_polygon), release=release,
        columns=columns + ["sources"],
    )
    for batch in batches:
        chunk = gpd.GeoDataFrame.from_arrow(pa.Table.from_batches([batch]))
        scanned += len(chunk)
        chunk = chunk[chunk.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        chunk = chunk.set_crs(epsg=4326, allow_override=True)
        if "sources" in chunk.columns:
            chunk[SOURCE_DATASET_COLUMN] = chunk["sources"].map(_geometry_source_dataset)
        else:
            chunk[SOURCE_DATASET_COLUMN] = None
        if aoi_polygon is not None:
            chunk = keep_intersecting_whole(chunk, aoi_polygon)
        chunk = chunk[~chunk["id"].isin(seen)]
        if chunk.empty:
            continue
        seen.update(chunk["id"].tolist())
        pyogrio.write_dataframe(
            chunk[out_columns], output_path, driver="GPKG",
            append=os.path.exists(output_path),
        )
        written += len(chunk)

    if not os.path.exists(output_path):
        gpd.GeoDataFrame(
            {"id": [], "subtype": [], "class": [], SOURCE_DATASET_COLUMN: []},
            geometry=[], crs=4326,
        ).to_file(output_path, driver="GPKG")
    logger.info(
        "Wrote %d Overture building footprints (release %s; %d scanned in "
        "the bounding boxes) to %s", written, release, scanned, output_path,
    )
    return written


def clip_and_normalize_user_footprints(
    input_path: str,
    aoi_polygon,
    output_path: str,
    *,
    overwrite: bool = False,
):
    """Select a user-supplied building-footprint file's buildings in an AOI.

    Reads ``input_path`` (a ``.gpkg``, ``.geojson``, ``.gdb`` folder,
    ``.gdb.zip``, GeoParquet, a zip of those, or a folder holding several —
    see :mod:`hastegeo.core.utils.footprint_sources`), reprojects to EPSG:4326 if needed, filters to polygonal
    geometries, keeps every building touching ``aoi_polygon`` (EPSG:4326)
    whole (never sliced at the edge), normalizes the
    schema to ``(id, geometry, subtype, class)`` synthesizing any
    missing non-geometry column, and writes a GeoPackage to
    ``output_path``.

    This is the "user-supplied" counterpart to
    :func:`download_building_footprints`: both produce GPKG files with
    the same schema so the rest of HASTE's pipeline can treat them
    interchangeably.

    Args:
        input_path: Path to a local building-footprint GPKG (or any
            geopandas-readable vector file).
        aoi_polygon: ``shapely.geometry.Polygon`` (or any geometry)
            describing the AOI, **in EPSG:4326**. Typically the output
            of :func:`hastegeo.core.utils.aoi.extract_aoi_polygon`.
        output_path: Destination ``.gpkg`` path.
        overwrite: Replace the output if it already exists.

    Returns:
        Number of features written.

    Raises:
        ImportError: If geopandas is not available.
        ValueError: If the input has no CRS, no polygonal geometries, or
            no building touches the AOI.
        FileExistsError: If ``overwrite`` is False and ``output_path``
            already exists.
    """
    if not output_path.endswith(".gpkg"):
        raise ValueError("output_path must end with .gpkg")
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Output file '{output_path}' already exists "
                "(pass overwrite=True to replace)."
            )
        os.remove(output_path)

    # footprint_sources handles .gpkg / .geojson / .gdb(.zip) / .parquet /
    # zips and folders of those, and pushes the AOI bbox into the read so a
    # statewide file is never fully loaded.
    from .footprint_sources import read_footprints

    gdf = read_footprints(input_path, aoi_polygon)
    if gdf.crs is None:
        raise ValueError(
            "Input is missing a CRS; embed one (e.g. EPSG:4326) and retry."
        )

    polygon_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf.loc[polygon_mask].copy()
    if gdf.empty:
        raise ValueError(
            "Input contains no Polygon/MultiPolygon features within the "
            "AOI bounding box."
        )

    if gdf.crs.to_epsg() != 4326:
        logger.info(
            "Reprojecting %d building footprints from %s to EPSG:4326",
            len(gdf),
            gdf.crs,
        )
        gdf = gdf.to_crs(epsg=4326)

    clipped = keep_intersecting_whole(gdf, aoi_polygon)

    if clipped.empty:
        raise ValueError(
            "No user-supplied building footprints intersected the AOI."
        )

    # Synthesize ``id`` from row index when missing; preserve any
    # existing values otherwise. Missing ``subtype``/``class`` are left
    # as ``None`` (downstream readers gate on column presence).
    #
    # The ``id`` column is force-cast to *string* even when the input
    # already provided it. The Overture path produces string GUIDs;
    # downstream consumers (the validation UI, then the validation /
    # assessment reports that join labels to inference) key on the
    # JSON-serialized id, which for an integer column comes through as
    # a JS number → stringly-coerced object key on PUT → str on the
    # server. Meanwhile this gpkg's fiona-read id stays int. That mix
    # caused 'No validation labels could be matched to inference
    # results' for every user GPKG with int ids (which is most GIS
    # data). Normalizing here makes the two paths interchangeable.
    if "id" not in clipped.columns:
        clipped = clipped.copy()
        clipped["id"] = [
            canonical_building_id(g, "user") for g in clipped.geometry
        ]
    else:
        clipped["id"] = clipped["id"].astype(str)
    if "subtype" not in clipped.columns:
        clipped["subtype"] = None
    if "class" not in clipped.columns:
        clipped["class"] = None

    out = clipped[list(_FOOTPRINT_OUTPUT_COLUMNS)].copy()
    out.set_crs(epsg=4326, inplace=True, allow_override=True)
    out.to_file(output_path, driver="GPKG")

    logger.info(
        "Wrote %d user-supplied building footprints (whole, touching the AOI) to %s",
        out.shape[0],
        output_path,
    )
    return int(out.shape[0])


def _main():
    """Argparse entry point so the workflow can call this in a subprocess.

    See ``ImageryWorkflow.download_building_footprints`` in
    ``hastegeo.workflows.prepare_imagery`` — it spawns
    ``python -m hastegeo.core.utils.footprints`` so a crash in pyarrow's
    native code (or a stuck Overture query) is contained to a subprocess and
    doesn't bring down the parent imageryprep workflow.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bbox",
        required=True,
        help="AOI bounding box in EPSG:4326 as 'xmin,ymin,xmax,ymax'",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination .gpkg filename",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    parser.add_argument(
        "--aoi-geojson",
        default=None,
        help=(
            "Optional path to a GeoJSON file with the valid-area polygon "
            "(EPSG:4326). Footprints whose geometry does not intersect "
            "this polygon are dropped, removing buildings from the bbox "
            "corners that fall outside the imagery's valid-data region. "
            "Loading failures are logged and AOI filtering is skipped — "
            "the bbox-only result is still written."
        ),
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Optional path; a JSON {source, version} describing the dataset is written there",
    )
    args = parser.parse_args()

    try:
        xmin, ymin, xmax, ymax = (float(v) for v in args.bbox.split(","))
    except ValueError as e:
        parser.error(f"--bbox must be 'xmin,ymin,xmax,ymax': {e}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    aoi_polygon = None
    if args.aoi_geojson:
        try:
            aoi_gdf = gpd.read_file(args.aoi_geojson)
            if aoi_gdf.crs is not None and aoi_gdf.crs.to_epsg() != 4326:
                aoi_gdf = aoi_gdf.to_crs(epsg=4326)
            aoi_polygon = aoi_gdf.union_all()
            if aoi_polygon.is_empty:
                logger.warning(
                    "AOI geojson %s produced an empty geometry; "
                    "skipping AOI filter",
                    args.aoi_geojson,
                )
                aoi_polygon = None
        except Exception:
            logger.warning(
                "Failed to load AOI geojson %s; falling back to "
                "bbox-only filtering",
                args.aoi_geojson,
                exc_info=True,
            )
            aoi_polygon = None

    release = get_latest_release()
    count = download_building_footprints(
        bbox=(xmin, ymin, xmax, ymax),
        output_path=args.output_path,
        overwrite=args.overwrite,
        aoi_polygon=aoi_polygon,
        release=release,
    )
    if args.meta_out:
        import json

        with open(args.meta_out, "w") as f:
            json.dump({"source": "overture", "version": release}, f)
    sys.stdout.write(f"{count}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _main()
