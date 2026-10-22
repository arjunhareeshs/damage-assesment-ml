"""Walk a large AOI through the HASTE large-AOI pipeline and print every stage.

Run from this folder:   python check.py
Options:                python check.py --miles 10 --windows-per-batch 20 --masks-per-forward 64

Everything below runs the real code from hastelib on the real files in data/
(imagery TIFs + footprint gpkg). Nothing is mocked or made up. Stages that need
things this script cannot do without services (database, queue, model weights)
are reported as NOT RUN, never faked.

Stages
  1 inputs        2 catalogue (TIF headers)   3 AOI + validation
  4 imagery discovery   5 building discovery  6 tiling funnel + map
  7 units / sub-tiles   8 mask splitting      9 batching
  10 scheduling         11 real pixel read    12 aggregation preview
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _ensure_env():
    """Use an interpreter that has the geo stack; fall back to anaconda."""
    try:
        import rasterio, geopandas, pyogrio, shapely  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("HASTE_CHECK_REEXEC"):
        sys.exit("rasterio/geopandas/pyogrio/shapely are not installed in this Python.")
    cands = [Path.home() / "anaconda3" / "python.exe", Path.home() / "miniconda3" / "python.exe"]
    for c in cands:
        if c.exists():
            print(f"[info] this Python lacks the geo stack; re-running with {c}\n")
            env = dict(os.environ, HASTE_CHECK_REEXEC="1")
            sys.exit(subprocess.call([str(c), *sys.argv], env=env))
    sys.exit("No Python with rasterio/geopandas found. Install them or run from the conda env.")


_ensure_env()

# A Windows PostgreSQL install exports PROJ vars that break pyproj/rasterio.
for _v in ("PROJ_LIB", "GDAL_DATA", "PROJ_DATA"):
    if "PostgreSQL" in os.environ.get(_v, ""):
        os.environ.pop(_v, None)

for cand in (HERE / "Haste-DTF" / "haste-fork" / "hastelib" / "src", HERE / "hastelib" / "src"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))
        break

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import pyproj  # noqa: E402
import rasterio  # noqa: E402
import rasterio.windows  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402
from shapely.ops import transform  # noqa: E402

from hastegeo.core.utils.aoi_validate import validate_aoi  # noqa: E402
from hastegeo.core.utils.dtf_constants import DTF_IM_SIZE, compute_tile_windows  # noqa: E402
from hastegeo.core.utils.dtf_preprocess import PREPROCESSING_VERSION, normalize_tile  # noqa: E402
from hastegeo.core.utils.imagery_catalog import describe_raster  # noqa: E402
from hastegeo.core.utils.postgis_load import compute_canonical_building_id  # noqa: E402
from hastegeo.core.utils.region_windows import group_batches, select_units  # noqa: E402

MILE_M = 1609.344
W = shutil.get_terminal_size((100, 20)).columns


def banner(n, title):
    print("\n" + "=" * min(W, 100))
    print(f" STEP {n}: {title}")
    print("=" * min(W, 100))


def kv(k, v):
    print(f"  {k:<34} {v}")


def note(msg):
    print(f"  -> {msg}")


def to_crs(src, dst):
    return pyproj.Transformer.from_crs(src, dst, always_xy=True).transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miles", type=float, default=10.0, help="AOI side length (square), miles")
    ap.add_argument("--center", help="lon,lat of AOI centre (default: centre of the imagery)")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--windows-per-batch", type=int, help="env HASTE_REGION_WINDOWS_PER_BATCH")
    ap.add_argument("--masks-per-forward", type=int, help="env HASTE_REGION_MASKS_PER_FORWARD")
    ap.add_argument("--max-concurrent", type=int, help="env HASTE_REGION_MAX_CONCURRENT_BATCHES")
    ap.add_argument("--max-attempts", type=int, help="env HASTE_REGION_MAX_ATTEMPTS")
    ap.add_argument("--lease-seconds", type=int, help="env HASTE_REGION_LEASE_SECONDS")
    ap.add_argument("--max-area-km2", type=float, help="env HASTE_AOI_MAX_AREA_KM2")
    ap.add_argument("--max-units", type=int, help="env HASTE_AOI_MAX_UNITS")
    ap.add_argument("--normalization", default=os.environ.get("HASTE_DTF_NORMALIZATION", "imagenet"))
    a = ap.parse_args()
    t_all = time.time()

    def param(arg, env, cast, fallback, label):
        if arg is not None:
            return arg, "--flag"
        if os.environ.get(env):
            return cast(os.environ[env]), f"env {env}"
        return fallback, f"NOT SET (shown value is only for this walkthrough; production needs {env})"

    windows_per_batch, s1 = param(a.windows_per_batch, "HASTE_REGION_WINDOWS_PER_BATCH", int, 20, "")
    masks_per_forward, s2 = param(a.masks_per_forward, "HASTE_REGION_MASKS_PER_FORWARD", int, 64, "")
    max_conc, s3 = param(a.max_concurrent, "HASTE_REGION_MAX_CONCURRENT_BATCHES", int, 1, "")
    max_attempts, s4 = param(a.max_attempts, "HASTE_REGION_MAX_ATTEMPTS", int, 3, "")
    lease_s, s5 = param(a.lease_seconds, "HASTE_REGION_LEASE_SECONDS", int, 300, "")
    max_area, s6 = param(a.max_area_km2, "HASTE_AOI_MAX_AREA_KM2", float, 500.0, "")
    max_units, s7 = param(a.max_units, "HASTE_AOI_MAX_UNITS", int, 500, "")

    # ------------------------------------------------------------------ 1
    banner(1, "Inputs found on disk")
    data = Path(a.data)
    tifs = sorted(glob.glob(str(data / "imagery" / "*.tif")))
    gpkgs = sorted(glob.glob(str(data / "footprint" / "*.gpkg")))
    if not tifs or not gpkgs:
        sys.exit(f"Need data/imagery/*.tif and data/footprint/*.gpkg under {data}")
    gpkg = gpkgs[0]
    for t in tifs:
        kv("imagery", f"{Path(t).name}  ({os.path.getsize(t)/1e6:.0f} MB)")
    info = pyogrio.read_info(gpkg)
    kv("footprints", f"{Path(gpkg).name}  {info['features']:,} rows, crs={info['crs']}")
    kv("footprint fields", list(info["fields"]))
    kv("tile size (model constraint)", f"{DTF_IM_SIZE} px")
    kv("normalisation / version", f"{a.normalization} / {PREPROCESSING_VERSION}")
    print("  parameters (source):")
    for n, v, s in [("windows_per_batch", windows_per_batch, s1), ("masks_per_forward", masks_per_forward, s2),
                    ("max_concurrent_batches", max_conc, s3), ("max_attempts", max_attempts, s4),
                    ("lease_seconds", lease_s, s5), ("max_area_km2", max_area, s6), ("max_units", max_units, s7)]:
        print(f"    {n:<24} {v:<8} {s}")

    # ------------------------------------------------------------------ 2
    banner(2, "Imagery catalogue (header-only read, no pixels)")
    rasters = []
    for i, p in enumerate(tifs):
        m = describe_raster(p)
        m.update(imagery_id=f"img{i}", path=p, name=Path(p).name,
                 affine=rasterio.Affine(*m["transform"]))
        m["foot4326"] = shape(m["footprint"])
        rasters.append(m)
        print(f"  [{m['imagery_id']}] {m['name']}")
        kv("size / bands / dtype", f"{m['width']}x{m['height']} px, {m['band_count']} bands, {m['dtype']}")
        kv("pixel size / nodata", f"{m['pixel_size_x']} m / {m['nodata']}")
        kv("crs", m["crs"])
        kv("footprint (lon/lat bounds)", tuple(round(x, 4) for x in m["foot4326"].bounds))
        grid = len(compute_tile_windows(m["width"], m["height"], DTF_IM_SIZE))
        kv("1024-px grid windows", grid)
        if m["band_count"] != 3:
            note(f"{m['band_count']} bands: the pipeline reads bands 1-3 only (assumed R,G,B). "
                 "Brief says to confirm with you when a TIF is not 3-band RGB.")

    # ------------------------------------------------------------------ 3
    banner(3, f"Draw the AOI ({a.miles:g} x {a.miles:g} miles) and validate it")
    allb = [r["foot4326"].bounds for r in rasters]
    if a.center:
        clon, clat = (float(x) for x in a.center.split(","))
    else:
        clon = (min(b[0] for b in allb) + max(b[2] for b in allb)) / 2
        clat = (min(b[1] for b in allb) + max(b[3] for b in allb)) / 2
    utm = rasters[0]["crs"]
    cx, cy = to_crs("EPSG:4326", utm)(clon, clat)
    half = a.miles * MILE_M / 2
    aoi_geojson = transform(to_crs(utm, "EPSG:4326"), box(cx - half, cy - half, cx + half, cy + half)).__geo_interface__
    kv("centre lon/lat", f"{clon:.5f}, {clat:.5f}")
    kv("side length", f"{a.miles:g} mi = {a.miles*MILE_M/1000:.2f} km (measured in {utm})")
    kv("GeoJSON type / vertices", f"{aoi_geojson['type']} / {len(aoi_geojson['coordinates'][0])}")
    try:
        aoi, area_km2 = validate_aoi(aoi_geojson, max_area)
    except ValueError as e:
        sys.exit(f"  AOI REJECTED at submit: {e}")
    kv("validated area (EPSG:6933)", f"{area_km2:.2f} km2  ({area_km2/2.58999:.1f} sq mi)")
    kv("limit check", f"{area_km2:.1f} <= {max_area:g} km2  OK")
    note("Pipeline rule: reject (not repair) invalid, empty, lon/lat-out-of-range, antimeridian, over-limit AOIs.")

    # ------------------------------------------------------------------ 4
    banner(4, "Imagery discovery (which TIFs does the AOI touch?)")
    hit = []
    for r in rasters:
        inter = r["foot4326"].intersection(aoi)
        frac = transform(to_crs("EPSG:4326", "EPSG:6933"), inter).area / 1e6 if not inter.is_empty else 0
        r["aoi_overlap_km2"] = frac
        tag = "TOUCHED" if not inter.is_empty else "outside AOI -> no units will be made"
        print(f"  [{r['imagery_id']}] {r['name']}: overlap {frac:.2f} km2  {tag}")
        if not inter.is_empty:
            hit.append(r)
    if not hit:
        sys.exit("  no imagery covers this AOI -> assessment would be 'failed'")
    kv("TIFs the AOI crosses", f"{len(hit)} of {len(rasters)}")

    # ------------------------------------------------------------------ 5
    banner(5, "Building discovery (stream footprints, intersect AOI)")
    import geopandas as gpd
    t = time.time()
    chunk = int(os.environ.get("HASTE_FOOTPRINT_LOAD_CHUNK", "50000"))
    parts, off = [], 0
    while True:
        df = pyogrio.read_dataframe(gpkg, bbox=tuple(aoi.bounds), skip_features=off, max_features=chunk)
        if df.empty:
            break
        parts.append(df)
        off += len(df)
        if len(df) < chunk:
            break
    import pandas as pd
    cand = (gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
            if parts else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    kv("bbox candidates read", f"{len(cand):,} in {time.time()-t:.1f}s (chunks of {chunk:,}, whole file never loaded)")
    cand = cand[cand.intersects(aoi)].reset_index(drop=True)
    kv("intersect AOI polygon", f"{len(cand):,} eligible buildings")
    seen, blds = set(), []
    to6933 = to_crs("EPSG:4326", "EPSG:6933")
    for g in cand.geometry:
        bid = compute_canonical_building_id(None, g, "texas", seen)
        within = aoi.contains(g)
        ov = 1.0 if within else transform(to6933, g.intersection(aoi)).area / max(transform(to6933, g).area, 1e-9)
        blds.append({"building_id": bid, "geometry": g, "boundary_intersecting": not within,
                     "aoi_overlap_fraction": float(min(1.0, ov))})
    kv("canonical IDs", f"{len(seen):,} unique of {len(blds):,} (no source ID field -> sha1 of rounded WKB)")
    nb = sum(b["boundary_intersecting"] for b in blds)
    kv("boundary-crossing buildings", f"{nb:,}  (kept, flagged boundary_intersecting)")
    if blds:
        fr = [b["aoi_overlap_fraction"] for b in blds if b["boundary_intersecting"]]
        if fr:
            kv("their AOI overlap fraction", f"min {min(fr):.2f}  median {np.median(fr):.2f}")
    else:
        note("zero buildings -> assessment completes with zero rows (valid outcome)")

    # ------------------------------------------------------------------ 6
    banner(6, "Tiling funnel per TIF: grid -> in AOI -> has buildings -> unit")
    all_units = []
    for r in hit:
        to_r = to_crs("EPSG:4326", r["crs"])
        aoi_r = transform(to_r, aoi)
        wins = compute_tile_windows(r["width"], r["height"], DTF_IM_SIZE)
        in_aoi = [w for w in wins if box(*rasterio.windows.bounds(w, r["affine"])).intersects(aoi_r)]
        tif_blds = [b for b in blds if b["geometry"].intersects(r["foot4326"])]
        units = select_units(r["imagery_id"], r["width"], r["height"], r["affine"], aoi, tif_blds,
                             raster_crs=r["crs"], assessment_id="check")
        grid_units = [u for u in units if u["kind"] == "grid"]
        cen_units = [u for u in units if u["kind"] == "building_centred"]
        r["units"], r["wins"] = units, wins
        all_units += units
        print(f"  [{r['imagery_id']}] {r['name']}")
        kv("buildings on this TIF", f"{len(tif_blds):,}")
        kv("grid windows total", len(wins))
        kv("  intersect the AOI", f"{len(in_aoi)}  (skipped outside AOI: {len(wins)-len(in_aoi)})")
        kv("  AND hold >=1 building -> kept", f"{len(grid_units)}  (empty-of-buildings dropped: {len(in_aoi)-len(grid_units)})")
        kv("building-centred extra units", f"{len(cen_units)}  (buildings cut by a window edge)")
        cols = math.ceil(r["width"] / DTF_IM_SIZE)
        keep = {(u["col_off"], u["row_off"]) for u in grid_units}
        inaoi = {(int(w.col_off), int(w.row_off)) for w in in_aoi}
        print("  window map  (#=unit  o=in AOI, no buildings  .=outside AOI), north at top:")
        for ri in range(math.ceil(r["height"] / DTF_IM_SIZE)):
            row = ""
            for ci in range(cols):
                k = (ci * DTF_IM_SIZE, ri * DTF_IM_SIZE)
                row += " #" if k in keep else (" o" if k in inaoi else " .")
            print("    " + row)
    if len(all_units) > max_units:
        print(f"  !! {len(all_units)} units exceeds HASTE_AOI_MAX_UNITS={max_units} -> assessment would be rejected")
    kv("TOTAL units for assessment", len(all_units))

    # ------------------------------------------------------------------ 7
    banner(7, "Units and sub-tiles (what one unit is)")
    print("  A unit = one 1024x1024 window of one TIF. A 'building_centred' unit is an extra window")
    print("  centred on a building that a grid edge cut, so it can be seen whole.")
    for r in hit:
        us = sorted(r["units"], key=lambda u: -len(u["buildings"]))
        for u in us[:5]:
            covs = [b["coverage_fraction"] for b in u["buildings"]]
            print(f"  {u['unit_id']}")
            print(f"     {u['kind']:<17} offset=({u['col_off']},{u['row_off']}) {u['width']}x{u['height']}px "
                  f"buildings={len(u['buildings'])} coverage min={min(covs):.2f}")
        if len(us) > 5:
            print(f"  ... {len(us)-5} more units on {r['imagery_id']}")
    per = {}
    for u in all_units:
        for b in u["buildings"]:
            per.setdefault(b["building_id"], []).append((u["unit_id"], b["coverage_fraction"], u["kind"]))
    kv("buildings scheduled", f"{len(per):,} of {len(blds):,} eligible")
    miss = {b["building_id"] for b in blds} - set(per)
    on_tif = {b["building_id"] for b in blds
              if any(b["geometry"].intersects(r["foot4326"]) for r in hit)}
    dropped = miss & on_tif          # on a TIF but in no unit = a real scheduling bug
    no_imagery = miss - on_tif       # AOI is bigger than the imagery: nothing to score
    kv("outside all imagery", f"{len(no_imagery):,}  (no pixels exist -> insufficient_data, never class 0)")
    kv("on a TIF but in NO unit", f"{len(dropped)}" + ("  <-- BUG: would be silently unscored" if dropped else "  OK"))
    multi = {k: v for k, v in per.items() if len(v) > 1}
    kv("buildings seen in >1 unit", f"{len(multi)}  (resolved in step 12)")

    # ------------------------------------------------------------------ 8
    banner(8, f"Mask splitting: masks_per_forward = {masks_per_forward}")
    fwd = 0
    for u in all_units:
        n = len(u["buildings"])
        u["forwards"] = math.ceil(n / masks_per_forward)
        fwd += u["forwards"]
    big = max(all_units, key=lambda u: len(u["buildings"]))
    kv("forward passes, whole assessment", fwd)
    kv("busiest unit", f"{big['unit_id']}: {len(big['buildings'])} masks -> {big['forwards']} pass(es) of the SAME tile")
    note("On CUDA OOM the worker halves the chunk; at 1 it falls back to CPU; if that fails the unit is 'failed'.")

    # ------------------------------------------------------------------ 9
    banner(9, f"Batching: <= {windows_per_batch} units per runner task, one TIF per batch")
    batches = group_batches(all_units, windows_per_batch, assessment_id="check")
    for b in batches:
        print(f"  {b['batch_id']:<28} tif={b['imagery_id']} units={len(b['units']):>3} "
              f"masks={sum(len(u['buildings']) for u in b['units']):>5} status={b['status']}")
    kv("batches", len(batches))

    # ------------------------------------------------------------------ 10
    banner(10, "Scheduling (how the Postprocessor drives the batches)")
    waves = math.ceil(len(batches) / max_conc) if batches else 0
    print(f"  max_concurrent_batches={max_conc}  max_attempts={max_attempts}  lease={lease_s}s")
    print("  Each queue message runs one poll: promote pending batches up to the concurrency cap,")
    print("  renew leases of running ones, expire stale leases (attempts+1; at max_attempts -> failed),")
    print("  then re-enqueue itself until all units are completed/failed.\n")
    print("  planned dispatch order (assuming no failures):")
    for w in range(waves):
        grp = batches[w * max_conc:(w + 1) * max_conc]
        print(f"    wave {w+1}: " + ", ".join(b["batch_id"] for b in grp))
    print("\n  assessment status path:  submitted -> validating -> discovering -> tiling -> queued")
    print("                           -> running -> aggregating -> completed | completed_with_errors")
    print(f"  worst case if every batch needs {max_attempts} attempts: {len(batches)*max_attempts} task runs")
    note("Live queue, DB leases and the runner are NOT RUN here (need the Docker stack up).")

    # ------------------------------------------------------------------ 11
    banner(11, "Real pixel work for the first/last unit of each TIF (read + normalise + masks)")
    for r in hit:
        if not r["units"]:
            continue
        to_r = to_crs("EPSG:4326", r["crs"])
        bmap = {b["building_id"]: b["geometry"] for b in blds}
        with rasterio.open(r["path"]) as src:
            for u in (r["units"][0], r["units"][-1]) if len(r["units"]) > 1 else (r["units"][0],):
                win = Window(u["col_off"], u["row_off"], DTF_IM_SIZE, DTF_IM_SIZE)
                t0 = time.time()
                tile = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                t_read = time.time() - t0
                t0 = time.time()
                x = normalize_tile(tile, a.normalization)
                t_norm = time.time() - t0
                wt = rasterio.windows.transform(win, r["affine"])
                t0 = time.time()
                ms = [geometry_mask([transform(to_r, bmap[b["building_id"]])], out_shape=(DTF_IM_SIZE, DTF_IM_SIZE),
                                    transform=wt, all_touched=True, invert=True) for b in u["buildings"]]
                t_mask = time.time() - t0
                px = [int(m.sum()) for m in ms]
                nodata = float((tile == 0).all(axis=0).mean())
                print(f"  {u['unit_id']}")
                kv("tile read", f"{tuple(tile.shape)} {tile.dtype}  {t_read*1000:.0f} ms")
                kv(f"normalise ({a.normalization})", f"mean {tile.mean()/255:.3f} -> {x.mean():.3f}  {t_norm*1000:.0f} ms")
                kv("all-zero (nodata-like) pixels", f"{nodata*100:.1f}% of tile")
                kv("masks", f"{len(ms)} built in {t_mask*1000:.0f} ms; px/building min {min(px)} max {max(px)}"
                   + ("  <-- EMPTY MASK" if min(px) == 0 else ""))
    ck = os.environ.get("DTF_CHECKPOINT_PATH")
    if ck and Path(ck).exists():
        note(f"checkpoint present at {ck}; model forward is exercised by the region workflow, not by this script.")
    else:
        note("MODEL FORWARD NOT RUN: no checkpoint (set DTF_CHECKPOINT_PATH). Seconds/window is unmeasured.")

    # ------------------------------------------------------------------ 12
    banner(12, "Aggregation preview: one row per building")
    print("  Rule: highest coverage_fraction wins; tie -> highest confidence; tie -> lowest unit_id.")
    print("  Never averaged, independent of arrival order. (Confidence unknown here, so only the")
    print("  coverage rule is previewed.)")
    kv("buildings in 1 unit", len(per) - len(multi))
    kv("buildings in 2+ units", len(multi))
    shown = 0
    for k, v in multi.items():
        v = sorted(v, key=lambda t: (-t[1], t[0]))
        if shown < 5:
            print(f"    {k[:12]}..  " + " | ".join(f"{u.split('_',2)[-1][:22]} cov={c:.2f}" for u, c, _ in v)
                  + f"   => winner {v[0][0].split('_',2)[-1][:22]}")
            shown += 1
    # Run the REAL aggregate() on buildings with no evidence at all (outside
    # imagery). Scored buildings need the model, so they stay pending.
    from collections import Counter
    from hastegeo.core.utils.region_aggregate import aggregate
    rows = aggregate([], [{"building_id": b["building_id"],
                           "boundary_intersecting": b["boundary_intersecting"],
                           "aoi_overlap_fraction": b["aoi_overlap_fraction"]}
                          for b in blds if b["building_id"] in no_imagery],
                     {}, assessment_id="check", image_layer_id="check")
    kv("real aggregate() on no-imagery set", dict(Counter(r["result_status"] for r in rows)))
    bad = [r for r in rows if any(r[k] is not None for k in
                                  ("damage_class", "damage_class_name", "confidence", "damage_percentage"))]
    kv("any class/conf/% filled in?", "NO  OK" if not bad else f"YES {len(bad)} <-- BUG (must be NULL)")
    print("\n  final result rows, by how each will end:")
    kv("  scored by model (pending inference)", f"{len(per):,}")
    kv("  insufficient_data (outside imagery)", f"{len(no_imagery):,}")
    total = len(per) + len(no_imagery) + len(dropped)
    kv("  TOTAL (one row per eligible building)", f"{total:,} of {len(blds):,}" + ("" if total == len(blds) else "  <-- MISMATCH"))
    note("damage_percentage is NULL in every row.")

    print("\n" + "=" * min(W, 100))
    print(f" DONE in {time.time()-t_all:.1f}s.  NOT RUN: model inference, DB writes, queue, API, UI.")
    print("=" * min(W, 100))


if __name__ == "__main__":
    main()
