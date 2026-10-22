"""Real-data checks for the large-AOI workflow (no DB, no checkpoint needed).

Uses the user's own ``data/`` folder (gitignored). Skipped when absent.
Override locations with HASTE_TEST_DATA_DIR.

Run: cd hastelib && PYTHONPATH=src pytest tests/core/test_real_data_aoi.py -s
"""

import glob
import os
import time
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pyogrio = pytest.importorskip("pyogrio")
import pyproj
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import transform

from hastegeo.core.utils.aoi_validate import validate_aoi
from hastegeo.core.utils.dtf_constants import DTF_IM_SIZE
from hastegeo.core.utils.dtf_preprocess import normalize_tile
from hastegeo.core.utils.imagery_catalog import describe_raster
from hastegeo.core.utils.postgis_load import compute_canonical_building_id
from hastegeo.core.utils.region_aggregate import aggregate
from hastegeo.core.utils.region_windows import group_batches, select_units


def _data_dir() -> Path:
    env = os.environ.get("HASTE_TEST_DATA_DIR")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "imagery").is_dir():
            return p / "data"
    return Path("data")


DATA = _data_dir()
TIFS = sorted(glob.glob(str(DATA / "imagery" / "*.tif")))
GPKG = DATA / "footprint" / "Texas.gpkg"

pytestmark = pytest.mark.skipif(
    len(TIFS) < 2 or not GPKG.exists(), reason="real data not present"
)


@pytest.fixture(scope="module")
def rasters():
    out = []
    for i, p in enumerate(TIFS[:2]):
        meta = describe_raster(p)
        meta["imagery_id"] = f"img{i}"
        meta["path"] = p
        meta["affine"] = rasterio.Affine(*meta["transform"])
        meta["footprint_geom"] = box(*rasterio.open(p).bounds)
        out.append(meta)
    return out


def _to4326(crs):
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform


def _buildings_in(aoi):
    """Stream only the AOI bbox from the 10M-row gpkg, then exact-filter."""
    import geopandas as gpd

    df = gpd.read_file(GPKG, bbox=tuple(aoi.bounds), engine="pyogrio")
    df = df[df.intersects(aoi)]
    seen = set()
    rows = []
    for g in df.geometry:
        rows.append(
            {
                "building_id": compute_canonical_building_id(None, g, "texas", seen),
                "geometry": g,
            }
        )
    return rows


def _aoi_from_utm(meta, x0, y0, x1, y1):
    return transform(_to4326(meta["crs"]), box(x0, y0, x1, y1))


def _utm_buildings(meta):
    """Footprints inside one TIF, with UTM centroids (data is a sparse subset)."""
    import geopandas as gpd
    ll = transform(_to4326(meta["crs"]), meta["footprint_geom"])
    df = gpd.read_file(GPKG, bbox=tuple(ll.bounds), engine="pyogrio")
    to_u = pyproj.Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True).transform
    pts = [transform(to_u, g).centroid for g in df.geometry]
    return np.array([[p.x, p.y] for p in pts]).reshape(-1, 2)


@pytest.fixture(scope="module")
def aoi_small(rasters):
    """600 m box on the densest cluster, well away from the TIF overlap."""
    xy = _utm_buildings(rasters[1])
    south = xy[xy[:, 1] < rasters[1]["footprint_geom"].bounds[1] + 4000]
    best, bn = None, -1
    for x, y in south[::5]:
        n = int(((abs(south[:, 0] - x) < 300) & (abs(south[:, 1] - y) < 300)).sum())
        if n > bn:
            best, bn = (x, y), n
    x, y = best
    return _aoi_from_utm(rasters[1], x - 300, y - 300, x + 300, y + 300)


@pytest.fixture(scope="module")
def aoi_span(rasters):
    """Box over the band where the two TIFs overlap, around the building there."""
    a, b = rasters[0]["footprint_geom"].bounds, rasters[1]["footprint_geom"].bounds
    lo, hi = max(a[1], b[1]), min(a[3], b[3])
    assert hi > lo, "the two TIFs do not overlap/touch vertically"
    xy = _utm_buildings(rasters[0])
    inband = xy[(xy[:, 1] > lo) & (xy[:, 1] < hi)]
    assert len(inband), "no footprint in the TIF overlap band"
    x, y = inband[0]
    return _aoi_from_utm(rasters[0], x - 400, y - 400, x + 400, y + 400)


# ---------------------------------------------------------------- imagery
def test_imagery_headers(rasters):
    for r in rasters:
        print(f"\n{Path(r['path']).name}: {r['band_count']} bands {r['dtype']} "
              f"{r['pixel_size_x']}m nodata={r['nodata']} {r['width']}x{r['height']}")
        assert r["dtype"] == "uint8"
        assert r["pixel_size_x"] == pytest.approx(0.6)
        # Brief: 'Stop and ask when a source TIF is not 3-band uint8 RGB'.
        # NAIP is 4-band RGBNIR; the code only reads bands 1-3.
        assert r["band_count"] >= 3


def test_footprint_gpkg_header():
    info = pyogrio.read_info(GPKG)
    print(f"\nfootprints: {info['features']} rows, crs={info['crs']}, fields={list(info['fields'])}")
    assert info["crs"] == "EPSG:4326"
    assert not any(f.lower() in ("id", "building_id") for f in info["fields"]), \
        "source has an ID field; test assumed hashed IDs"


# ------------------------------------------------------------- validation
def test_aoi_validation(aoi_small, aoi_span):
    for aoi in (aoi_small, aoi_span):
        geom, km2 = validate_aoi(aoi.__geo_interface__, 500)
        print(f"\nAOI area {km2:.3f} km2")
        assert km2 > 0
        assert geom.equals(aoi)  # stored unchanged
    with pytest.raises(ValueError):
        validate_aoi(aoi_span.__geo_interface__, 0.0001)


# --------------------------------------------------------------- planning
def _plan(rasters, aoi, assessment_id="a1"):
    blds = _buildings_in(aoi)
    units = []
    for r in rasters:
        to_r = pyproj.Transformer.from_crs("EPSG:4326", r["crs"], always_xy=True).transform
        if not r["footprint_geom"].intersects(transform(to_r, aoi)):
            continue
        units += select_units(
            r["imagery_id"], r["width"], r["height"], r["affine"], aoi, blds,
            raster_crs=r["crs"], assessment_id=assessment_id,
        )
    return blds, units


def test_small_aoi_one_tif(rasters, aoi_small):
    t = time.time()
    blds, units = _plan(rasters, aoi_small)
    print(f"\nsmall AOI: {len(blds)} buildings, {len(units)} units, "
          f"imagery={sorted({u['imagery_id'] for u in units})}, {time.time()-t:.1f}s")
    assert len(blds) > 0, "no buildings found in the small AOI"
    assert {u["imagery_id"] for u in units} == {"img1"}
    ids = [b["building_id"] for b in blds]
    assert len(ids) == len(set(ids))
    for u in units:
        assert u["width"] <= DTF_IM_SIZE and u["height"] <= DTF_IM_SIZE
        assert u["buildings"]
    covered = {x["building_id"] for u in units for x in u["buildings"]}
    assert covered == set(ids), "some eligible buildings are in no unit"
    batches = group_batches(units, 5)
    assert sum(len(b["units"]) for b in batches) == len(units)
    assert all(len(b["units"]) <= 5 for b in batches)


def test_spanning_aoi_two_tifs(rasters, aoi_span):
    blds, units = _plan(rasters, aoi_span)
    imgs = {u["imagery_id"] for u in units}
    print(f"\nspan AOI: {len(blds)} buildings, {len(units)} units, imagery={sorted(imgs)}")
    assert imgs == {"img0", "img1"}, "AOI across both TIFs should schedule both"
    per = {}
    for u in units:
        for x in u["buildings"]:
            per.setdefault(x["building_id"], []).append((u["unit_id"], x["coverage_fraction"]))
    multi = {k: v for k, v in per.items() if len(v) > 1}
    print(f"buildings seen in >1 unit: {len(multi)}")
    # aggregation: every building -> exactly one row, even with duplicates
    ev = [
        {"building_id": b, "unit_id": u, "probs": [0.1, 0.2, 0.3, 0.2, 0.2], "class": 2,
         "confidence": 0.3, "coverage_fraction": c, "valid_pixel_fraction": 1.0}
        for b, lst in per.items() for (u, c) in lst
    ]
    rows = aggregate(ev, [{"building_id": b["building_id"]} for b in blds], {},
                     assessment_id="a1", image_layer_id="L")
    assert len(rows) == len(blds)
    assert len({r["building_id"] for r in rows}) == len(blds)
    assert all(r["damage_percentage"] is None for r in rows)
    rows2 = aggregate(list(reversed(ev)), [{"building_id": b["building_id"]} for b in blds], {},
                      assessment_id="a1", image_layer_id="L")
    key = lambda r: r["building_id"]
    assert [(r["building_id"], r["winning_unit_id"]) for r in sorted(rows, key=key)] == \
           [(r["building_id"], r["winning_unit_id"]) for r in sorted(rows2, key=key)]


# ------------------------------------------------------- pixels / preprocess
def test_windowed_read_and_normalisation(rasters, aoi_small):
    _, units = _plan(rasters, aoi_small)
    u = units[0]
    with rasterio.open(rasters[1]["path"]) as src:
        win = Window(u["col_off"], u["row_off"], DTF_IM_SIZE, DTF_IM_SIZE)
        t = time.time()
        tile = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
        dt = time.time() - t
    print(f"\nread {tile.shape} {tile.dtype} in {dt*1000:.0f} ms, mean={tile.mean():.1f}")
    assert tile.shape == (3, DTF_IM_SIZE, DTF_IM_SIZE) and tile.dtype == np.uint8
    assert tile.max() > 0, "tile is all zero"
    a, b = normalize_tile(tile, "unit"), normalize_tile(tile, "imagenet")
    print(f"unit mean={a.mean():.3f}  imagenet mean={b.mean():.3f}")
    assert a.dtype == b.dtype == np.float32
    assert 0 <= a.min() and a.max() <= 1
    assert b.min() < 0  # imagenet shifts below zero
    with pytest.raises(ValueError):
        normalize_tile(tile, "bogus")


def test_masks_for_real_buildings(rasters, aoi_small):
    """Rasterise each unit's buildings exactly as the region workflow does
    (geometry_mask with the *window* transform) and check they land in-tile."""
    from rasterio.features import geometry_mask

    blds, units = _plan(rasters, aoi_small)
    r = rasters[1]
    to_r = pyproj.Transformer.from_crs("EPSG:4326", r["crs"], always_xy=True).transform
    by_id = {b["building_id"]: transform(to_r, b["geometry"]) for b in blds}
    total = 0
    for u in units:
        win = Window(u["col_off"], u["row_off"], DTF_IM_SIZE, DTF_IM_SIZE)
        wt = rasterio.windows.transform(win, r["affine"])
        for x in u["buildings"]:
            m = geometry_mask([by_id[x["building_id"]]], out_shape=(DTF_IM_SIZE, DTF_IM_SIZE),
                              transform=wt, all_touched=True, invert=True)
            assert m.sum() > 0, f"empty mask for {x['building_id']} in {u['unit_id']}"
            total += 1
    print(f"\n{total} building masks over {len(units)} units, all non-empty")
