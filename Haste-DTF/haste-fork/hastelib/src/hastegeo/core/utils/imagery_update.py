# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Replace part of an image layer's imagery with newer TIFs.

After a disaster, new imagery usually arrives for part of an area a layer
already covers. This module works out what each new TIF covers, which of
the layer's existing TIFs it replaces, and puts it on exactly the same
pixel grid as the existing imagery so the result tiles without seams.

1. **Where is a TIF?** Its header gives the CRS and pixel grid, so its
   rectangle on the ground. The rectangle includes empty collars, so the
   footprint used here is the outline of the pixels that hold data
   (``valid_data_footprint``), read at reduced resolution so it costs
   seconds and little memory even for very large files.

2. **Which existing TIF does it correspond to?** Footprints are compared
   (``plan_imagery_update``). An existing TIF whose data area the new
   imagery covers almost entirely (``RETIRE_COVERAGE``) is retired. One
   covered only in part stays, with the new imagery drawn on top of it.
   New imagery outside every existing TIF extends the layer.

3. **Perfect alignment and tiling** (``coregister_to_reference``). Every
   new TIF is resampled onto the existing imagery's grid: same CRS, same
   pixel size, edges snapped to the same pixel lines, so tiles meet pixel
   for pixel with no gap, overlap or half-pixel seam. The new TIF is then
   registered to the old imagery of the same place by matching edges
   (roofs, roads); a shift is applied only when several windows agree.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from pyproj import Geod, Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from .footprint_alignment import (
    _best_window_shift,
    _edge_strength,
    _metres_per_pixel,
    _metres_per_pixel_xy,
)

logger = logging.getLogger(__name__)

# An existing TIF is retired when new imagery covers at least this share of
# its data area; below it, it is kept under the new imagery.
RETIRE_COVERAGE = 0.98
# Overlaps smaller than this share of a TIF are edge slivers, not a match.
MIN_OVERLAP = 0.01
# Reduced-resolution read used to trace a TIF's data area.
FOOTPRINT_MAX_DIM = 2048
# Image-to-image registration.
REGISTER_MAX_SHIFT_M = 10.0
REGISTER_WINDOW_PX = 1024
REGISTER_MAX_WINDOWS = 16
REGISTER_MIN_WINDOWS = 3
REGISTER_MIN_VALID = 0.8
REGISTER_MAX_SPREAD_PX = 1.0
REGISTER_MIN_GAIN = 1.1

_GEOD = Geod(ellps="WGS84")


# --------------------------------------------------------------------------
# 1. Where is a TIF?
# --------------------------------------------------------------------------


def _has_alpha(src) -> bool:
    return any(ci == ColorInterp.alpha for ci in src.colorinterp)


def _data_bands(src) -> List[int]:
    """Band indexes holding image data (an alpha band is a mask, not data)."""
    bands = list(range(1, src.count + 1))
    if _has_alpha(src) and len(bands) > 1:
        bands = bands[:-1]
    return bands


def valid_data_footprint(path: str, max_dim: int = FOOTPRINT_MAX_DIM) -> BaseGeometry:
    """Outline of the pixels that hold data, as a (Multi)Polygon in EPSG:4326.

    A pixel holds data unless it is nodata, transparent (alpha 0) or, for a
    file that declares neither, 0 in every band (black collars).
    """
    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / max_dim)
        out_h = max(1, int(round(src.height / scale)))
        out_w = max(1, int(round(src.width / scale)))
        if src.nodata is not None or _has_alpha(src):
            mask = src.dataset_mask(out_shape=(out_h, out_w), resampling=Resampling.nearest) > 0
        else:
            data = src.read(_data_bands(src), out_shape=(len(_data_bands(src)), out_h, out_w),
                            resampling=Resampling.nearest)
            mask = np.any(data != 0, axis=0)
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        crs = src.crs

    polygons = [
        shape(geom)
        for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if value == 1
    ]
    if not polygons:
        raise ValueError(f"{path} holds no valid data")
    # Drop holes and smooth the stair-steps of the reduced-resolution mask.
    pixel = abs(transform.a)
    outline = unary_union([p.buffer(pixel / 2).buffer(-pixel / 2) for p in polygons])
    outline = outline.simplify(pixel, preserve_topology=True)
    if crs is not None and crs.to_epsg() != 4326:
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        outline = shapely_transform(to_wgs84, outline.segmentize(pixel * 8))
    return outline


def area_km2(geom: BaseGeometry) -> float:
    """Geodesic area of an EPSG:4326 geometry in km²."""
    if geom.is_empty:
        return 0.0
    return abs(_GEOD.geometry_area_perimeter(geom)[0]) / 1e6


def describe_tif(path: str) -> Dict[str, Any]:
    """Where a TIF is and what it holds.

    Returns its data-area footprint (EPSG:4326, GeoJSON and km²), its
    rectangle, CRS, pixel size in metres, size, bands and any capture date
    recorded in the file's tags.
    """
    footprint = valid_data_footprint(path)
    with rasterio.open(path) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        tags = src.tags()
        acquired = next(
            (tags[k] for k in ("ACQUISITION_DATE", "TIFFTAG_DATETIME", "DATE_ACQUIRED", "acquisition_date")
             if tags.get(k)),
            None,
        )
        info = {
            "file": os.path.basename(path),
            "crs": src.crs.to_string() if src.crs else None,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "data_bands": len(_data_bands(src)),
            "dtype": src.dtypes[0],
            "pixel_size_m": round(_metres_per_pixel(src), 4),
            "bounds_4326": [round(v, 7) for v in bounds],
            "acquired": acquired,
        }
    info["footprint"] = mapping(footprint)
    info["area_km2"] = round(area_km2(footprint), 6)
    return info


# --------------------------------------------------------------------------
# 2. Which existing TIF does it correspond to?
# --------------------------------------------------------------------------


def plan_imagery_update(
    existing: Sequence[Dict[str, Any]],
    incoming: Sequence[Dict[str, Any]],
    retire_coverage: float = RETIRE_COVERAGE,
) -> Dict[str, Any]:
    """Decide what the new imagery replaces.

    ``existing`` and ``incoming`` are lists of ``{"id": ..., "footprint":
    GeoJSON or shapely geometry (EPSG:4326)}`` (extra keys are kept).

    Returns ``{"existing": [...], "incoming": [...], "keep_ids": [...],
    "retire_ids": [...], "updated_area": GeoJSON, "updated_area_km2"}``.
    ``keep_ids`` is the mosaic order for the kept existing TIFs; the
    incoming TIFs go after them so their pixels are drawn on top.
    """

    def geom(item):
        g = item["footprint"]
        return g if isinstance(g, BaseGeometry) else shape(g)

    new_geoms = [geom(i) for i in incoming]
    new_union = unary_union(new_geoms) if new_geoms else None
    old_geoms = [geom(e) for e in existing]
    old_union = unary_union(old_geoms) if old_geoms else None

    existing_out, keep_ids, retire_ids = [], [], []
    for item, g in zip(existing, old_geoms):
        g_area = area_km2(g)
        covered = area_km2(g.intersection(new_union)) / g_area if new_union and g_area else 0.0
        if covered >= retire_coverage:
            action = "retire"
            retire_ids.append(item["id"])
        else:
            action = "keep_under_new" if covered >= MIN_OVERLAP else "unchanged"
            keep_ids.append(item["id"])
        existing_out.append({**_strip(item), "covered_by_new": round(covered, 4), "action": action})

    incoming_out = []
    for item, g in zip(incoming, new_geoms):
        g_area = area_km2(g)
        overlaps = []
        for old, og in zip(existing, old_geoms):
            shared = area_km2(g.intersection(og))
            if g_area and shared / g_area >= MIN_OVERLAP or (area_km2(og) and shared / area_km2(og) >= MIN_OVERLAP):
                overlaps.append({
                    "id": old["id"],
                    "share_of_new": round(shared / g_area, 4) if g_area else 0.0,
                    "share_of_existing": round(shared / area_km2(og), 4) if area_km2(og) else 0.0,
                })
        overlaps.sort(key=lambda o: -o["share_of_new"])
        outside = area_km2(g.difference(old_union)) / g_area if old_union and g_area else 1.0
        incoming_out.append({
            **_strip(item),
            "area_km2": round(g_area, 6),
            "corresponds_to": overlaps,
            "new_area_share": round(outside, 4),
        })

    return {
        "existing": existing_out,
        "incoming": incoming_out,
        "keep_ids": keep_ids,
        "retire_ids": retire_ids,
        "updated_area": mapping(new_union) if new_union is not None else None,
        "updated_area_km2": round(area_km2(new_union), 6) if new_union is not None else 0.0,
    }


def _strip(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if k != "footprint"}


# --------------------------------------------------------------------------
# 3. Perfect alignment and tiling
# --------------------------------------------------------------------------


@dataclass
class Grid:
    """A pixel grid: CRS, pixel size and one pixel corner (the origin)."""

    crs: str
    res_x: float
    res_y: float
    origin_x: float
    origin_y: float

    def snap(self, minx, miny, maxx, maxy):
        """Expand bounds outward to the nearest pixel lines of this grid."""
        x0 = self.origin_x + math.floor((minx - self.origin_x) / self.res_x + 1e-9) * self.res_x
        x1 = self.origin_x + math.ceil((maxx - self.origin_x) / self.res_x - 1e-9) * self.res_x
        y1 = self.origin_y - math.floor((self.origin_y - maxy) / self.res_y + 1e-9) * self.res_y
        y0 = self.origin_y - math.ceil((self.origin_y - miny) / self.res_y - 1e-9) * self.res_y
        return x0, y0, x1, y1


def reference_grid(paths: Sequence[str]) -> Grid:
    """The grid of the existing imagery: the first TIF's CRS, origin and the
    finest pixel size among the TIFs in that CRS (never coarsen imagery)."""
    with rasterio.open(paths[0]) as first:
        crs = first.crs
        t = first.transform
        res_x, res_y = abs(t.a), abs(t.e)
        origin_x, origin_y = t.c, t.f
    for p in paths[1:]:
        with rasterio.open(p) as src:
            if src.crs == crs:
                res_x = min(res_x, abs(src.transform.a))
                res_y = min(res_y, abs(src.transform.e))
    return Grid(crs.to_string(), res_x, res_y, origin_x, origin_y)


def _gdal_ok(dataset, action: str, path: str):
    """Close a GDAL result, raising if the call failed.

    GDAL's process-wide exception mode is left alone (other HASTE code
    relies on its default), so failures are detected from the result.
    """
    from osgeo import gdal

    if dataset is None:
        raise RuntimeError(f"GDAL could not {action} {path}: {gdal.GetLastErrorMsg()}")
    dataset.FlushCache()
    return None


def _on_grid(src, grid: Grid, tol: float = 1e-3) -> bool:
    """True when ``src`` already uses ``grid``'s CRS, pixel size and lines."""
    if src.crs is None or src.crs.to_string() != grid.crs:
        return False
    t = src.transform
    if t.b or t.d:
        return False
    if abs(abs(t.a) - grid.res_x) > tol * grid.res_x or abs(abs(t.e) - grid.res_y) > tol * grid.res_y:
        return False
    fx = (t.c - grid.origin_x) / grid.res_x
    fy = (grid.origin_y - t.f) / grid.res_y
    return abs(fx - round(fx)) < tol and abs(fy - round(fy)) < tol


def warp_to_grid(src_path: str, out_path: str, grid: Grid, resampling: str = "bilinear") -> str:
    """Resample ``src_path`` onto ``grid`` as a tiled GeoTIFF with nodata 0.

    The output covers the source's extent snapped outward to the grid's
    pixel lines. An alpha band becomes the nodata mask (and is dropped), and
    a file with neither nodata nor alpha has its 0-valued collars treated as
    nodata, matching how the rest of the pipeline reads imagery.
    """
    from osgeo import gdal

    with rasterio.open(src_path) as src:
        bounds = transform_bounds(src.crs, grid.crs, *src.bounds, densify_pts=21)
        alpha = _has_alpha(src)
        bands = _data_bands(src)
        src_nodata = src.nodata
    out_bounds = grid.snap(*bounds)
    warp_kwargs = dict(
        format="VRT",
        dstSRS=grid.crs,
        xRes=grid.res_x,
        yRes=grid.res_y,
        outputBounds=out_bounds,
        resampleAlg=resampling,
        dstNodata=0,
        multithread=True,
    )
    if alpha:
        warp_kwargs.update(srcAlpha=True, dstAlpha=True)
        warp_kwargs.pop("dstNodata")
    elif src_nodata is None:
        warp_kwargs["srcNodata"] = 0
    vrt_path = out_path + ".warp.vrt"
    try:
        _gdal_ok(gdal.Warp(vrt_path, src_path, options=gdal.WarpOptions(**warp_kwargs)), "warp", src_path)
        _gdal_ok(gdal.Translate(
            out_path,
            vrt_path,
            options=gdal.TranslateOptions(
                bandList=list(range(1, len(bands) + 1)),
                noData=0,
                creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
            ),
        ), "write", out_path)
    finally:
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
    return out_path


@dataclass
class ImageShift:
    """Registration of a new image to the reference imagery."""

    dx_px: float = 0.0
    dy_px: float = 0.0
    dx_m: float = 0.0
    dy_m: float = 0.0
    windows_used: int = 0
    spread_px: float = 0.0
    gain: float = 1.0
    applied: bool = False
    reason: str = ""

    def as_dict(self):
        return asdict(self)


def measure_image_shift(
    moving_path: str,
    reference_path: str,
    max_shift_m: float = REGISTER_MAX_SHIFT_M,
    window_px: int = REGISTER_WINDOW_PX,
    max_windows: int = REGISTER_MAX_WINDOWS,
) -> ImageShift:
    """Integer-pixel shift that lays ``moving_path`` onto ``reference_path``.

    Both must be on the same grid (``warp_to_grid``). Edge strength of both
    images is cross-correlated on windows spread over their overlap; the
    median shift is accepted only when the windows agree.
    """
    result = ImageShift()
    with rasterio.open(moving_path) as mov, rasterio.open(reference_path) as ref:
        if mov.crs != ref.crs or not np.allclose(mov.res, ref.res):
            raise ValueError("moving and reference images must share one grid")
        left = max(mov.bounds.left, ref.bounds.left)
        right = min(mov.bounds.right, ref.bounds.right)
        bottom = max(mov.bounds.bottom, ref.bounds.bottom)
        top = min(mov.bounds.top, ref.bounds.top)
        if right <= left or top <= bottom:
            result.reason = "no overlap with existing imagery"
            return result
        m_per_px = _metres_per_pixel(mov)
        mx, my = _metres_per_pixel_xy(mov)
        max_px = max(1, min(int(math.ceil(max_shift_m / m_per_px)), window_px // 4))
        sigma = max(1.0, 0.25 / m_per_px)
        res_x, res_y = mov.res
        win_w = min(window_px, int((right - left) / res_x))
        win_h = min(window_px, int((top - bottom) / res_y))
        if win_w < 4 * max_px or win_h < 4 * max_px:
            result.reason = "overlap with existing imagery too small to register"
            return result
        # Candidate windows on a regular lattice over the overlap.
        n = max(1, int(math.ceil(math.sqrt(max_windows * 4))))
        xs = np.linspace(left, right - win_w * res_x, n)
        ys = np.linspace(top, bottom + win_h * res_y, n)
        mbands, rbands = _data_bands(mov)[:3], _data_bands(ref)[:3]
        shifts, gains = [], []
        for y in ys:
            for x in xs:
                if len(shifts) >= max_windows:
                    break
                wb = (x, y - win_h * res_y, x + win_w * res_x, y)
                mw = rasterio.windows.from_bounds(*wb, transform=mov.transform).round_offsets().round_lengths()
                rw = rasterio.windows.from_bounds(*wb, transform=ref.transform).round_offsets().round_lengths()
                m = mov.read(mbands, window=mw, boundless=True, fill_value=0)
                r = ref.read(rbands, window=rw, boundless=True, fill_value=0)
                if m.shape[1:] != r.shape[1:]:
                    continue
                valid = np.any(m != 0, axis=0) & np.any(r != 0, axis=0)
                if valid.mean() < REGISTER_MIN_VALID:
                    continue
                em = _edge_strength(m) * valid
                er = _edge_strength(r) * valid
                best = _best_window_shift(er, em - em.mean(), max_px, sigma)
                if best is None:
                    continue
                shifts.append(best[:2])
                gains.append(best[2])

    result.windows_used = len(shifts)
    if len(shifts) < REGISTER_MIN_WINDOWS:
        result.reason = f"only {len(shifts)} usable window(s); need {REGISTER_MIN_WINDOWS}"
        return result
    arr = np.array(shifts, dtype=float)
    dx, dy = np.median(arr, axis=0)
    spread = float(np.median(np.abs(arr - [dx, dy]).max(axis=1)))
    result.dx_px, result.dy_px = float(dx), float(dy)
    result.dx_m, result.dy_m = float(dx * mx), float(-dy * my)
    result.spread_px = spread
    result.gain = float(np.median(gains))
    shift_px = max(abs(dx), abs(dy))
    if shift_px < 1:
        result.reason = "already registered"
    elif spread > max(REGISTER_MAX_SPREAD_PX, 0.5 * shift_px):
        result.reason = f"windows disagree (spread {spread:.1f} px for a {shift_px:.0f} px shift)"
    elif result.gain < REGISTER_MIN_GAIN:
        result.reason = f"fit gain {result.gain:.3f} below {REGISTER_MIN_GAIN}"
    else:
        result.applied = True
        result.reason = "registered"
    return result


def apply_image_shift(path: str, shift: ImageShift) -> None:
    """Move an image by whole pixels (it stays on its grid) in place."""
    if not shift.applied:
        return
    with rasterio.open(path, "r+") as dst:
        t = dst.transform
        dst.transform = rasterio.Affine(
            t.a, t.b, t.c + shift.dx_px * t.a,
            t.d, t.e, t.f + shift.dy_px * t.e,
        )


def build_reference(paths: Sequence[str], grid: Grid, out_vrt: str) -> str:
    """A virtual mosaic of ``paths`` on ``grid`` (no pixels copied)."""
    from osgeo import gdal

    # GDAL cannot warp several files into one VRT in a single call, so each
    # file is put on the grid (virtually) on its own, keeping only its data
    # bands with nodata 0, and those per-file VRTs are mosaicked. Every one
    # is snapped to ``grid``, so all pixels line up exactly with the files
    # ``warp_to_grid`` writes.
    parts_dir = out_vrt + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    band_count = None
    with_alpha = []
    for p in paths:
        with rasterio.open(p) as src:
            n = len(_data_bands(src))
            band_count = n if band_count is None else min(band_count, n)
            with_alpha.append((_has_alpha(src), src.nodata, transform_bounds(
                src.crs, grid.crs, *src.bounds, densify_pts=21)))
    parts = []
    for i, (p, (alpha, nodata, bounds)) in enumerate(zip(paths, with_alpha)):
        warp_kwargs = dict(
            format="VRT", dstSRS=grid.crs, xRes=grid.res_x, yRes=grid.res_y,
            outputBounds=grid.snap(*bounds), resampleAlg="bilinear",
        )
        if alpha:
            warp_kwargs.update(srcAlpha=True, dstAlpha=True)
        else:
            warp_kwargs.update(dstNodata=0, **({"srcNodata": 0} if nodata is None else {}))
        warped = os.path.join(parts_dir, f"part_{i:05d}_warp.vrt")
        _gdal_ok(gdal.Warp(warped, p, options=gdal.WarpOptions(**warp_kwargs)), "warp", p)
        part = os.path.join(parts_dir, f"part_{i:05d}.vrt")
        _gdal_ok(gdal.Translate(part, warped, options=gdal.TranslateOptions(
            format="VRT", bandList=list(range(1, band_count + 1)), noData=0)), "select bands of", p)
        parts.append(part)
    _gdal_ok(gdal.BuildVRT(out_vrt, parts, options=gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0)),
             "mosaic", out_vrt)
    return out_vrt


def coregister_to_reference(
    src_path: str, reference_vrt: str, grid: Grid, out_path: str
) -> Dict[str, Any]:
    """Put ``src_path`` on ``grid`` and register it to the reference imagery.

    Returns ``{"file": out_path, "registration": ImageShift dict}``.
    """
    warp_to_grid(src_path, out_path, grid)
    try:
        shift = measure_image_shift(out_path, reference_vrt)
    except Exception as exc:  # registration is an improvement, never a blocker
        logger.warning("Registration of %s failed: %s", src_path, exc)
        shift = ImageShift(reason=f"registration failed: {exc}")
    apply_image_shift(out_path, shift)
    logger.info(
        "Co-registered %s: %s (dx=%.2f m, dy=%.2f m, %d windows)",
        os.path.basename(src_path), shift.reason, shift.dx_m, shift.dy_m, shift.windows_used,
    )
    return {"file": out_path, "registration": shift.as_dict()}


# --------------------------------------------------------------------------
# Whole update
# --------------------------------------------------------------------------


@dataclass
class UpdateFile:
    path: str
    url: Optional[str] = None


@dataclass
class UpdateResult:
    """What ``prepare_update`` decided. ``mosaic_paths`` is the ordered list
    of files to mosaic (kept existing first, new on top)."""

    mosaic_paths: List[str] = field(default_factory=list)
    active_urls: List[str] = field(default_factory=list)
    retired_urls: List[str] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


def prepare_update(
    existing: Sequence[UpdateFile], incoming: Sequence[UpdateFile], work_dir: str
) -> UpdateResult:
    """Align the new TIFs to the existing imagery and decide what they replace."""
    if not existing:
        raise ValueError("an imagery update needs the layer's existing imagery")
    if not incoming:
        raise ValueError("an imagery update needs at least one new TIF")
    os.makedirs(work_dir, exist_ok=True)

    grid = reference_grid([e.path for e in existing])
    reference = build_reference([e.path for e in existing], grid, os.path.join(work_dir, "reference.vrt"))

    aligned, registrations = [], []
    for i, item in enumerate(incoming):
        stem = os.path.splitext(os.path.basename(item.path))[0]
        out = os.path.join(work_dir, f"aligned_{i:03d}_{stem}.tif")
        reg = coregister_to_reference(item.path, reference, grid, out)
        aligned.append(out)
        registrations.append(reg["registration"])

    existing_desc = [
        {"id": i, "url": e.url, **describe_tif(e.path)} for i, e in enumerate(existing)
    ]
    incoming_desc = [
        {"id": f"new{i}", "url": item.url, **describe_tif(path), "registration": reg}
        for i, (item, path, reg) in enumerate(zip(incoming, aligned, registrations))
    ]
    plan = plan_imagery_update(existing_desc, incoming_desc)
    # A file whose source URL is unknown is never retired: retiring it would
    # drop it from the mosaic while the layer could not record which URL went.
    for i in [i for i in plan["retire_ids"] if not existing[i].url]:
        plan["retire_ids"].remove(i)
        plan["keep_ids"].append(i)
        plan["existing"][i]["action"] = "keep_under_new"
    plan["keep_ids"].sort()

    keep = [existing[i] for i in plan["keep_ids"]]
    retired = [existing[i] for i in plan["retire_ids"]]
    # Kept files go through the same grid step as the new ones, so every
    # mosaic input has the same bands, nodata and pixel lines: the mosaic
    # then tiles pixel for pixel. A file already on the grid is copied with
    # nearest-neighbour, which leaves its pixels exactly as they were.
    kept_paths = []
    for i, item in zip(plan["keep_ids"], keep):
        stem = os.path.splitext(os.path.basename(item.path))[0]
        out = os.path.join(work_dir, f"kept_{i:04d}_{stem}.tif")
        with rasterio.open(item.path) as src:
            resampling = "near" if _on_grid(src, grid) else "bilinear"
        kept_paths.append(warp_to_grid(item.path, out, grid, resampling=resampling))
    report = {
        "grid": asdict(grid),
        **plan,
    }
    return UpdateResult(
        mosaic_paths=kept_paths + aligned,
        active_urls=[k.url for k in keep if k.url] + [i.url for i in incoming if i.url],
        retired_urls=[r.url for r in retired if r.url],
        report=report,
    )
