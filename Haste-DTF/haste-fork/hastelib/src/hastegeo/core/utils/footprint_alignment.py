# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Register building footprints to the imagery they will be scored on.

Footprint datasets (Overture, Microsoft, county files) are rarely drawn on
the same orthophoto that is being assessed, so they are often offset from
the roofs by one to a few metres. Every footprint then crops and masks the
wrong pixels: part of the roof is cut off and part of the yard is scored.

This module measures that offset from the imagery itself and removes it.
Footprint outlines are rasterised and slid over the image's edge strength
(gradient magnitude); the shift where outlines best follow roof edges wins.
The search runs on several windows spread over the image and the median is
taken, so one window with trees or shadows cannot decide the result. The
shift is applied only when the windows agree and it clearly improves the
fit; otherwise the footprints are returned unchanged.

One translation is applied to the whole layer. That matches how these
offsets arise (a dataset registered to a different base image) and keeps
neighbouring buildings consistent with each other.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import rasterio
import rasterio.windows
from rasterio.features import rasterize

logger = logging.getLogger(__name__)

# Shifts larger than this are not searched: a bigger offset means the
# footprints belong to another place, which no translation should "fix".
DEFAULT_MAX_SHIFT_M = 6.0
DEFAULT_WINDOW_PX = 1024
DEFAULT_MAX_WINDOWS = 16
MIN_BUILDINGS_PER_WINDOW = 5
# Independent windows that must have measured a shift. Two can agree by
# chance in a sparse rural area; three cannot easily.
MIN_WINDOWS = 3
# Windows must agree within this many metres (median absolute deviation),
# and within this fraction of the shift itself, so a small shift cannot
# hide inside a large disagreement.
MAX_SPREAD_M = 1.0
MAX_SPREAD_FRACTION = 0.5
# The shifted fit must beat the unshifted fit by at least this ratio.
MIN_GAIN = 1.1
# A shift under 2 pixels is registration noise, not misalignment: moving
# footprints by it would as likely make them worse. Must also be >= 0.1 m.
MIN_SHIFT_PX = 2
MIN_SHIFT_M = 0.1


@dataclass
class FootprintShift:
    """Result of :func:`estimate_footprint_shift`.

    ``dx_px``/``dy_px`` are raster columns/rows (``dy_px`` > 0 = south on a
    north-up image). ``dx``/``dy`` are the same shift in the raster's CRS
    units, ready for ``GeoSeries.translate``.
    """

    dx_px: float = 0.0
    dy_px: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dx_m: float = 0.0
    dy_m: float = 0.0
    windows_used: int = 0
    spread_m: float = 0.0
    gain: float = 1.0
    applied: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _metres_per_pixel_xy(raster) -> tuple:
    """Ground size of one pixel (east-west, north-south) in metres. In a
    geographic CRS a degree of longitude shrinks with latitude, so the two
    differ."""
    sx, sy = abs(raster.transform.a), abs(raster.transform.e)
    if raster.crs is not None and raster.crs.is_geographic:
        lat = (raster.bounds.top + raster.bounds.bottom) / 2.0
        return sx * 111_320.0 * math.cos(math.radians(lat)), sy * 110_574.0
    return sx, sy


def _metres_per_pixel(raster) -> float:
    """The smaller of the two ground pixel sizes, in metres."""
    return min(_metres_per_pixel_xy(raster))


def _edge_strength(rgb: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the grey image, as float32."""
    grey = rgb.astype(np.float32).mean(axis=0)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


def _gaussian_fft(shape, sigma: float) -> np.ndarray:
    """Frequency response of a Gaussian blur, for rfft2-shaped spectra."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    return np.exp(-2.0 * (math.pi * sigma) ** 2 * (fx**2 + fy**2)).astype(
        np.float32
    )


def _best_window_shift(
    edges: np.ndarray, outlines: np.ndarray, max_px: int, sigma: float
) -> Optional[tuple]:
    """(dx, dy, gain) that best lays ``outlines`` onto ``edges``.

    Cross-correlation by FFT: ``corr[k] = sum(edges[x] * outlines[x - k])``,
    so the peak ``k`` is how far the outlines must move.
    """
    if outlines.sum() == 0 or not np.isfinite(edges).all():
        return None
    e = edges - edges.mean()
    f_e = np.fft.rfft2(e)
    f_o = np.fft.rfft2(outlines)
    corr = np.fft.irfft2(f_e * np.conj(f_o) * _gaussian_fft(e.shape, sigma), s=e.shape)
    # Wrap-around indexing: keep only shifts within +-max_px.
    rows = np.r_[0 : max_px + 1, -max_px:0]
    cols = np.r_[0 : max_px + 1, -max_px:0]
    sub = corr[np.ix_(rows % e.shape[0], cols % e.shape[1])]
    i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    dy, dx = int(rows[i]), int(cols[j])
    zero = float(corr[0, 0])
    peak = float(sub[i, j])
    if peak <= 0:
        return None
    gain = peak / zero if zero > 0 else float("inf")
    return dx, dy, gain


def _sample_windows(footprints, raster, window_px: int, max_windows: int):
    """Up to ``max_windows`` non-overlapping windows centred on buildings,
    spread over the layer (deterministic: every n-th building)."""
    points = footprints.geometry.representative_point()
    order = np.argsort(points.x.values * 1e6 + points.y.values)
    step = max(1, len(order) // (max_windows * 4))
    taken = []
    for idx in order[::step]:
        p = points.iloc[idx]
        row, col = raster.index(p.x, p.y)
        r0 = int(min(max(row - window_px // 2, 0), max(raster.height - window_px, 0)))
        c0 = int(min(max(col - window_px // 2, 0), max(raster.width - window_px, 0)))
        if any(abs(r0 - r) < window_px and abs(c0 - c) < window_px for r, c in taken):
            continue
        taken.append((r0, c0))
        if len(taken) >= max_windows:
            break
    return [
        rasterio.windows.Window(
            c0, r0, min(window_px, raster.width - c0), min(window_px, raster.height - r0)
        )
        for r0, c0 in taken
    ]


def estimate_footprint_shift(
    raster_path: str,
    footprints,
    *,
    max_shift_m: float = DEFAULT_MAX_SHIFT_M,
    window_px: int = DEFAULT_WINDOW_PX,
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> FootprintShift:
    """Measure the translation that puts ``footprints`` on the roofs of
    ``raster_path``. ``footprints`` is a GeoDataFrame in any CRS."""
    result = FootprintShift()
    if footprints is None or len(footprints) == 0:
        result.reason = "no footprints"
        return result
    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            result.reason = "raster has no CRS"
            return result
        fp = footprints.to_crs(raster.crs) if footprints.crs != raster.crs else footprints
        fp = fp[~fp.geometry.is_empty & fp.geometry.notna()]
        m_per_px = _metres_per_pixel(raster)
        mx, my = _metres_per_pixel_xy(raster)
        max_px = max(1, min(int(math.ceil(max_shift_m / m_per_px)), window_px // 4))
        sigma = max(1.0, 0.25 / m_per_px)  # ~25 cm of blur, at least 1 px
        bands = [1, 2, 3] if raster.count >= 3 else [1]
        sindex = fp.sindex

        shifts, gains = [], []
        for window in _sample_windows(fp, raster, window_px, max_windows):
            w_bounds = rasterio.windows.bounds(window, raster.transform)
            hits = fp.iloc[sindex.query(_box(*w_bounds), predicate="intersects")]
            if len(hits) < MIN_BUILDINGS_PER_WINDOW:
                continue
            w_transform = rasterio.windows.transform(window, raster.transform)
            shape = (int(window.height), int(window.width))
            outlines = rasterize(
                ((g.boundary, 1) for g in hits.geometry),
                out_shape=shape,
                transform=w_transform,
                all_touched=True,
                dtype="uint8",
            ).astype(np.float32)
            rgb = raster.read(bands, window=window)
            if raster.nodata is not None:
                valid = ~np.all(rgb == raster.nodata, axis=0)
            else:
                valid = ~np.all(rgb == 0, axis=0)
            if valid.mean() < 0.5:
                continue
            edges = _edge_strength(rgb) * valid
            best = _best_window_shift(edges, outlines, max_px, sigma)
            if best is None:
                continue
            shifts.append(best[:2])
            gains.append(best[2])

    result.windows_used = len(shifts)
    if len(shifts) < MIN_WINDOWS:
        result.reason = f"only {len(shifts)} usable window(s); need {MIN_WINDOWS}"
        return result
    arr = np.array(shifts, dtype=np.float64)
    dx_px, dy_px = np.median(arr, axis=0)
    spread_px = float(np.median(np.abs(arr - [dx_px, dy_px]).max(axis=1)))
    result.dx_px, result.dy_px = float(dx_px), float(dy_px)
    result.dx_m, result.dy_m = float(dx_px * mx), float(-dy_px * my)
    result.spread_m = spread_px * m_per_px
    result.gain = float(np.median(gains))
    # Raster pixel shift -> CRS units (transform.e is negative on north-up).
    with rasterio.open(raster_path) as raster:
        result.dx = float(dx_px * raster.transform.a)
        result.dy = float(dy_px * raster.transform.e)

    shift_m = math.hypot(result.dx_m, result.dy_m)
    if max(abs(dx_px), abs(dy_px)) < MIN_SHIFT_PX or shift_m < MIN_SHIFT_M:
        result.reason = "footprints already aligned (offset under 2 px)"
    elif result.spread_m > min(MAX_SPREAD_M, MAX_SPREAD_FRACTION * shift_m):
        result.reason = (
            f"windows disagree (spread {result.spread_m:.2f} m for a "
            f"{shift_m:.2f} m shift)"
        )
    elif result.gain < MIN_GAIN:
        result.reason = f"fit gain {result.gain:.3f} below {MIN_GAIN}"
    else:
        result.applied = True
        result.reason = "aligned"
    return result


def _box(minx, miny, maxx, maxy):
    from shapely.geometry import box

    return box(minx, miny, maxx, maxy)


def apply_footprint_shift(footprints, shift: FootprintShift, raster_crs):
    """Translate ``footprints`` by ``shift`` (in ``raster_crs`` units) and
    return them in their original CRS. Unchanged if not ``shift.applied``."""
    if not shift.applied:
        return footprints
    original_crs = footprints.crs
    moved = footprints.to_crs(raster_crs) if original_crs != raster_crs else footprints.copy()
    moved = moved.set_geometry(moved.geometry.translate(xoff=shift.dx, yoff=shift.dy))
    return moved.to_crs(original_crs) if original_crs != raster_crs else moved


def align_footprints_to_raster(raster_path: str, footprints, **kwargs):
    """Estimate and apply the shift. Returns ``(aligned_gdf, FootprintShift)``."""
    shift = estimate_footprint_shift(raster_path, footprints, **kwargs)
    with rasterio.open(raster_path) as raster:
        crs = raster.crs
    aligned = apply_footprint_shift(footprints, shift, crs)
    logger.info(
        "Footprint alignment: %s (dx=%.2f m east, dy=%.2f m north, %d windows, "
        "spread %.2f m, gain %.3f)",
        shift.reason, shift.dx_m, shift.dy_m, shift.windows_used,
        shift.spread_m, shift.gain,
    )
    return aligned, shift


def align_footprints_file(raster_path: str, gpkg_path: str, **kwargs) -> FootprintShift:
    """Align a footprints GeoPackage in place (rewritten only if shifted)."""
    import geopandas as gpd

    gdf = gpd.read_file(gpkg_path)
    aligned, shift = align_footprints_to_raster(raster_path, gdf, **kwargs)
    if shift.applied:
        aligned.to_file(gpkg_path, driver="GPKG")
    return shift
