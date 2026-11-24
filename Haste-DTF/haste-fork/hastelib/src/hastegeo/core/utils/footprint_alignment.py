# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Register building footprints to the imagery they will be scored on.

Footprint datasets (Overture, Microsoft, county files) are rarely drawn on
the same orthophoto that is being assessed, so they are offset from the roofs
by one to a few metres. Every footprint then crops and masks the wrong pixels:
part of the roof is cut off and part of the yard is scored.

The offset is measured from the imagery itself: footprint outlines are
rasterised and slid over the image's edge strength (gradient magnitude); the
shift where outlines best follow roof edges wins (FFT cross-correlation, peak
refined to sub-pixel precision).

Two estimates are made:

* **Global** (:func:`estimate_footprint_shift`): one translation for the whole
  layer, from the median of windows spread over the image.
* **Local** (:func:`align_footprints_file`, the pipeline default): a grid of
  tie-point windows, each with its own measured shift. Every building takes
  the median shift of its nearest tie points. Aerial orthophotos are not
  true-orthos (roofs lean away from the camera by an amount that changes over
  the image) and footprint datasets are traced from imagery of different
  dates, so the offset changes within a layer; measured on NOAA imagery it
  went from ~1 m west to ~3 m east between neighbouring 1 km tiles.

Footprints from different datasets (Overture's ``source_dataset``: OSM,
Microsoft, Google, ...) are registered separately, since each has its own
offset. Nothing is moved unless enough tie points agree, and the result is
checked: the outline-to-edge fit is measured before and after, and the
footprints are left unchanged if the shifted fit is not better.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

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
# Below this the offset is registration noise. The peak is located to
# sub-pixel precision, so one pixel is resolvable; 0.25 m is the floor on
# very fine imagery (a quarter of a typical wall thickness + eave).
MIN_SHIFT_PX = 1.0
MIN_SHIFT_M = 0.25

# Local (tie-point) alignment.
LOCAL_MAX_TIES = 400  # cap on tie-point windows per footprint group
LOCAL_NEIGHBOURS = 5  # tie points consulted per building
LOCAL_MIN_TIES = 3  # ... of which at least this many must be in range
LOCAL_RADIUS_WINDOWS = 2.5  # range, in window sizes
MIN_GROUP_BUILDINGS = 50  # smaller source groups use the all-sources field
SOURCE_DATASET_COLUMN = "source_dataset"


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


@dataclass
class TiePoint:
    """One window's measured shift, located at the window centre (raster CRS)."""

    x: float
    y: float
    dx_px: float
    dy_px: float
    gain: float
    buildings: int


@dataclass
class FootprintAlignment:
    """Result of :func:`align_footprints_file` / :func:`align_footprints_local`."""

    applied: bool = False
    reason: str = ""
    mode: str = "local"
    buildings: int = 0
    buildings_shifted: int = 0
    tie_points: int = 0
    shift_m_median: float = 0.0
    shift_m_p90: float = 0.0
    shift_m_max: float = 0.0
    fit_before: float = 0.0
    fit_after: float = 0.0
    groups: Dict[str, dict] = field(default_factory=dict)
    crs_transform: str = ""
    crs_transform_accuracy_m: Optional[float] = None
    global_shift: dict = field(default_factory=dict)

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


def _parabolic_offset(c_minus: float, c_0: float, c_plus: float) -> float:
    """Sub-pixel position of a peak from three samples (-0.5 .. 0.5)."""
    den = c_minus - 2.0 * c_0 + c_plus
    if den >= 0:  # flat or not a maximum
        return 0.0
    return float(np.clip(0.5 * (c_minus - c_plus) / den, -0.5, 0.5))


def _best_window_shift(
    edges: np.ndarray, outlines: np.ndarray, max_px: int, sigma: float
) -> Optional[tuple]:
    """(dx, dy, gain) that best lays ``outlines`` onto ``edges``.

    Cross-correlation by FFT: ``corr[k] = sum(edges[x] * outlines[x - k])``,
    so the peak ``k`` is how far the outlines must move. The peak is refined
    to sub-pixel precision with a parabola through its neighbours. A peak on
    the edge of the search range is rejected: the real optimum lies outside
    it (or there is none), so the value would be an artefact of the limit.
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
    if max_px > 1 and (abs(dx) >= max_px or abs(dy) >= max_px):
        return None
    zero = float(corr[0, 0])
    peak = float(sub[i, j])
    if peak <= 0:
        return None
    h, w = e.shape
    oy = _parabolic_offset(
        corr[(dy - 1) % h, dx % w], peak, corr[(dy + 1) % h, dx % w]
    )
    ox = _parabolic_offset(
        corr[dy % h, (dx - 1) % w], peak, corr[dy % h, (dx + 1) % w]
    )
    gain = peak / zero if zero > 0 else float("inf")
    return dx + ox, dy + oy, gain


def _valid_pixels(raster, window, rgb: np.ndarray) -> np.ndarray:
    """Pixels holding image data in ``window``: the file's nodata / alpha /
    internal mask when it declares one, else not all-zero RGB."""
    from .raster_bands import valid_pixel_mask

    return valid_pixel_mask(raster, window, rgb)


def _box(minx, miny, maxx, maxy):
    from shapely.geometry import box

    return box(minx, miny, maxx, maxy)


class _Scene:
    """One open raster plus the per-window measurement shared by the global
    and local estimators."""

    def __init__(self, raster, max_shift_m: float, window_px: int):
        self.raster = raster
        self.window_px = window_px
        self.m_per_px = _metres_per_pixel(raster)
        self.mx, self.my = _metres_per_pixel_xy(raster)
        self.max_px = max(
            1, min(int(math.ceil(max_shift_m / self.m_per_px)), window_px // 4)
        )
        self.sigma = max(1.0, 0.25 / self.m_per_px)  # ~25 cm blur, >= 1 px
        if raster.count >= 3:
            from .raster_bands import rgb_band_indexes

            self.bands = rgb_band_indexes(raster)
        else:
            self.bands = [1]

    def window_data(self, window):
        """(edges, valid) for ``window``, or None when mostly no-data."""
        rgb = self.raster.read(self.bands, window=window)
        valid = _valid_pixels(self.raster, window, rgb)
        if valid.mean() < 0.5:
            return None
        return _edge_strength(rgb) * valid, valid

    def outlines(self, geoms, window) -> np.ndarray:
        shape = (int(window.height), int(window.width))
        return rasterize(
            ((g.boundary, 1) for g in geoms),
            out_shape=shape,
            transform=rasterio.windows.transform(window, self.raster.transform),
            all_touched=True,
            dtype="uint8",
        ).astype(np.float32)

    def measure(self, fp, sindex, window):
        """(dx_px, dy_px, gain, n_buildings) for one window, or None."""
        w_bounds = rasterio.windows.bounds(window, self.raster.transform)
        hits = fp.iloc[sindex.query(_box(*w_bounds), predicate="intersects")]
        if len(hits) < MIN_BUILDINGS_PER_WINDOW:
            return None
        data = self.window_data(window)
        if data is None:
            return None
        edges, _ = data
        best = _best_window_shift(
            edges, self.outlines(hits.geometry, window), self.max_px, self.sigma
        )
        if best is None:
            return None
        return best[0], best[1], best[2], len(hits)

    def shift_m(self, dx_px, dy_px):
        return dx_px * self.mx, -dy_px * self.my


def _sample_windows(footprints, raster, window_px: int, max_windows: int):
    """Up to ``max_windows`` non-overlapping windows spread evenly over the
    whole layer (grid cells holding buildings, thinned evenly).

    The previous version walked the buildings in west-to-east order and
    stopped at ``max_windows``, so on a large layer every window came from
    its western part and the "global" shift described one corner.
    """
    return _grid_windows(footprints, raster, window_px, max_windows)


def _grid_windows(footprints, raster, window_px: int, max_windows: int):
    """Tie-point windows: the raster grid cells of ``window_px`` that hold
    footprints, thinned evenly to at most ``max_windows``."""
    pts = footprints.geometry.representative_point()
    rows, cols = rasterio.transform.rowcol(raster.transform, pts.x.values, pts.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < raster.height) & (cols >= 0) & (cols < raster.width)
    cells, counts = np.unique(
        np.stack([rows[inside] // window_px, cols[inside] // window_px], axis=1),
        axis=0, return_counts=True,
    ) if inside.any() else (np.empty((0, 2), int), np.empty(0, int))
    keep = counts >= MIN_BUILDINGS_PER_WINDOW
    cells = cells[keep]
    if len(cells) > max_windows:
        cells = cells[np.linspace(0, len(cells) - 1, max_windows).round().astype(int)]
    out = []
    for r, c in cells:
        r0, c0 = int(r) * window_px, int(c) * window_px
        out.append(rasterio.windows.Window(
            c0, r0, min(window_px, raster.width - c0), min(window_px, raster.height - r0)
        ))
    return out


def _decide(shifts_px: np.ndarray, gains, scene: _Scene, result: FootprintShift):
    """Median shift of the windows plus the agreement / size / gain gates."""
    dx_px, dy_px = np.median(shifts_px, axis=0)
    spread_px = float(np.median(np.abs(shifts_px - [dx_px, dy_px]).max(axis=1)))
    result.dx_px, result.dy_px = float(dx_px), float(dy_px)
    result.dx_m, result.dy_m = scene.shift_m(dx_px, dy_px)
    result.spread_m = spread_px * scene.m_per_px
    result.gain = float(np.median(gains))
    # Raster pixel shift -> CRS units (transform.e is negative on north-up).
    result.dx = float(dx_px * scene.raster.transform.a)
    result.dy = float(dy_px * scene.raster.transform.e)
    shift_m = math.hypot(result.dx_m, result.dy_m)
    if max(abs(dx_px), abs(dy_px)) < MIN_SHIFT_PX or shift_m < MIN_SHIFT_M:
        result.reason = (
            f"footprints already aligned (offset {shift_m:.2f} m, under "
            f"max({MIN_SHIFT_PX:g} px, {MIN_SHIFT_M} m))"
        )
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


def estimate_footprint_shift(
    raster_path: str,
    footprints,
    *,
    max_shift_m: float = DEFAULT_MAX_SHIFT_M,
    window_px: int = DEFAULT_WINDOW_PX,
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> FootprintShift:
    """Measure the single translation that puts ``footprints`` on the roofs
    of ``raster_path``. ``footprints`` is a GeoDataFrame in any CRS."""
    result = FootprintShift()
    if footprints is None or len(footprints) == 0:
        result.reason = "no footprints"
        return result
    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            result.reason = "raster has no CRS"
            return result
        fp = _in_raster_crs(footprints, raster.crs)
        scene = _Scene(raster, max_shift_m, window_px)
        sindex = fp.sindex
        shifts, gains = [], []
        for window in _sample_windows(fp, raster, window_px, max_windows):
            m = scene.measure(fp, sindex, window)
            if m is not None:
                shifts.append(m[:2])
                gains.append(m[2])
        result.windows_used = len(shifts)
        if len(shifts) < MIN_WINDOWS:
            result.reason = f"only {len(shifts)} usable window(s); need {MIN_WINDOWS}"
            return result
        return _decide(np.array(shifts, dtype=np.float64), gains, scene, result)


def _in_raster_crs(footprints, crs):
    fp = footprints.to_crs(crs) if footprints.crs != crs else footprints
    return fp[fp.geometry.notna() & ~fp.geometry.is_empty]


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
    """Estimate and apply one global shift. Returns ``(aligned_gdf, FootprintShift)``."""
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


# ---------------------------------------------------------------------------
# Local (tie-point) alignment
# ---------------------------------------------------------------------------


def _tie_points(scene: _Scene, fp, max_ties: int) -> List[TiePoint]:
    sindex = fp.sindex
    ties = []
    for window in _grid_windows(fp, scene.raster, scene.window_px, max_ties):
        m = scene.measure(fp, sindex, window)
        if m is None or m[2] < MIN_GAIN:
            continue
        cx, cy = rasterio.transform.xy(
            scene.raster.transform,
            window.row_off + window.height / 2.0,
            window.col_off + window.width / 2.0,
            offset="ul",
        )
        ties.append(TiePoint(float(cx), float(cy), m[0], m[1], m[2], m[3]))
    return ties


def _building_shifts_px(scene: _Scene, centres_xy: np.ndarray, ties: List[TiePoint]):
    """Per-building (dx_px, dy_px) from the median of its nearest tie points.

    Rows are NaN when too few tie points are in range or they disagree.
    """
    n = len(centres_xy)
    out = np.full((n, 2), np.nan)
    if not ties or n == 0:
        return out
    # Work in metres so a geographic CRS gets the right distances.
    to_m = np.array([
        scene.mx / abs(scene.raster.transform.a),
        scene.my / abs(scene.raster.transform.e),
    ])
    tie_xy = np.array([[t.x, t.y] for t in ties]) * to_m
    tie_shift = np.array([[t.dx_px, t.dy_px] for t in ties])
    radius = LOCAL_RADIUS_WINDOWS * scene.window_px * scene.m_per_px
    k = min(LOCAL_NEIGHBOURS, len(ties))
    pts = centres_xy * to_m
    for s in range(0, n, 20_000):
        chunk = pts[s : s + 20_000]
        d = np.hypot(
            chunk[:, None, 0] - tie_xy[None, :, 0], chunk[:, None, 1] - tie_xy[None, :, 1]
        )
        nearest = np.argsort(d, axis=1)[:, :k]
        for i, idx in enumerate(nearest):
            idx = idx[d[i, idx] <= radius]
            if len(idx) < LOCAL_MIN_TIES:
                continue
            sh = tie_shift[idx]
            med = np.median(sh, axis=0)
            spread_m = float(np.median(np.abs(sh - med).max(axis=1))) * scene.m_per_px
            mag_m = math.hypot(*scene.shift_m(*med))
            if spread_m > min(MAX_SPREAD_M, max(MAX_SPREAD_FRACTION * mag_m, 0.5)):
                continue
            out[s + i] = med
    return out


def _outline_fit(scene: _Scene, geoms, windows) -> float:
    """Mean edge strength under the footprint outlines, normalised by the mean
    edge strength of the valid image (1.0 = no better than random)."""
    import geopandas as gpd

    gs = gpd.GeoSeries(list(geoms), crs=scene.raster.crs)
    sindex = gs.sindex
    num = den = 0.0
    for window in windows:
        data = scene.window_data(window)
        if data is None:
            continue
        edges, valid = data
        w_bounds = rasterio.windows.bounds(window, scene.raster.transform)
        hits = gs.iloc[sindex.query(_box(*w_bounds), predicate="intersects")]
        if len(hits) == 0:
            continue
        o = scene.outlines(hits, window) * valid
        mean_edge = float(edges[valid].mean()) if valid.any() else 0.0
        if o.sum() == 0 or mean_edge <= 0:
            continue
        num += float((edges * o).sum()) / mean_edge
        den += float(o.sum())
    return num / den if den else 0.0


def _crs_transform_info(src_crs, dst_crs):
    """(description, accuracy in metres or None) of the footprint -> raster
    CRS transformation PROJ will use. NAD83 <-> WGS84 is a null ("ballpark")
    transformation with ~1-2 m real error in the US; the image-based
    registration is what removes it."""
    try:
        from pyproj import Transformer

        if src_crs is None or dst_crs is None or src_crs == dst_crs:
            return "none (same CRS)", 0.0
        t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        acc = t.accuracy
        return t.description, (None if acc is None or acc < 0 else float(acc))
    except Exception as exc:  # pragma: no cover - informational only
        return f"unknown ({exc})", None


def align_footprints_local(
    raster_path: str,
    footprints,
    *,
    group_column: Optional[str] = SOURCE_DATASET_COLUMN,
    max_shift_m: float = DEFAULT_MAX_SHIFT_M,
    window_px: int = DEFAULT_WINDOW_PX,
    max_ties: int = LOCAL_MAX_TIES,
):
    """Shift every building by the locally measured offset.

    Returns ``(aligned_gdf, FootprintAlignment)``; the GeoDataFrame keeps its
    CRS, columns and row order. Buildings without enough agreeing tie points
    nearby take the global shift when that one is trusted, else stay put.
    """
    from shapely import affinity

    report = FootprintAlignment(mode="local", buildings=len(footprints))
    if footprints is None or len(footprints) == 0:
        report.reason = "no footprints"
        return footprints, report
    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            report.reason = "raster has no CRS"
            return footprints, report
        report.crs_transform, report.crs_transform_accuracy_m = _crs_transform_info(
            footprints.crs, raster.crs
        )
        scene = _Scene(raster, max_shift_m, window_px)
        fp_all = footprints.to_crs(raster.crs) if footprints.crs != raster.crs else footprints
        usable = fp_all.geometry.notna() & ~fp_all.geometry.is_empty
        centres = fp_all.geometry.representative_point()
        centres_xy = np.column_stack([centres.x.values, centres.y.values])

        # Global estimate: the fallback where tie points are sparse.
        glob = FootprintShift()
        gfp = fp_all[usable]
        g_shifts, g_gains = [], []
        for window in _sample_windows(gfp, raster, window_px, DEFAULT_MAX_WINDOWS):
            m = scene.measure(gfp, gfp.sindex, window)
            if m is not None:
                g_shifts.append(m[:2])
                g_gains.append(m[2])
        glob.windows_used = len(g_shifts)
        if len(g_shifts) >= MIN_WINDOWS:
            _decide(np.array(g_shifts), g_gains, scene, glob)
        else:
            glob.reason = f"only {len(g_shifts)} usable window(s); need {MIN_WINDOWS}"
        report.global_shift = glob.as_dict()

        # Tie points for all sources together, and per source dataset.
        ties_all = _tie_points(scene, gfp, max_ties)
        shifts_px = _building_shifts_px(scene, centres_xy, ties_all)
        report.tie_points = len(ties_all)
        report.groups["all"] = {"buildings": int(len(fp_all)), "tie_points": len(ties_all)}
        if group_column and group_column in fp_all.columns:
            labels = fp_all[group_column].fillna("unknown").astype(str)
            if labels.nunique() > 1:
                for name, idx in labels.groupby(labels).groups.items():
                    pos = fp_all.index.get_indexer(idx)
                    sel = usable.values[pos]
                    info = {"buildings": int(len(pos))}
                    if sel.sum() >= MIN_GROUP_BUILDINGS:
                        ties = _tie_points(scene, fp_all.iloc[pos[sel]], max_ties)
                        g = _building_shifts_px(scene, centres_xy[pos], ties)
                        has = ~np.isnan(g[:, 0])
                        shifts_px[pos[has]] = g[has]
                        info["tie_points"] = len(ties)
                        info["buildings_with_local_shift"] = int(has.sum())
                    else:
                        info["note"] = "too few buildings; uses the all-sources field"
                    report.groups[str(name)] = info

        # Fallback to the trusted global shift; then drop noise-level shifts.
        missing = np.isnan(shifts_px[:, 0])
        if glob.applied:
            shifts_px[missing] = [glob.dx_px, glob.dy_px]
        shifts_px = np.nan_to_num(shifts_px)
        dx_m, dy_m = scene.shift_m(shifts_px[:, 0], shifts_px[:, 1])
        mag_m = np.hypot(dx_m, dy_m)
        tiny = (np.abs(shifts_px).max(axis=1) < MIN_SHIFT_PX) | (mag_m < MIN_SHIFT_M)
        shifts_px[tiny] = 0.0
        mag_m[tiny] = 0.0
        moved = (mag_m > 0) & usable.values

        if not moved.any():
            report.reason = "no consistent offset found (footprints left unchanged)"
            if glob.reason:
                report.reason += f"; global: {glob.reason}"
            return footprints, report

        a, e = raster.transform.a, raster.transform.e
        new_geoms = [
            affinity.translate(g, xoff=sx * a, yoff=sy * e) if mv else g
            for g, (sx, sy), mv in zip(fp_all.geometry.values, shifts_px, moved)
        ]

        # Quality gate: outlines must follow roof edges better than before.
        qc_windows = _grid_windows(gfp, raster, window_px, 64)
        report.fit_before = round(_outline_fit(scene, fp_all.geometry.values[usable.values], qc_windows), 4)
        report.fit_after = round(
            _outline_fit(scene, np.asarray(new_geoms, dtype=object)[usable.values], qc_windows), 4
        )
        report.buildings_shifted = int(moved.sum())
        report.shift_m_median = round(float(np.median(mag_m[moved])), 3)
        report.shift_m_p90 = round(float(np.percentile(mag_m[moved], 90)), 3)
        report.shift_m_max = round(float(mag_m[moved].max()), 3)
        if report.fit_after <= report.fit_before:
            report.reason = (
                f"QC rejected: fit {report.fit_before:.3f} -> {report.fit_after:.3f} "
                "did not improve (footprints left unchanged)"
            )
            report.buildings_shifted = 0
            return footprints, report

    out = fp_all.set_geometry(new_geoms)
    if footprints.crs != out.crs:
        out = out.to_crs(footprints.crs)
    report.applied = True
    report.reason = (
        f"aligned {report.buildings_shifted}/{report.buildings} buildings "
        f"(median {report.shift_m_median} m); fit {report.fit_before:.3f} -> "
        f"{report.fit_after:.3f}"
    )
    return out, report


def align_footprints_file(
    raster_path: str,
    gpkg_path: str,
    *,
    apply: bool = True,
    keep_original_path: Optional[str] = None,
    mode: Optional[str] = None,
    **kwargs,
) -> FootprintAlignment:
    """Align a footprints GeoPackage in place (rewritten only if shifted).

    ``mode`` is ``"local"`` (default; ``HASTE_FOOTPRINT_ALIGN_MODE``) or
    ``"global"`` (one translation for the layer). ``apply=False`` measures
    and reports without moving anything. With ``keep_original_path``, the
    unshifted file is copied there before it is rewritten.
    """
    import os
    import shutil

    import geopandas as gpd

    mode = (mode or os.getenv("HASTE_FOOTPRINT_ALIGN_MODE", "local")).lower()
    gdf = gpd.read_file(gpkg_path)
    if mode == "global":
        shift = estimate_footprint_shift(raster_path, gdf, **kwargs)
        report = FootprintAlignment(
            mode="global", buildings=len(gdf), applied=shift.applied,
            reason=shift.reason, global_shift=shift.as_dict(),
            buildings_shifted=len(gdf) if shift.applied else 0,
        )
        with rasterio.open(raster_path) as raster:
            crs = raster.crs
        aligned = apply_footprint_shift(gdf, shift, crs)
    else:
        aligned, report = align_footprints_local(raster_path, gdf, **kwargs)
    if report.applied and not apply:
        report.applied = False
        report.reason = f"measured ({report.reason}) but not applied"
    if report.applied:
        if keep_original_path:
            shutil.copy2(gpkg_path, keep_original_path)
        aligned.to_file(gpkg_path, driver="GPKG")
    logger.info("Footprint alignment (%s): %s", report.mode, report.reason)
    return report
