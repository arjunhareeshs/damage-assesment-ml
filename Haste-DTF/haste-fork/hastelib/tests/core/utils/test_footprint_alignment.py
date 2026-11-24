# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from hastegeo.core.utils.footprint_alignment import (
    _best_window_shift,
    align_footprints_file,
    align_footprints_local,
    align_footprints_to_raster,
)

PX = 0.1  # metres per pixel (projected CRS)


def _scene(path, roofs_px):
    """A 1200x1200 dark image with bright rectangular roofs (UTM 17N)."""
    img = np.full((3, 1200, 1200), 40, np.uint8)
    for r0, c0, r1, c1 in roofs_px:
        img[:, r0:r1, c0:c1] = 200
    with rasterio.open(
        path, "w", driver="GTiff", width=1200, height=1200, count=3,
        dtype="uint8", crs="EPSG:32617", transform=from_origin(500000, 3000000, PX, PX),
    ) as dst:
        dst.write(img)


def _roofs():
    """A 6x6 grid of 80x120 px roofs."""
    return [
        (60 + 180 * i, 60 + 180 * j, 140 + 180 * i, 180 + 180 * j)
        for i in range(6)
        for j in range(6)
    ]


def _footprints(roofs_px, dx_m=0.0, dy_m=0.0):
    """Roof outlines in map units, displaced by (dx_m east, dy_m north)."""
    geoms = [
        box(
            500000 + c0 * PX + dx_m, 3000000 - r1 * PX + dy_m,
            500000 + c1 * PX + dx_m, 3000000 - r0 * PX + dy_m,
        )
        for r0, c0, r1, c1 in roofs_px
    ]
    return gpd.GeoDataFrame({"id": [str(i) for i in range(len(geoms))]},
                            geometry=geoms, crs="EPSG:32617")


class TestFootprintAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tif = os.path.join(self.tmp.name, "scene.tif")
        _scene(self.tif, _roofs())

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovers_known_offset(self):
        # Footprints drawn 1.5 m east and 2.0 m north of the roofs.
        fp = _footprints(_roofs(), dx_m=1.5, dy_m=2.0)
        aligned, shift = align_footprints_to_raster(self.tif, fp, window_px=600)
        self.assertTrue(shift.applied, shift.reason)
        self.assertAlmostEqual(shift.dx_m, -1.5, delta=PX)
        self.assertAlmostEqual(shift.dy_m, -2.0, delta=PX)
        truth = _footprints(_roofs())
        err = aligned.geometry.symmetric_difference(truth.geometry).area.sum()
        self.assertLess(err, 0.05 * truth.area.sum())
        # Attributes and CRS are kept.
        self.assertEqual(list(aligned["id"]), list(fp["id"]))
        self.assertEqual(aligned.crs, fp.crs)

    def test_aligned_footprints_are_left_alone(self):
        fp = _footprints(_roofs())
        aligned, shift = align_footprints_to_raster(self.tif, fp, window_px=600)
        self.assertFalse(shift.applied)
        self.assertTrue(aligned.geometry.geom_equals(fp.geometry).all())

    def test_works_across_crs(self):
        fp = _footprints(_roofs(), dx_m=-1.0, dy_m=1.0).to_crs(4326)
        aligned, shift = align_footprints_to_raster(self.tif, fp, window_px=600)
        self.assertTrue(shift.applied, shift.reason)
        self.assertEqual(aligned.crs, fp.crs)
        self.assertAlmostEqual(shift.dx_m, 1.0, delta=PX)
        self.assertAlmostEqual(shift.dy_m, -1.0, delta=PX)

    def test_sub_threshold_offset_is_treated_as_noise(self):
        # 0.1 m: under the 0.25 m floor, so nothing is moved.
        fp = _footprints(_roofs(), dx_m=PX, dy_m=0.0)
        aligned, shift = align_footprints_to_raster(self.tif, fp, window_px=600)
        self.assertFalse(shift.applied, shift.reason)
        self.assertTrue(aligned.geometry.geom_equals(fp.geometry).all())

    def test_too_few_windows_is_not_trusted(self):
        # One window covering the whole scene: a single measurement can be
        # a fluke, so even a real offset is not applied on its own.
        fp = _footprints(_roofs(), dx_m=1.5, dy_m=2.0)
        aligned, shift = align_footprints_to_raster(self.tif, fp, window_px=1200)
        self.assertFalse(shift.applied)
        self.assertIn("window", shift.reason)
        self.assertTrue(aligned.geometry.geom_equals(fp.geometry).all())

    def test_no_buildings_is_a_no_op(self):
        fp = _footprints([])
        aligned, shift = align_footprints_to_raster(self.tif, fp)
        self.assertFalse(shift.applied)
        self.assertEqual(len(aligned), 0)


def _scene_big(path, roofs_px, size=2400):
    img = np.full((3, size, size), 40, np.uint8)
    for r0, c0, r1, c1 in roofs_px:
        img[:, r0:r1, c0:c1] = 200
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3,
        dtype="uint8", crs="EPSG:32617", transform=from_origin(500000, 3000000, PX, PX),
    ) as dst:
        dst.write(img)


def _roofs_big(size=2400):
    return [
        (40 + 120 * i, 40 + 120 * j, 100 + 120 * i, 120 + 120 * j)
        for i in range(size // 120)
        for j in range(size // 120)
    ]


class TestSubPixelPeak(unittest.TestCase):
    def test_peak_is_refined_below_one_pixel(self):
        # Smooth edges with a 2.4 px true offset along x: the integer peak
        # is 2, the refined one must be closer to 2.4 than to 2.
        x = np.arange(256, dtype=np.float32)
        edges = np.tile(np.exp(-0.5 * ((x - 128.4) / 2.0) ** 2), (64, 1))
        outlines = np.zeros_like(edges)
        outlines[:, 126] = 1.0
        dx, dy, gain = _best_window_shift(edges, outlines, 8, 1.0)
        self.assertAlmostEqual(dx, 2.4, delta=0.15)
        self.assertAlmostEqual(dy, 0.0, delta=0.15)

    def test_peak_on_search_edge_is_rejected(self):
        x = np.arange(256, dtype=np.float32)
        edges = np.tile(np.exp(-0.5 * ((x - 150) / 2.0) ** 2), (64, 1))
        outlines = np.zeros_like(edges)
        outlines[:, 100] = 1.0  # true shift 50 px, search only +-8
        self.assertIsNone(_best_window_shift(edges, outlines, 8, 1.0))


class TestLocalAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tif = os.path.join(self.tmp.name, "big.tif")
        _scene_big(self.tif, _roofs_big())

    def tearDown(self):
        self.tmp.cleanup()

    def _offset_field(self, west_dx, east_dx):
        """Footprints of the west half shifted by west_dx m, east half by east_dx m."""
        roofs = _roofs_big()
        geoms = []
        for r0, c0, r1, c1 in roofs:
            d = west_dx if c0 < 1200 else east_dx
            geoms.append(box(500000 + c0 * PX + d, 3000000 - r1 * PX,
                             500000 + c1 * PX + d, 3000000 - r0 * PX))
        return gpd.GeoDataFrame({"id": [str(i) for i in range(len(geoms))]},
                                geometry=geoms, crs="EPSG:32617")

    def test_spatially_varying_offset_is_followed(self):
        # West half 1.5 m east of the roofs, east half 1.5 m west: no single
        # translation fits; the local field corrects both halves.
        fp = self._offset_field(1.5, -1.5)
        aligned, report = align_footprints_local(self.tif, fp, window_px=400)
        self.assertTrue(report.applied, report.reason)
        self.assertGreater(report.fit_after, report.fit_before)
        truth = self._offset_field(0.0, 0.0)
        err = aligned.geometry.symmetric_difference(truth.geometry).area.sum()
        before = fp.geometry.symmetric_difference(truth.geometry).area.sum()
        self.assertLess(err, 0.25 * before)
        self.assertEqual(list(aligned["id"]), list(fp["id"]))
        self.assertEqual(aligned.crs, fp.crs)

    def test_global_estimate_refuses_the_same_field(self):
        fp = self._offset_field(1.5, -1.5)
        _, shift = align_footprints_to_raster(self.tif, fp, window_px=400)
        self.assertFalse(shift.applied)

    def test_aligned_footprints_stay_put(self):
        fp = self._offset_field(0.0, 0.0)
        aligned, report = align_footprints_local(self.tif, fp, window_px=400)
        self.assertFalse(report.applied, report.reason)
        self.assertTrue(aligned.geometry.geom_equals(fp.geometry).all())

    def test_sources_are_registered_separately(self):
        # Same buildings, two datasets with opposite offsets, interleaved.
        roofs = _roofs_big()
        geoms, src = [], []
        for k, (r0, c0, r1, c1) in enumerate(roofs):
            d = 1.5 if k % 2 else -1.5
            geoms.append(box(500000 + c0 * PX + d, 3000000 - r1 * PX,
                             500000 + c1 * PX + d, 3000000 - r0 * PX))
            src.append("A" if k % 2 else "B")
        fp = gpd.GeoDataFrame({"id": [str(i) for i in range(len(geoms))],
                               "source_dataset": src}, geometry=geoms, crs="EPSG:32617")
        aligned, report = align_footprints_local(self.tif, fp, window_px=400)
        self.assertTrue(report.applied, report.reason)
        self.assertIn("A", report.groups)
        self.assertIn("B", report.groups)
        truth = self._offset_field(0.0, 0.0)
        err = aligned.geometry.symmetric_difference(truth.geometry).area.sum()
        before = fp.geometry.symmetric_difference(truth.geometry).area.sum()
        self.assertLess(err, 0.25 * before)

    def test_file_mode_writes_report_and_keeps_original(self):
        fp = self._offset_field(1.5, -1.5)
        gpkg = os.path.join(self.tmp.name, "fp.gpkg")
        orig = os.path.join(self.tmp.name, "fp_unaligned.gpkg")
        fp.to_file(gpkg, driver="GPKG")
        report = align_footprints_file(self.tif, gpkg, keep_original_path=orig, window_px=400)
        self.assertTrue(report.applied, report.reason)
        self.assertTrue(os.path.exists(orig))
        d = report.as_dict()
        for key in ("fit_before", "fit_after", "buildings_shifted", "crs_transform"):
            self.assertIn(key, d)

    def test_measure_only(self):
        fp = self._offset_field(1.5, -1.5)
        gpkg = os.path.join(self.tmp.name, "fp.gpkg")
        fp.to_file(gpkg, driver="GPKG")
        report = align_footprints_file(self.tif, gpkg, apply=False, window_px=400)
        self.assertFalse(report.applied)
        self.assertIn("not applied", report.reason)
        again = gpd.read_file(gpkg)
        self.assertTrue(again.geometry.geom_equals(fp.geometry).all())


class TestRasterBands(unittest.TestCase):
    def _write(self, path, count, colorinterp=None, alpha=False, nodata=None):
        arr = np.full((count, 20, 20), 100, np.uint8)
        arr[:, :, :5] = 0
        with rasterio.open(
            path, "w", driver="GTiff", width=20, height=20, count=count, dtype="uint8",
            crs="EPSG:32617", transform=from_origin(0, 20, 1, 1), nodata=nodata,
        ) as dst:
            if alpha:
                arr[count - 1] = 255
                arr[count - 1, :, 15:] = 0  # transparent strip, RGB not zero
                arr[:count - 1, :, 15:] = 77
            dst.write(arr)
            if colorinterp:
                dst.colorinterp = colorinterp

    def test_bgr_file_is_read_as_rgb(self):
        from rasterio.enums import ColorInterp
        from hastegeo.core.utils.raster_bands import rgb_band_indexes

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bgrn.tif")
            self._write(p, 4, [ColorInterp.blue, ColorInterp.green, ColorInterp.red,
                               ColorInterp.undefined])
            with rasterio.open(p) as r:
                self.assertEqual(rgb_band_indexes(r), [3, 2, 1])

    def test_alpha_marks_transparent_pixels_invalid(self):
        from rasterio.enums import ColorInterp
        from rasterio.windows import Window
        from hastegeo.core.utils.raster_bands import valid_pixel_mask

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rgba.tif")
            self._write(p, 4, [ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                               ColorInterp.alpha], alpha=True)
            with rasterio.open(p) as r:
                w = Window(0, 0, 20, 20)
                m = valid_pixel_mask(r, w, r.read([1, 2, 3], window=w))
            self.assertFalse(m[:, 15:].any())  # alpha 0, RGB 77: still invalid
            self.assertTrue(m[:, 5:15].all())

    def test_no_declared_mask_uses_black_collar(self):
        from rasterio.windows import Window
        from hastegeo.core.utils.raster_bands import valid_pixel_mask

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rgb.tif")
            self._write(p, 3)
            with rasterio.open(p) as r:
                w = Window(0, 0, 20, 20)
                m = valid_pixel_mask(r, w, r.read([1, 2, 3], window=w))
            self.assertFalse(m[:, :5].any())
            self.assertTrue(m[:, 5:].all())


class TestMosaicNoSeam(unittest.TestCase):
    def test_black_collar_does_not_cover_neighbour(self):
        from hastegeo.core.utils.imagery import ImageryUtils

        with tempfile.TemporaryDirectory() as d:
            def write(name, x0, arr):
                with rasterio.open(
                    os.path.join(d, name), "w", driver="GTiff", width=arr.shape[2],
                    height=arr.shape[1], count=3, dtype="uint8", crs="EPSG:32617",
                    transform=from_origin(x0, 1000, 1, 1),
                ) as dst:
                    dst.write(arr)

            a = np.full((3, 50, 200), 100, np.uint8)
            b = np.full((3, 50, 200), 200, np.uint8)
            b[:, :, :60] = 0  # black collar overlapping tile a (x 150..200)
            write("a.tif", 0, a)
            write("b.tif", 150, b)
            out = ImageryUtils.mosaic_imagery(
                [os.path.join(d, "a.tif"), os.path.join(d, "b.tif")],
                os.path.join(d, "mosaic.tif"),
                "-of COG",
            )
            with rasterio.open(out) as m:
                row = m.read(1)[25]
                self.assertEqual(m.nodata, 0)
            # Tile a's pixels show through b's collar: no black seam.
            self.assertTrue((row[150:200] == 100).all())
            self.assertTrue((row[210:350] == 200).all())


if __name__ == "__main__":
    unittest.main()
