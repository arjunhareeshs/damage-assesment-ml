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

from hastegeo.core.utils.footprint_alignment import align_footprints_to_raster

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

    def test_sub_two_pixel_offset_is_treated_as_noise(self):
        # 1 px (0.1 m): within registration noise, so nothing is moved.
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
