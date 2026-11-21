# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Imagery updates: what a TIF covers, what it replaces, grid + registration."""

import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, shape

from hastegeo.core.utils import imagery_update as iu

X0, Y0, PX = 500000.0, 3000000.0, 0.5  # UTM 17N, 0.5 m pixels


def _scene(size=1600, seed=7):
    """A textured RGB 'town': random rectangles of random brightness."""
    rng = np.random.default_rng(seed)
    img = np.full((3, size, size), 60, np.uint8)
    for _ in range(400):
        r, c = rng.integers(0, size - 40, 2)
        h, w = rng.integers(10, 40, 2)
        img[:, r:r + h, c:c + w] = rng.integers(90, 250)
    return img


SCENE = _scene()


def _write(path, x_off_px, y_off_px, w, h, *, shift_m=(0.0, 0.0), alpha=False,
           collar=0, res=PX, crs="EPSG:32617", nodata=None):
    """Write the part of SCENE starting at (x_off_px, y_off_px), georeferenced
    where it truly is plus ``shift_m`` (east, north) of misregistration."""
    data = SCENE[:, y_off_px:y_off_px + h, x_off_px:x_off_px + w].copy()
    if collar:
        data[:, :, :collar] = 0
    bands = 4 if alpha else 3
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=bands, dtype="uint8",
        crs=crs, nodata=nodata,
        transform=from_origin(X0 + x_off_px * PX + shift_m[0], Y0 - y_off_px * PX + shift_m[1], res, res),
    ) as dst:
        dst.write(data, [1, 2, 3])
        if alpha:
            a = np.full((h, w), 255, np.uint8)
            a[:, :collar] = 0
            dst.write(a, 4)
            dst.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                               rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
    return path


class TestFootprint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_black_collar_is_not_part_of_the_footprint(self):
        p = _write(os.path.join(self.d, "a.tif"), 0, 0, 800, 800, collar=200)
        fp = iu.valid_data_footprint(p)
        full = iu.area_km2(shape(iu.describe_tif(p)["footprint"]))
        # 3/4 of the image holds data.
        with rasterio.open(p) as s:
            from rasterio.warp import transform_bounds
            rect = box(*transform_bounds(s.crs, "EPSG:4326", *s.bounds))
        self.assertAlmostEqual(iu.area_km2(fp) / iu.area_km2(rect), 0.75, delta=0.02)
        self.assertAlmostEqual(full, iu.area_km2(fp), places=6)

    def test_alpha_mask_defines_the_footprint(self):
        p = _write(os.path.join(self.d, "a.tif"), 0, 0, 800, 800, alpha=True, collar=400)
        info = iu.describe_tif(p)
        self.assertEqual(info["data_bands"], 3)
        self.assertAlmostEqual(info["area_km2"], 0.4 * 0.4 / 2, delta=0.004)


class TestPlan(unittest.TestCase):
    def test_retire_keep_and_extend(self):
        existing = [
            {"id": 0, "footprint": box(0, 0, 1, 1)},      # fully covered -> retire
            {"id": 1, "footprint": box(1, 0, 2, 1)},      # half covered -> keep under new
            {"id": 2, "footprint": box(5, 5, 6, 6)},      # untouched
        ]
        incoming = [{"id": "new0", "footprint": box(-0.01, -0.01, 1.5, 1.01)},
                    {"id": "new1", "footprint": box(10, 10, 11, 11)}]  # outside: extends
        plan = iu.plan_imagery_update(existing, incoming)
        actions = {e["id"]: e["action"] for e in plan["existing"]}
        self.assertEqual(actions, {0: "retire", 1: "keep_under_new", 2: "unchanged"})
        self.assertEqual(plan["retire_ids"], [0])
        self.assertEqual(plan["keep_ids"], [1, 2])
        new0, new1 = plan["incoming"]
        self.assertEqual([o["id"] for o in new0["corresponds_to"]], [0, 1])
        self.assertAlmostEqual(new0["corresponds_to"][0]["share_of_existing"], 1.0, places=3)
        # Only its thin 0.01-degree margin lies outside the existing tiles.
        self.assertGreater(new0["new_area_share"], 0.0)
        self.assertLess(new0["new_area_share"], 0.05)
        self.assertEqual(new1["corresponds_to"], [])
        self.assertEqual(new1["new_area_share"], 1.0)
        self.assertGreater(plan["updated_area_km2"], 0)


class TestGridAndRegistration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_warp_snaps_to_the_grid(self):
        ref = _write(os.path.join(self.d, "ref.tif"), 0, 0, 800, 800)
        grid = iu.reference_grid([ref])
        # Different pixel size and an origin off the grid lines.
        odd = _write(os.path.join(self.d, "odd.tif"), 300, 300, 400, 400,
                     shift_m=(0.13, -0.21), res=0.6, alpha=True)
        out = iu.warp_to_grid(odd, os.path.join(self.d, "out.tif"), grid)
        with rasterio.open(out) as s:
            self.assertEqual(s.count, 3)
            self.assertEqual(s.nodata, 0)
            self.assertAlmostEqual(s.res[0], PX)
            fx = (s.transform.c - grid.origin_x) / grid.res_x
            fy = (grid.origin_y - s.transform.f) / grid.res_y
            self.assertAlmostEqual(fx, round(fx), places=6)
            self.assertAlmostEqual(fy, round(fy), places=6)

    def test_registration_recovers_a_known_shift(self):
        a = _write(os.path.join(self.d, "a.tif"), 0, 0, 1600, 1600)
        grid = iu.reference_grid([a])
        ref = iu.build_reference([a], grid, os.path.join(self.d, "ref.vrt"))
        # New imagery of the middle, misregistered 3 m east and 2 m south.
        new = _write(os.path.join(self.d, "new.tif"), 400, 400, 800, 800, shift_m=(3.0, -2.0))
        rep = iu.coregister_to_reference(new, ref, grid, os.path.join(self.d, "new_al.tif"))
        reg = rep["registration"]
        self.assertTrue(reg["applied"], reg["reason"])
        self.assertAlmostEqual(reg["dx_m"], -3.0, delta=PX)
        self.assertAlmostEqual(reg["dy_m"], 2.0, delta=PX)
        # After registration the new pixels equal the reference pixels.
        with rasterio.open(rep["file"]) as n, rasterio.open(ref) as r:
            w = rasterio.windows.from_bounds(*n.bounds, transform=r.transform).round_offsets().round_lengths()
            diff = np.abs(n.read(1).astype(int) - r.read(1, window=w).astype(int))
        self.assertLess(float(np.mean(diff)), 1.0)

    def test_new_area_without_overlap_is_snapped_not_shifted(self):
        a = _write(os.path.join(self.d, "a.tif"), 0, 0, 600, 600)
        grid = iu.reference_grid([a])
        ref = iu.build_reference([a], grid, os.path.join(self.d, "ref.vrt"))
        far = _write(os.path.join(self.d, "far.tif"), 1000, 1000, 500, 500)
        rep = iu.coregister_to_reference(far, ref, grid, os.path.join(self.d, "far_al.tif"))
        self.assertFalse(rep["registration"]["applied"])
        self.assertIn("overlap", rep["registration"]["reason"])


class TestPrepareUpdate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        # Existing: two side-by-side tiles. New: covers the left tile fully
        # (and a strip of the right one), misregistered by 2 m.
        self.left = _write(os.path.join(self.d, "left.tif"), 0, 0, 800, 1600)
        self.right = _write(os.path.join(self.d, "right.tif"), 800, 0, 800, 1600)
        self.new = _write(os.path.join(self.d, "new.tif"), 0, 0, 1000, 1600, shift_m=(2.0, 0.0))

    def tearDown(self):
        self.tmp.cleanup()

    def test_replaces_the_covered_tile_and_tiles_on_one_grid(self):
        r = iu.prepare_update(
            [iu.UpdateFile(self.left, "u/left.tif"), iu.UpdateFile(self.right, "u/right.tif")],
            [iu.UpdateFile(self.new, "u/new.tif")],
            os.path.join(self.d, "work"),
        )
        self.assertEqual(r.retired_urls, ["u/left.tif"])
        self.assertEqual(r.active_urls, ["u/right.tif", "u/new.tif"])
        self.assertEqual(len(r.mosaic_paths), 2)  # right (kept) then new on top
        self.assertTrue(r.report["incoming"][0]["registration"]["applied"])
        g = r.report["grid"]
        for p in r.mosaic_paths:
            with rasterio.open(p) as s:
                self.assertEqual((s.count, s.nodata), (3, 0))
                fx = (s.transform.c - g["origin_x"]) / g["res_x"]
                self.assertAlmostEqual(fx, round(fx), places=6)

    def test_file_with_unknown_url_is_never_retired(self):
        r = iu.prepare_update(
            [iu.UpdateFile(self.left, None), iu.UpdateFile(self.right, "u/right.tif")],
            [iu.UpdateFile(self.new, "u/new.tif")],
            os.path.join(self.d, "work"),
        )
        self.assertEqual(r.retired_urls, [])
        self.assertEqual(len(r.mosaic_paths), 3)

    def test_requires_both_sides(self):
        with self.assertRaises(ValueError):
            iu.prepare_update([], [iu.UpdateFile(self.new, "n")], self.d)
        with self.assertRaises(ValueError):
            iu.prepare_update([iu.UpdateFile(self.left, "l")], [], self.d)


if __name__ == "__main__":
    unittest.main()
