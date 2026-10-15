# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for hastegeo.core.utils.region_windows."""

import unittest
from rasterio.transform import from_bounds
from shapely.geometry import box

from hastegeo.core.utils.region_windows import (
    building_coverage,
    group_batches,
    select_units,
)


class TestBuildingCoverage(unittest.TestCase):
    def test_fully_contained(self):
        bldg = box(10, 10, 20, 20)
        win = box(0, 0, 100, 100)
        self.assertAlmostEqual(building_coverage(bldg, win), 1.0)

    def test_disjoint(self):
        bldg = box(10, 10, 20, 20)
        win = box(50, 50, 100, 100)
        self.assertEqual(building_coverage(bldg, win), 0.0)

    def test_partial_coverage(self):
        # 10x10 building = area 100
        # window covers [10, 15] x [10, 20] = 5x10 = area 50 -> 0.5
        bldg = box(10, 10, 20, 20)
        win = box(0, 0, 15, 100)
        self.assertAlmostEqual(building_coverage(bldg, win), 0.5)


class TestSelectUnits(unittest.TestCase):
    def setUp(self):
        # 2000x2000 raster spanning (-95.5, 29.5) to (-95.3, 29.7)
        self.raster_width = 2048
        self.raster_height = 2048
        self.transform = from_bounds(-95.5, 29.5, -95.3, 29.7, 2048, 2048)
        self.tile_size = 1024

    def test_tif_outside_aoi_produces_no_units(self):
        # AOI completely far away
        aoi = box(-94.0, 31.0, -93.9, 31.1)
        buildings = [{"building_id": "b1", "geometry": box(-95.45, 29.55, -95.44, 29.56)}]
        units = select_units(
            imagery_id="img_1",
            raster_width=self.raster_width,
            raster_height=self.raster_height,
            raster_transform=self.transform,
            aoi_geom=aoi,
            eligible_buildings=buildings,
            tile_size=self.tile_size,
        )
        self.assertEqual(len(units), 0)

    def test_tif_inside_aoi_with_no_buildings_produces_no_units(self):
        aoi = box(-95.5, 29.5, -95.3, 29.7)
        units = select_units(
            imagery_id="img_1",
            raster_width=self.raster_width,
            raster_height=self.raster_height,
            raster_transform=self.transform,
            aoi_geom=aoi,
            eligible_buildings=[],
            tile_size=self.tile_size,
        )
        self.assertEqual(len(units), 0)

    def test_selects_only_windows_with_eligible_buildings(self):
        # Whole raster inside AOI
        aoi = box(-95.5, 29.5, -95.3, 29.7)
        # 1 building in top-left tile
        b_box = from_bounds(-95.5, 29.6, -95.4, 29.7, 1024, 1024)
        bldg_geom = box(-95.48, 29.65, -95.47, 29.66)
        buildings = [{"building_id": "b1", "geometry": bldg_geom}]

        units = select_units(
            imagery_id="img_1",
            raster_width=self.raster_width,
            raster_height=self.raster_height,
            raster_transform=self.transform,
            aoi_geom=aoi,
            eligible_buildings=buildings,
            tile_size=self.tile_size,
        )
        # Only the tile containing b1 should be selected
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["kind"], "grid")
        self.assertEqual(units[0]["buildings"][0]["building_id"], "b1")

    def test_adds_building_centred_unit_when_split(self):
        # Place building right on the boundary between col_off 0 and 1024
        # Inverse transform maps x, y to pixel col, row
        aoi = box(-95.5, 29.5, -95.3, 29.7)
        # Center x is at col 1024:
        center_x = -95.4  # exact midpoint of -95.5 and -95.3
        bldg_geom = box(center_x - 0.002, 29.65, center_x + 0.002, 29.66)
        buildings = [{"building_id": "split_bldg", "geometry": bldg_geom}]

        units = select_units(
            imagery_id="img_1",
            raster_width=self.raster_width,
            raster_height=self.raster_height,
            raster_transform=self.transform,
            aoi_geom=aoi,
            eligible_buildings=buildings,
            tile_size=self.tile_size,
        )
        # Should have 2 grid units (one for each half) + 1 building_centred unit
        kinds = [u["kind"] for u in units]
        self.assertIn("building_centred", kinds)
        self.assertIn("grid", kinds)


class TestGroupBatches(unittest.TestCase):
    def test_groups_by_tif_and_batch_size(self):
        units = [
            {"unit_id": f"u_img1_{i}", "imagery_id": "img1", "col_off": i, "row_off": 0}
            for i in range(5)
        ] + [
            {"unit_id": f"u_img2_{i}", "imagery_id": "img2", "col_off": i, "row_off": 0}
            for i in range(3)
        ]

        batches = group_batches(units, windows_per_batch=2, assessment_id="ast_1")

        # img1: 5 units with batch size 2 -> 3 batches (2, 2, 1)
        # img2: 3 units with batch size 2 -> 2 batches (2, 1)
        # Total = 5 batches
        self.assertEqual(len(batches), 5)
        img1_batches = [b for b in batches if b["imagery_id"] == "img1"]
        img2_batches = [b for b in batches if b["imagery_id"] == "img2"]
        self.assertEqual(len(img1_batches), 3)
        self.assertEqual(len(img2_batches), 2)

        # Ensure batch_id set on units
        for b in batches:
            for u in b["units"]:
                self.assertEqual(u["batch_id"], b["batch_id"])

    def test_rejects_non_positive_batch_size(self):
        with self.assertRaises(ValueError):
            group_batches([], windows_per_batch=0)


if __name__ == "__main__":
    unittest.main()
