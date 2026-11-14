# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from shapely.geometry import box, mapping

from hastegeo.core.utils.aoi_analysis import (
    _area_km2,
    INCOMPLETE_IMAGERY,
    NO_BUILDINGS,
    NO_IMAGERY,
    READY,
    analyze_aoi,
    estimate_windows,
)


def _imagery(imagery_id, geom, width=4096, height=4096, origin=(-97.5, 30.3)):
    px = (geom.bounds[2] - geom.bounds[0]) / width
    return {
        "imagery_id": imagery_id,
        "footprint": mapping(geom),
        "crs": "EPSG:4326",
        "width": width,
        "height": height,
        "pixel_size_x": px,
        "pixel_size_y": px,
        "file_uri": f"/data/{imagery_id}.tif",
        "source_version": None,
        "transform": [px, 0, geom.bounds[0], 0, -px, geom.bounds[3]],
    }


class TestAoiAnalysis(unittest.TestCase):
    def setUp(self):
        self.tile_a = box(-97.5, 30.2, -97.4, 30.3)
        self.tile_b = box(-97.4, 30.2, -97.3, 30.3)

    def test_fully_covered_single_tile_ready(self):
        aoi = box(-97.48, 30.22, -97.42, 30.28)
        out = analyze_aoi(aoi, _area_km2(aoi), [_imagery("a", self.tile_a)], 120)
        self.assertEqual(out["state"], READY)
        self.assertEqual(out["fullyCoveredTiles"], 1)
        self.assertEqual(out["partiallyCoveredTiles"], 0)
        self.assertAlmostEqual(out["coveragePercent"], 100.0, places=1)
        self.assertIsNone(out["missingGeometry"])
        self.assertTrue(out["canSubmit"])

    def test_aoi_spanning_two_tiles_counts_both_partial(self):
        aoi = box(-97.45, 30.22, -97.35, 30.28)
        out = analyze_aoi(aoi, _area_km2(aoi), [_imagery("a", self.tile_a), _imagery("b", self.tile_b)], 10,
        )
        self.assertEqual(out["imageryCount"], 2)
        self.assertEqual(out["partiallyCoveredTiles"], 2)
        self.assertEqual(out["state"], READY)

    def test_partial_coverage_is_reported_not_hidden(self):
        # AOI extends 0.1 deg beyond the only tile.
        aoi = box(-97.45, 30.22, -97.3, 30.28)
        out = analyze_aoi(aoi, _area_km2(aoi), [_imagery("a", self.tile_a)], 10)
        self.assertEqual(out["state"], INCOMPLETE_IMAGERY)
        self.assertLess(out["coveragePercent"], 100.0)
        self.assertGreater(out["missingAreaKm2"], 0.0)
        self.assertIsNotNone(out["missingGeometry"])
        self.assertTrue(out["canSubmit"])

    def test_no_imagery(self):
        out = analyze_aoi(box(0, 0, 0.1, 0.1), _area_km2(box(0, 0, 0.1, 0.1)), [], 5)
        self.assertEqual(out["state"], NO_IMAGERY)
        self.assertFalse(out["canSubmit"])
        self.assertEqual(out["coveragePercent"], 0.0)

    def test_no_buildings(self):
        aoi = box(-97.48, 30.22, -97.42, 30.28)
        out = analyze_aoi(aoi, _area_km2(aoi), [_imagery("a", self.tile_a)], 0)
        self.assertEqual(out["state"], NO_BUILDINGS)
        self.assertFalse(out["canSubmit"])

    def test_max_units_blocks_submit(self):
        aoi = box(-97.5, 30.2, -97.4, 30.3)
        out = analyze_aoi(aoi, _area_km2(aoi), [_imagery("a", self.tile_a)], 10, max_units=2)
        self.assertTrue(out["exceedsMaxUnits"])
        self.assertFalse(out["canSubmit"])

    def test_estimate_windows_is_tile_aligned_and_clipped(self):
        img = _imagery("a", self.tile_a)  # 4096 px -> 4x4 windows of 1024
        whole = box(-97.5, 30.2, -97.4, 30.3)
        self.assertEqual(estimate_windows(whole, img), 16)
        # Quarter of the tile = 2x2 windows
        quarter = box(-97.5, 30.25, -97.45, 30.3)
        self.assertEqual(estimate_windows(quarter, img), 4)
        outside = box(-96.0, 30.25, -95.9, 30.3)
        self.assertEqual(estimate_windows(outside, img), 0)


if __name__ == "__main__":
    unittest.main()
