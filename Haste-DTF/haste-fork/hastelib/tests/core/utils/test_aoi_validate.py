# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from shapely.geometry import LineString, Point, Polygon

from hastegeo.core.utils.aoi_validate import validate_aoi


class TestAoiValidate(unittest.TestCase):
    def test_valid_polygon(self):
        # A small box in Texas: roughly 0.1 deg (~11 km on a side)
        coords = [
            [-97.5, 30.2],
            [-97.4, 30.2],
            [-97.4, 30.3],
            [-97.5, 30.3],
            [-97.5, 30.2],
        ]
        geojson = {"type": "Polygon", "coordinates": [coords]}
        geom, area_km2 = validate_aoi(geojson, max_area_km2=500.0)

        self.assertIsInstance(geom, Polygon)
        self.assertGreater(area_km2, 100.0)
        self.assertLess(area_km2, 130.0)
        self.assertEqual(list(geom.exterior.coords), [tuple(c) for c in coords])

    def test_valid_feature_dict(self):
        coords = [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]]
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {},
        }
        geom, area_km2 = validate_aoi(feature)
        self.assertIsInstance(geom, Polygon)
        self.assertGreater(area_km2, 0.0)

    def test_reject_non_polygon(self):
        point_json = {"type": "Point", "coordinates": [0.0, 0.0]}
        with self.assertRaises(ValueError) as ctx:
            validate_aoi(point_json)
        self.assertIn("Unsupported geometry type", str(ctx.exception))

        line_geom = LineString([[0, 0], [1, 1]])
        with self.assertRaises(ValueError) as ctx:
            validate_aoi(line_geom)
        self.assertIn("Unsupported geometry type", str(ctx.exception))

    def test_reject_empty_geometry(self):
        empty_poly = Polygon()
        with self.assertRaises(ValueError) as ctx:
            validate_aoi(empty_poly)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_reject_invalid_self_intersecting(self):
        # Bowtie self-intersecting polygon
        bowtie_coords = [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
        bowtie_json = {"type": "Polygon", "coordinates": [bowtie_coords]}
        with self.assertRaises(ValueError) as ctx:
            validate_aoi(bowtie_json)
        self.assertIn("invalid", str(ctx.exception).lower())

    def test_reject_out_of_bounds_coords(self):
        oob_coords = [
            [190.0, 0.0],
            [191.0, 0.0],
            [191.0, 1.0],
            [190.0, 1.0],
            [190.0, 0.0],
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_aoi({"type": "Polygon", "coordinates": [oob_coords]})
        self.assertIn("WGS84 range", str(ctx.exception))

    def test_reject_antimeridian_crossing(self):
        antimeridian_coords = [
            [179.0, 0.0],
            [-179.0, 0.0],
            [-179.0, 1.0],
            [179.0, 1.0],
            [179.0, 0.0],
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_aoi({"type": "Polygon", "coordinates": [antimeridian_coords]})
        self.assertIn("180th meridian", str(ctx.exception))

    def test_reject_exceeding_max_area(self):
        coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
        # ~1 degree near equator is ~12,300 km²
        with self.assertRaises(ValueError) as ctx:
            validate_aoi({"type": "Polygon", "coordinates": [coords]}, max_area_km2=5000.0)
        self.assertIn("exceeds the maximum allowable limit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
