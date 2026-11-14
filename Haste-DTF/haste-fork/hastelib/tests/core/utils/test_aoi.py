# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for hastegeo.core.utils.aoi helpers.

``extract_aoi_polygon`` itself reads GeoTIFFs and is exercised by the
imageryprep workflow tests that mock it out; here we cover the simpler
``save_polygon_as_geojson`` helper directly so the file shape stays
stable for the UI's "Download Valid Area Mask" feature.
"""

import json
import os
import tempfile
import unittest

import shapely.geometry
from hastegeo.core.utils.aoi import save_polygon_as_geojson


class TestSavePolygonAsGeojson(unittest.TestCase):
    """Unit tests for ``save_polygon_as_geojson``."""

    def _polygon(self):
        return shapely.geometry.Polygon(
            [(-1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 1.0)]
        )

    def test_writes_featurecollection_with_polygon_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "mask.geojson")
            returned = save_polygon_as_geojson(self._polygon(), out)
            self.assertEqual(returned, out)
            self.assertTrue(os.path.exists(out))

            with open(out) as f:
                fc = json.load(f)

            self.assertEqual(fc["type"], "FeatureCollection")
            self.assertEqual(len(fc["features"]), 1)
            feat = fc["features"][0]
            self.assertEqual(feat["type"], "Feature")
            self.assertEqual(feat["geometry"]["type"], "Polygon")
            # First coordinate matches what we passed in.
            ring = feat["geometry"]["coordinates"][0]
            self.assertEqual(ring[0], [-1.0, 0.0])
            self.assertEqual(feat["properties"], {})

    def test_properties_are_written_when_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "mask.geojson")
            save_polygon_as_geojson(
                self._polygon(),
                out,
                properties={"source": "post_event_mosaic"},
            )
            with open(out) as f:
                fc = json.load(f)
            self.assertEqual(
                fc["features"][0]["properties"]["source"],
                "post_event_mosaic",
            )


if __name__ == "__main__":
    unittest.main()


class TestExtractAoiPolygon(unittest.TestCase):
    """``extract_aoi_polygon`` keeps every real imagery region."""

    def _tif(self, blocks):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        data = np.zeros((3, 400, 400), dtype=np.uint8)
        for r0, r1, c0, c1 in blocks:
            data[:, r0:r1, c0:c1] = 200
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.close()
        self.addCleanup(os.remove, tmp.name)
        with rasterio.open(
            tmp.name, "w", driver="GTiff", width=400, height=400, count=3,
            dtype="uint8", crs="EPSG:4326",
            transform=from_origin(-102.0, 35.5, 0.001, 0.001),
        ) as dst:
            dst.write(data)
        return tmp.name

    def test_two_separate_tiles_are_both_kept_and_specks_dropped(self):
        from hastegeo.core.utils.aoi import extract_aoi_polygon

        north = (0, 100, 0, 400)    # 100 x 400 px
        south = (250, 400, 0, 400)  # 150 x 400 px, larger; gap between
        speck = (180, 182, 10, 12)  # 4 px of noise in the gap
        geom = extract_aoi_polygon(self._tif([north, south, speck]))

        self.assertEqual(geom.geom_type, "MultiPolygon")
        self.assertEqual(len(geom.geoms), 2)
        # Both tiles' centres are inside; the gap and the speck are not.
        self.assertTrue(geom.contains(shapely.geometry.Point(-101.8, 35.45)))
        self.assertTrue(geom.contains(shapely.geometry.Point(-101.8, 35.18)))
        self.assertFalse(geom.contains(shapely.geometry.Point(-101.8, 35.32)))
        self.assertFalse(geom.contains(shapely.geometry.Point(-101.989, 35.319)))

    def test_single_tile_is_a_polygon(self):
        from hastegeo.core.utils.aoi import extract_aoi_polygon

        geom = extract_aoi_polygon(self._tif([(50, 350, 50, 350)]))
        self.assertEqual(geom.geom_type, "Polygon")
        self.assertAlmostEqual(geom.area, 0.3 * 0.3, places=3)

    def test_all_nodata_raises(self):
        from hastegeo.core.utils.aoi import extract_aoi_polygon

        with self.assertRaises(ValueError):
            extract_aoi_polygon(self._tif([]))
