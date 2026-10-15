# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for hastegeo.core.utils.imagery_catalog."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping

from hastegeo.core.utils.imagery_catalog import imagery_for_aoi, register_imagery


def _create_synthetic_tif(path, bounds=(-95.5, 29.5, -95.0, 30.0), width=100, height=100):
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        data = np.zeros((3, height, width), dtype=np.uint8)
        dst.write(data)


class TestImageryCatalog(unittest.TestCase):
    def setUp(self):
        self.image_layer = SimpleNamespace(
            imageLayerId="layer-cat-1", projectId="proj-cat-1"
        )

    def _mock_connection(self):
        connection = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = ["img_test_123"]
        connection.cursor.return_value.__enter__.return_value = cursor
        return connection, cursor

    def test_register_imagery_reads_header_only_and_inserts(self):
        connection, cursor = self._mock_connection()
        with tempfile.TemporaryDirectory() as tmp:
            tif_path = os.path.join(tmp, "sample.tif")
            _create_synthetic_tif(tif_path)

            rec = register_imagery(
                image_layer=self.image_layer,
                file_uri="sample_cog.tif",
                local_path=tif_path,
                conn=connection,
            )

        self.assertEqual(rec["imagery_id"], "img_test_123")
        self.assertEqual(rec["image_layer_id"], "layer-cat-1")
        self.assertEqual(rec["project_id"], "proj-cat-1")
        self.assertEqual(rec["width"], 100)
        self.assertEqual(rec["height"], 100)
        self.assertEqual(rec["band_count"], 3)
        self.assertEqual(rec["dtype"], "uint8")
        self.assertEqual(len(rec["transform"]), 6)
        self.assertEqual(rec["footprint"]["type"], "Polygon")

        cursor.execute.assert_called_once()
        connection.commit.assert_called_once()

    def test_imagery_for_aoi_executes_spatial_query(self):
        connection = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            (
                "img_1",
                "proj-1",
                "layer-1",
                "file_1.tif",
                json.dumps(mapping(box(-95.5, 29.5, -95.0, 30.0))),
                "EPSG:4326",
                100,
                100,
                0.005,
                0.005,
                3,
                "uint8",
                None,
                "post",
                "1.0",
                [0.005, 0.0, -95.5, 0.0, -0.005, 30.0],
            )
        ]
        connection.cursor.return_value.__enter__.return_value = cursor

        aoi_poly = mapping(box(-95.4, 29.6, -95.1, 29.9))
        results = imagery_for_aoi("layer-1", aoi_poly, conn=connection)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["imagery_id"], "img_1")
        self.assertEqual(results[0]["file_uri"], "file_1.tif")
        self.assertEqual(results[0]["transform"][2], -95.5)
        self.assertIn("ST_Intersects", cursor.execute.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
