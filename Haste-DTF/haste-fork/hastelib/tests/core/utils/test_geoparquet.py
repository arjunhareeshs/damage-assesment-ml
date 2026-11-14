# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import tempfile
import unittest

import geopandas as gpd
import pyarrow.parquet as pq
import shapely
from hastegeo.core.utils.geoparquet import write_geoparquet


class TestWriteGeoparquet(unittest.TestCase):
    def test_round_trips_as_geoparquet(self):
        polys = [shapely.box(-101.9, 35.4, -101.8, 35.5), shapely.box(-101.7, 35.3, -101.6, 35.35)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.parquet")
            n = write_geoparquet(path, {"building_id": ["a", "b"], "confidence": [0.9, 0.7]},
                                 [shapely.to_wkb(p) for p in polys])
            self.assertEqual(n, 2)
            self.assertFalse(os.path.exists(path + ".tmp"))
            meta = json.loads(pq.read_schema(path).metadata[b"geo"])
            self.assertEqual(meta["primary_column"], "geometry")
            self.assertEqual(meta["columns"]["geometry"]["bbox"], [-101.9, 35.3, -101.6, 35.5])
            gdf = gpd.read_parquet(path)  # readable by standard GeoParquet tools
            self.assertEqual(gdf.crs.to_epsg(), 4326)
            self.assertEqual(list(gdf["building_id"]), ["a", "b"])
            self.assertTrue(gdf.geometry.iloc[0].equals(polys[0]))


if __name__ == "__main__":
    unittest.main()
