# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import unittest

try:
    import geopandas as gpd
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    from hastegeo.core.utils.twdb_buildings import rows_from_frame, to_storage_geometry
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "geopandas not installed")
class TestTwdbBuildings(unittest.TestCase):
    def _frame(self):
        # ~Texas, expressed in EPSG:3857, as a Z multipolygon like the source.
        sq = Polygon([(-10_800_000, 3_800_000, 0), (-10_799_900, 3_800_000, 0),
                      (-10_799_900, 3_800_100, 0), (-10_800_000, 3_800_100, 0)])
        # second building sits elsewhere so the multi-part one is a valid MultiPolygon
        tri_far = lambda p: Polygon([(x + 5_000, y, 0) for x, y, *_ in p.exterior.coords])
        tri = Polygon([(-10_800_000, 3_800_000, 0), (-10_799_900, 3_800_000, 0),
                       (-10_799_950, 3_800_100, 0)])
        two = MultiPolygon([sq, tri_far(tri)])
        return gpd.GeoDataFrame(
            {"Bldg_ID": [1, 2, 2, 3], "Simp_type": ["Residential"] * 4,
             "SVI": [0.5, 0.5, 0.5, None], "Est_floors": [None] * 4},
            geometry=[MultiPolygon([sq]), MultiPolygon([tri]), MultiPolygon([tri]), two],
            crs=3857,
        )

    def test_reprojects_flattens_and_unwraps_single_parts(self):
        geoms = to_storage_geometry(self._frame().geometry)
        self.assertEqual(geoms.crs.to_epsg(), 4326)
        x, y = geoms.iloc[0].exterior.coords[0]
        self.assertTrue(-100 < x < -90 and 30 < y < 40)
        self.assertEqual(geoms.iloc[0].geom_type, "Polygon")
        self.assertEqual(geoms.iloc[3].geom_type, "MultiPolygon")
        self.assertFalse(any(g.has_z for g in geoms))

    def test_rows_make_ids_unique_and_keep_originals(self):
        rows = rows_from_frame(self._frame(), "layer", "proj", "FPR02")
        self.assertEqual(
            [r[0] for r in rows], ["FPR02-1", "FPR02-2", "FPR02-2-2", "FPR02-3"]
        )
        self.assertEqual(json.loads(rows[2][7])["bldg_id"], "2")
        self.assertEqual(rows[0][4], "Residential")
        attrs = json.loads(rows[3][7])
        self.assertIsNone(attrs["svi"])
        self.assertEqual(rows[0][5], "twdb_cycle2_2025")
        self.assertEqual(shapely.from_wkb(rows[0][3]).geom_type, "Polygon")


if __name__ == "__main__":
    unittest.main()
