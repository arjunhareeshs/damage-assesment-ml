# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for hastegeo.core.utils.postgis_query.

All DB access is mocked (connection/cursor), so these assert on the SQL
executed and its params rather than against a real PostGIS instance —
mirroring hastelib/tests/core/utils/test_postgis_load.py.
"""

import json
import unittest
from unittest.mock import MagicMock

from hastegeo.core.utils.postgis_query import (
    building_ids_in_aoi,
    buildings_in_aoi,
    layer_buildings_feature_collection,
    count_buildings_in_aoi,
)


def _mock_connection():
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection, cursor


class TestBuildingsInAoi(unittest.TestCase):
    def test_requires_aoi_or_building_ids(self):
        with self.assertRaises(ValueError):
            buildings_in_aoi("layer-1", conn=MagicMock())

    def test_building_ids_alone_skip_spatial_filter(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = [
            ("B101", '{"type": "Polygon", "coordinates": []}')
        ]

        result = buildings_in_aoi("layer-1", building_ids=["B101"], conn=connection)

        sql_text, params = cursor.execute.call_args.args
        self.assertIn("building_id = ANY(%s)", sql_text)
        self.assertNotIn("ST_Intersects", sql_text)
        self.assertEqual(params, ("layer-1", ["B101"]))
        self.assertEqual(result, [{"building_id": "B101", "geometry": {"type": "Polygon", "coordinates": []}}])

    def test_building_ids_with_aoi_keeps_aoi_filter(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = []
        aoi = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}

        buildings_in_aoi("layer-1", aoi_geojson=aoi, building_ids=["B101"], conn=connection)

        sql_text, params = cursor.execute.call_args.args
        self.assertIn("building_id = ANY(%s)", sql_text)
        self.assertIn("ST_Intersects", sql_text)
        self.assertEqual(params, ("layer-1", ["B101"], json.dumps(aoi)))

    def test_aoi_only_uses_st_intersects(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = []
        aoi = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}

        buildings_in_aoi("layer-1", aoi_geojson=aoi, conn=connection)

        sql_text, params = cursor.execute.call_args.args
        self.assertIn("ST_Intersects", sql_text)
        self.assertEqual(params[0], "layer-1")

    def test_does_not_close_injected_connection(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = []
        buildings_in_aoi(
            "layer-1", aoi_geojson={"type": "Polygon", "coordinates": []},
            conn=connection,
        )
        connection.close.assert_not_called()


class TestCountBuildingsInAoi(unittest.TestCase):
    def test_returns_int_count(self):
        connection, cursor = _mock_connection()
        cursor.fetchone.return_value = (42,)

        result = count_buildings_in_aoi(
            "layer-1", building_ids=["B101", "B102"], conn=connection
        )

        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)


class TestLayerBuildingsFeatureCollection(unittest.TestCase):
    def test_returns_features_total_and_truncation(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = [
            ("B1", '{"type": "Polygon", "coordinates": []}', 5),
            ("B2", '{"type": "Polygon", "coordinates": []}', 5),
        ]
        fc = layer_buildings_feature_collection("layer-1", limit=2, conn=connection)
        sql_text, params = cursor.execute.call_args.args
        self.assertIn("count(*) OVER ()", sql_text)
        self.assertIn("ORDER BY building_id", sql_text)
        self.assertNotIn("ST_MakeEnvelope", sql_text)
        self.assertEqual(params, ("layer-1", 2))
        self.assertEqual([f["properties"]["id"] for f in fc["features"]], ["B1", "B2"])
        self.assertEqual(fc["total"], 5)
        self.assertTrue(fc["truncated"])

    def test_bbox_filter_and_empty_result(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = []
        fc = layer_buildings_feature_collection(
            "layer-1", bbox=(-102, 35, -101, 36), conn=connection
        )
        sql_text, params = cursor.execute.call_args.args
        self.assertIn("ST_MakeEnvelope(%s, %s, %s, %s, 4326)", sql_text)
        self.assertEqual(params[:5], ("layer-1", -102.0, 35.0, -101.0, 36.0))
        self.assertEqual(fc, {"type": "FeatureCollection", "features": [], "total": 0, "truncated": False})


class TestBuildingIdsInAoi(unittest.TestCase):
    def test_uses_the_run_predicate(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = [("B1",), ("B7",)]
        aoi = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
        ids = building_ids_in_aoi("layer-1", aoi, conn=connection)
        sql_text, params = cursor.execute.call_args.args
        self.assertIn("ST_Intersects", sql_text)
        self.assertEqual(params[:2], ("layer-1", json.dumps(aoi)))
        self.assertEqual(ids, ["B1", "B7"])

    def test_requires_aoi(self):
        with self.assertRaises(ValueError):
            building_ids_in_aoi("layer-1", None, conn=MagicMock())
