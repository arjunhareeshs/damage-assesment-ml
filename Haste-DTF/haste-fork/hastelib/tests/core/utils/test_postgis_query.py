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
    set_review_status,
    update_building_predictions,
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


class TestUpdateBuildingPredictions(unittest.TestCase):
    def test_geometry_never_in_set_clause(self):
        connection, cursor = _mock_connection()
        cursor.rowcount = 2
        rows = [
            {
                "building_id": "B101",
                "class": 4,
                "class_name": "Total Structural Collapse",
                "confidence": 0.96,
                "mask_url": "mask_B101.png",
                "crop_url": "building_B101.png",
            },
            {
                "building_id": "B102",
                "class": 0,
                "class_name": "Undamaged",
                "confidence": 0.88,
                "mask_url": "mask_B102.png",
                "crop_url": "building_B102.png",
            },
        ]

        updated = update_building_predictions(
            "layer-1", "model-1", rows, conn=connection
        )

        self.assertEqual(updated, 2)
        sql_text, params = cursor.executemany.call_args.args
        self.assertNotIn("geometry", sql_text)
        self.assertIn("review_status = %(review_status)s", sql_text)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0]["review_status"], "pending_review")
        self.assertEqual(params[0]["model_id"], "model-1")
        self.assertEqual(params[0]["image_layer_id"], "layer-1")
        connection.commit.assert_called_once()

    def test_custom_review_status_propagates(self):
        connection, cursor = _mock_connection()
        cursor.rowcount = 1
        rows = [
            {
                "building_id": "B101",
                "class": 0,
                "class_name": "Undamaged",
                "confidence": 0.5,
                "mask_url": "",
                "crop_url": "",
            }
        ]

        update_building_predictions(
            "layer-1", "model-1", rows, review_status="accepted", conn=connection
        )

        _sql_text, params = cursor.executemany.call_args.args
        self.assertEqual(params[0]["review_status"], "accepted")


class TestSetReviewStatus(unittest.TestCase):
    def test_rejects_invalid_status(self):
        with self.assertRaises(ValueError):
            set_review_status("layer-1", "B101", "not-a-status", conn=MagicMock())

    def test_valid_status_updates_and_commits(self):
        connection, cursor = _mock_connection()
        cursor.rowcount = 1

        result = set_review_status(
            "layer-1", "B101", "accepted", conn=connection
        )

        self.assertTrue(result)
        sql_text, params = cursor.execute.call_args.args
        self.assertIn("SET review_status = %s", sql_text)
        self.assertEqual(params, ("accepted", "layer-1", "B101"))
        connection.commit.assert_called_once()

    def test_no_matching_row_returns_false(self):
        connection, cursor = _mock_connection()
        cursor.rowcount = 0

        result = set_review_status(
            "layer-1", "does-not-exist", "rejected", conn=connection
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()


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


class TestRecordRunResults(unittest.TestCase):
    def test_upserts_one_row_per_building_with_probabilities(self):
        from hastegeo.core.utils import postgis_query as pq_mod

        pq_mod._RUN_RESULTS_READY = True  # skip the DDL in this unit test
        connection, cursor = _mock_connection()
        n = pq_mod.record_run_results(
            "layer-1", "9187",
            [{"building_id": "B1", "class": 0, "class_name": "Undamaged",
              "confidence": 0.9, "probabilities": {"Undamaged": 0.9},
              "crop_url": "c", "mask_url": ""}],
            project_id="p", conn=connection,
        )
        sql_text, params = cursor.execute.call_args.args
        self.assertEqual(n, 1)
        self.assertIn("ON CONFLICT (model_id, building_id) DO UPDATE", sql_text)
        self.assertEqual(params[:7], ("9187", "B1", "layer-1", "p", 0, "Undamaged", 0.9))
        self.assertEqual(json.loads(params[7]), {"Undamaged": 0.9})
        self.assertEqual(params[8:], ("c", None))
        connection.commit.assert_called_once()

    def test_no_rows_is_a_no_op(self):
        from hastegeo.core.utils.postgis_query import record_run_results

        self.assertEqual(record_run_results("layer-1", "1", [], conn=MagicMock()), 0)
