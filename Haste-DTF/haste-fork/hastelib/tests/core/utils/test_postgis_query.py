# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for hastegeo.core.utils.postgis_query.

All DB access is mocked (connection/cursor), so these assert on the SQL
executed and its params rather than against a real PostGIS instance —
mirroring hastelib/tests/core/utils/test_postgis_load.py.
"""

import unittest
from unittest.mock import MagicMock

from hastegeo.core.utils.postgis_query import (
    buildings_in_aoi,
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

    def test_building_ids_take_precedence_over_aoi(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = [
            ("B101", '{"type": "Polygon", "coordinates": []}')
        ]

        result = buildings_in_aoi(
            "layer-1",
            aoi_geojson={"type": "Polygon", "coordinates": []},
            building_ids=["B101"],
            conn=connection,
        )

        sql_text, params = cursor.execute.call_args.args
        self.assertIn("building_id = ANY(%s)", sql_text)
        self.assertNotIn("ST_Intersects", sql_text)
        self.assertEqual(params, ("layer-1", ["B101"]))
        self.assertEqual(result, [{"building_id": "B101", "geometry": {"type": "Polygon", "coordinates": []}}])

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
