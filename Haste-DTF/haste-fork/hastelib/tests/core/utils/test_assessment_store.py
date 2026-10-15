# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from unittest.mock import MagicMock

from hastegeo.core.utils.assessment_store import (
    claim_next_batch,
    complete_batch,
    create_assessment,
    expire_leases,
    fail_batch,
    get_assessment,
    get_assessment_statistics,
    insert_processing_units,
    update_assessment_status,
    upsert_damage_assessments,
    upsert_prediction_evidence,
)


def _mock_connection():
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection, cursor


class TestAssessmentStore(unittest.TestCase):
    def test_create_assessment(self):
        connection, cursor = _mock_connection()
        assessment_data = {
            "assessment_id": "asmt-1",
            "project_id": "proj-1",
            "image_layer_id": "layer-1",
            "aoi_geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            "aoi_area_km2": 120.5,
        }
        res = create_assessment(assessment_data, conn=connection)
        self.assertEqual(res["assessment_id"], "asmt-1")

        sql_text, params = cursor.execute.call_args.args
        self.assertIn("INSERT INTO hg_assessments", sql_text)
        self.assertIn("ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)", sql_text)
        self.assertEqual(params[0], "asmt-1")
        self.assertEqual(params[1], "proj-1")
        connection.commit.assert_called_once()

    def test_get_assessment(self):
        connection, cursor = _mock_connection()
        # Mock row return
        cursor.fetchone.return_value = (
            "asmt-1", "proj-1", "layer-1", None, "Test Assessment",
            '{"type": "Polygon", "coordinates": []}', 120.5, "intersects_aoi",
            "running", '{"tile_size": 1024}', "repo", "digest123", "1.0.0",
            100, 10, 5, 0, "analyst", None, None, None, None
        )
        res = get_assessment("asmt-1", conn=connection)
        self.assertIsNotNone(res)
        self.assertEqual(res["assessment_id"], "asmt-1")
        self.assertEqual(res["aoi_area_km2"], 120.5)
        self.assertEqual(res["config"], {"tile_size": 1024})

    def test_update_status_terminal(self):
        connection, cursor = _mock_connection()
        cursor.rowcount = 1
        updated = update_assessment_status("asmt-1", "completed", conn=connection)
        self.assertTrue(updated)
        sql_text, params = cursor.execute.call_args.args
        self.assertIn("completed_at = NOW()", sql_text)
        self.assertEqual(params[0], "completed")
        self.assertEqual(params[-1], "asmt-1")

    def test_claim_next_batch(self):
        connection, cursor = _mock_connection()
        # Row 1 from batch update, Row 2 from unit query
        cursor.fetchone.return_value = ("batch-1", "asmt-1", 1)
        cursor.fetchall.return_value = [
            ("u-1", "img-1", 0, 0, 1024, 1024, "file://tif1.tif")
        ]

        claimed = claim_next_batch("asmt-1", lease_seconds=300, conn=connection)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["batch_id"], "batch-1")
        self.assertEqual(len(claimed["units"]), 1)
        self.assertEqual(claimed["units"][0]["file_uri"], "file://tif1.tif")

    def test_expire_leases(self):
        connection, cursor = _mock_connection()
        cursor.rowcount = 2  # mock 2 rows affected
        requeued, failed = expire_leases("asmt-1", max_attempts=3, conn=connection)
        self.assertEqual(requeued, 2)
        self.assertEqual(failed, 2)
        self.assertEqual(cursor.execute.call_count, 2)

    def test_complete_and_fail_batch(self):
        connection, cursor = _mock_connection()
        complete_batch("batch-1", runner_job_id="job-1", runner_task_id="task-1", conn=connection)
        self.assertEqual(cursor.execute.call_count, 2)

        fail_batch("batch-2", error="OOM failure", conn=connection)
        self.assertEqual(cursor.execute.call_count, 4)

    def test_upsert_prediction_evidence(self):
        connection, cursor = _mock_connection()
        rows = [
            {
                "assessment_id": "asmt-1",
                "building_id": "B101",
                "unit_id": "u-1",
                "probs": [0.9, 0.05, 0.02, 0.02, 0.01],
                "class": 0,
                "confidence": 0.9,
                "coverage_fraction": 1.0,
            }
        ]
        count = upsert_prediction_evidence(rows, conn=connection)
        self.assertEqual(count, 1)
        sql_text, _ = cursor.executemany.call_args.args
        self.assertIn("ON CONFLICT (assessment_id, building_id, unit_id) DO UPDATE", sql_text)

    def test_upsert_damage_assessments_damage_percentage_null(self):
        connection, cursor = _mock_connection()
        rows = [
            {
                "building_id": "B101",
                "image_layer_id": "layer-1",
                "damage_class": 0,
                "damage_class_name": "Undamaged",
                "confidence": 0.95,
                "result_status": "completed",
            }
        ]
        count = upsert_damage_assessments("asmt-1", rows, conn=connection)
        self.assertEqual(count, 1)
        sql_text, params_list = cursor.executemany.call_args.args
        self.assertIn("NULL", sql_text)
        self.assertIn("damage_percentage = NULL", sql_text)
        self.assertIn("ON CONFLICT (assessment_id, building_id) DO UPDATE", sql_text)

    def test_get_statistics(self):
        connection, cursor = _mock_connection()
        cursor.fetchall.return_value = [
            (0, "Undamaged", "completed", 50),
            (1, "Partial Roof Damage", "completed", 20),
            (-1, "Unknown", "insufficient_data", 5),
        ]
        stats = get_assessment_statistics("asmt-1", conn=connection)
        self.assertEqual(stats["total_assessed_buildings"], 75)
        self.assertEqual(stats["class_counts"]["Undamaged"], 50)
        self.assertEqual(stats["class_counts"]["Partial Roof Damage"], 20)
        self.assertEqual(stats["status_counts"]["completed"], 70)
        self.assertEqual(stats["status_counts"]["insufficient_data"], 5)


if __name__ == "__main__":
    unittest.main()
