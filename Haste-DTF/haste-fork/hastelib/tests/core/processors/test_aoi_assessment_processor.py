# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Targeted unit tests for AoiAssessmentPreprocessor and Postprocessor."""

import unittest
from unittest.mock import MagicMock, patch

import hastegeo.core.processors.aoi_assessment as aoi_proc
from hastegeo.core.models.projects import (
    AoiAssessmentStatus,
    BatchStatus,
    BuildingResultStatus,
)


class TestAoiAssessmentPreprocessor(unittest.TestCase):
    def test_send_to_queue_puts_message(self):
        with patch.object(aoi_proc, "AzureQueueHandler") as mock_q:
            pre = aoi_proc.AoiAssessmentPreprocessor("ast-test-123")
            msg = pre.send_to_queue()

            self.assertEqual(msg["assessment_id"], "ast-test-123")
            pre.queue_client.put_message.assert_called_once()


class TestAoiAssessmentPostprocessor(unittest.TestCase):
    @patch.object(aoi_proc, "UnifiedRunner")
    @patch.object(aoi_proc, "AzureQueueHandler")
    @patch.object(aoi_proc, "assessment_store")
    def test_cancelling_flow(self, mock_store, mock_queue, mock_runner):
        mock_store.get_assessment.return_value = {
            "assessment_id": "ast-cancel-1",
            "status": AoiAssessmentStatus.CANCELLING.value,
        }
        mock_store.get_batches.return_value = [
            {
                "batch_id": "b-1",
                "status": BatchStatus.RUNNING.value,
                "runner_job_id": "job-1",
                "runner_task_id": "task-1",
            }
        ]

        post = aoi_proc.AoiAssessmentPostprocessor("ast-cancel-1")
        post.process()

        mock_store.update_assessment_status.assert_called_with(
            "ast-cancel-1",
            AoiAssessmentStatus.CANCELLED.value,
            conn=None,
        )

    @patch.object(aoi_proc, "UnifiedRunner")
    @patch.object(aoi_proc, "AzureQueueHandler")
    @patch.object(aoi_proc, "imagery_for_aoi", return_value=[])
    @patch.object(aoi_proc, "assessment_store")
    def test_fails_when_no_imagery_covers_aoi(
        self, mock_store, mock_imagery, mock_queue, mock_runner
    ):
        mock_store.get_assessment.return_value = {
            "assessment_id": "ast-noimg-1",
            "image_layer_id": "layer-1",
            "aoi_geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            "status": AoiAssessmentStatus.SUBMITTED.value,
        }

        post = aoi_proc.AoiAssessmentPostprocessor("ast-noimg-1")
        post.process()

        mock_store.update_assessment_status.assert_any_call(
            "ast-noimg-1",
            AoiAssessmentStatus.FAILED.value,
            conn=None,
            error="no imagery covers this AOI",
            eligible_buildings=0,
            total_units=0,
        )

    @patch.object(aoi_proc, "UnifiedRunner")
    @patch.object(aoi_proc, "AzureQueueHandler")
    @patch.object(aoi_proc.postgis_query, "count_buildings_in_aoi", return_value=0)
    @patch.object(aoi_proc, "imagery_for_aoi", return_value=[{"imagery_id": "img1"}])
    @patch.object(aoi_proc, "assessment_store")
    def test_completes_with_zero_results_when_no_buildings(
        self, mock_store, mock_imagery, mock_query, mock_queue, mock_runner
    ):
        mock_store.get_assessment.return_value = {
            "assessment_id": "ast-nobldg-1",
            "image_layer_id": "layer-1",
            "aoi_geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            "status": AoiAssessmentStatus.SUBMITTED.value,
        }

        post = aoi_proc.AoiAssessmentPostprocessor("ast-nobldg-1")
        post.process()

        mock_store.update_assessment_status.assert_any_call(
            "ast-nobldg-1",
            AoiAssessmentStatus.COMPLETED.value,
            conn=None,
            eligible_buildings=0,
            total_units=0,
            completed_units=0,
            failed_units=0,
        )
        # A finished assessment is not re-enqueued.
        mock_queue.return_value.put_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
