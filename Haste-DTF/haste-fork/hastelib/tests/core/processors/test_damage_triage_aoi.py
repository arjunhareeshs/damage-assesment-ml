# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Targeted unit tests for DTFAoiInferencePostprocessor / Preprocessor.

Mirrors hastelib/tests/core/processors/test_imagery_preprocess_config.py's
approach: heavy external dependencies (UnifiedDataLayer, UnifiedRunner,
AzureQueueHandler, the PostGIS query layer, the HF checkpoint download) are
mocked/patched, and we assert on the shape of the config/queue calls each
method makes rather than exercising real Azure Batch or Postgres.
"""

import unittest
from unittest.mock import MagicMock, patch


class _StopAfterCall(Exception):
    pass


def _build_model(**overrides):
    from hastegeo.core.models.projects import Model

    defaults = dict(
        modelId="model-1",
        projectId="proj-1",
        imageLayerId="layer-1",
        modelType="dtf-aoi",
        dtfAoi={"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        dtfBuildingIds=None,
        status="pending",
        statusMessage="",
        currentStep=0,
        totalSteps=0,
    )
    defaults.update(overrides)
    return Model(**defaults)


def _build_image_layer():
    from hastegeo.core.models.projects import ImageLayer

    image_layer = MagicMock(spec=ImageLayer)
    image_layer.postEventMosaicCogImageryUrl = (
        "https://example.blob.core.windows.net/c/mosaic.tif?sv=x"
    )
    return image_layer


class TestDTFAoiInferencePreprocessor(unittest.TestCase):
    def test_send_to_queue_sets_pending_status_and_two_steps(self):
        with patch(
            "hastegeo.core.processors.damage_triage_aoi.AzureQueueHandler",
            autospec=True,
        ):
            from hastegeo.core.processors.damage_triage_aoi import (
                DTFAoiInferencePreprocessor,
            )

            model = _build_model()
            preprocessor = DTFAoiInferencePreprocessor(model)
            result = preprocessor.send_to_queue()

        from hastegeo.core.config import Config

        self.assertEqual(
            result.status, Config.get_status_types().PENDING.value
        )
        self.assertEqual(result.totalSteps, 2)
        self.assertEqual(result.currentStep, 0)
        preprocessor.queue_client.put_message.assert_called_once()


class TestDTFAoiInferencePostprocessorExecuteInference(unittest.TestCase):
    def _build_processor(self):
        with patch(
            "hastegeo.core.processors.damage_triage_aoi.UnifiedDataLayer",
            autospec=True,
        ), patch(
            "hastegeo.core.processors.damage_triage_aoi.UnifiedRunner",
            autospec=True,
        ), patch(
            "hastegeo.core.processors.damage_triage_aoi.AzureQueueHandler",
            autospec=True,
        ):
            from hastegeo.core.processors.damage_triage_aoi import (
                DTFAoiInferencePostprocessor,
            )

            model = _build_model()
            processor = DTFAoiInferencePostprocessor(
                model, image_layer=_build_image_layer()
            )
        return processor

    @patch(
        "hastegeo.core.processors.damage_triage_aoi._download_hf_checkpoint",
        return_value=("/tmp/best_model.pth", "/tmp/dtf_config.json"),
    )
    @patch("hastegeo.core.processors.damage_triage_aoi.postgis_query")
    def test_no_buildings_marks_model_failed(
        self, mock_postgis_query, _mock_download
    ):
        mock_postgis_query.buildings_in_aoi.return_value = []
        processor = self._build_processor()

        result = processor._execute_inference()

        from hastegeo.core.config import Config

        self.assertEqual(
            result.status, Config.get_status_types().FAILED.value
        )
        self.assertIn(
            "No buildings found", result.statusMessage
        )

    @patch(
        "hastegeo.core.processors.damage_triage_aoi._download_hf_checkpoint",
        return_value=("/tmp/best_model.pth", "/tmp/dtf_config.json"),
    )
    @patch("hastegeo.core.processors.damage_triage_aoi.postgis_query")
    def test_config_includes_resolved_buildings(
        self, mock_postgis_query, _mock_download
    ):
        resolved = [
            {"building_id": "B101", "geometry": {"type": "Polygon", "coordinates": []}},
            {"building_id": "B102", "geometry": {"type": "Polygon", "coordinates": []}},
        ]
        mock_postgis_query.buildings_in_aoi.return_value = resolved
        processor = self._build_processor()

        captured = {}

        def fake_save(**kwargs):
            captured.update(kwargs)
            raise _StopAfterCall()

        processor.storage.save = MagicMock(side_effect=fake_save)
        processor.storage.get_file_remote_path = MagicMock(
            return_value="https://example/config.json?sv=x"
        )

        try:
            processor._execute_inference()
        except _StopAfterCall:
            pass

        cfg = captured["data"]
        self.assertEqual(cfg["buildings"], resolved)
        self.assertEqual(cfg["project_id"], "proj-1")
        self.assertEqual(cfg["image_layer_id"], "layer-1")
        self.assertEqual(cfg["model_id"], "model-1")

    @patch(
        "hastegeo.core.processors.damage_triage_aoi._download_hf_checkpoint",
        return_value=("/tmp/best_model.pth", "/tmp/dtf_config.json"),
    )
    @patch("hastegeo.core.processors.damage_triage_aoi.postgis_query")
    def test_building_ids_selection_passed_through(
        self, mock_postgis_query, _mock_download
    ):
        mock_postgis_query.buildings_in_aoi.return_value = [
            {"building_id": "B101", "geometry": {"type": "Polygon", "coordinates": []}}
        ]
        model = _build_model(dtfAoi=None, dtfBuildingIds=["B101"])
        with patch(
            "hastegeo.core.processors.damage_triage_aoi.UnifiedDataLayer",
            autospec=True,
        ), patch(
            "hastegeo.core.processors.damage_triage_aoi.UnifiedRunner",
            autospec=True,
        ), patch(
            "hastegeo.core.processors.damage_triage_aoi.AzureQueueHandler",
            autospec=True,
        ):
            from hastegeo.core.processors.damage_triage_aoi import (
                DTFAoiInferencePostprocessor,
            )

            processor = DTFAoiInferencePostprocessor(
                model, image_layer=_build_image_layer()
            )
        processor.storage.save = MagicMock(side_effect=_StopAfterCall())
        try:
            processor._execute_inference()
        except _StopAfterCall:
            pass

        mock_postgis_query.buildings_in_aoi.assert_called_once_with(
            image_layer_id="layer-1", aoi_geojson=None, building_ids=["B101"]
        )


class TestDTFAoiInferencePostprocessorUpdateResults(unittest.TestCase):
    def _build_processor(self):
        with patch(
            "hastegeo.core.processors.damage_triage_aoi.UnifiedDataLayer",
            autospec=True,
        ), patch(
            "hastegeo.core.processors.damage_triage_aoi.UnifiedRunner",
            autospec=True,
        ), patch(
            "hastegeo.core.processors.damage_triage_aoi.AzureQueueHandler",
            autospec=True,
        ):
            from hastegeo.core.processors.damage_triage_aoi import (
                DTFAoiInferencePostprocessor,
            )

            model = _build_model(totalSteps=2, currentStep=1)
            processor = DTFAoiInferencePostprocessor(
                model, image_layer=_build_image_layer()
            )
        from hastegeo.core.models.projects import TrainingJob

        processor.model_data.dtfAoiJob = TrainingJob(
            jobId="job-1", taskId="task-1"
        )
        return processor

    @patch("hastegeo.core.processors.damage_triage_aoi.postgis_query")
    def test_pending_review_never_auto_accepted(self, mock_postgis_query):
        processor = self._build_processor()
        manifest = {
            "num_buildings": 1,
            "num_classified": 1,
            "predictions_gpkg_filename": "dtf_aoi_predictions_model-1.gpkg",
            "buildings": [
                {
                    "building_id": "B101",
                    "class": 4,
                    "class_name": "Total Structural Collapse",
                    "confidence": 0.96,
                    "crop_url": "building_B101.png",
                    "mask_url": "mask_B101.png",
                }
            ],
        }
        processor.runner.get_filecontent_from_task = MagicMock(
            return_value=__import__("json").dumps(manifest)
        )
        processor.storage.get_file_remote_path = MagicMock(
            return_value="https://example/artifact"
        )

        processor._update_results_from_job()

        mock_postgis_query.update_building_predictions.assert_called_once()
        _args, kwargs = mock_postgis_query.update_building_predictions.call_args
        self.assertEqual(kwargs["review_status"], "pending_review")
        self.assertEqual(kwargs["image_layer_id"], "layer-1")
        self.assertEqual(kwargs["model_id"], "model-1")
        self.assertEqual(len(kwargs["rows"]), 1)
        self.assertEqual(kwargs["rows"][0]["building_id"], "B101")

    @patch("hastegeo.core.processors.damage_triage_aoi.postgis_query")
    def test_unclassified_buildings_are_not_written(self, mock_postgis_query):
        processor = self._build_processor()
        manifest = {
            "num_buildings": 1,
            "num_classified": 0,
            "predictions_gpkg_filename": "",
            "buildings": [
                {
                    "building_id": "B999",
                    "class": -1,
                    "class_name": "Unclassified",
                    "confidence": 0.0,
                    "crop_url": "",
                    "mask_url": "",
                }
            ],
        }
        processor.runner.get_filecontent_from_task = MagicMock(
            return_value=__import__("json").dumps(manifest)
        )

        processor._update_results_from_job()

        mock_postgis_query.update_building_predictions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
