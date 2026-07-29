# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DamageTriageFormer inference job, scoped to an analyst-drawn AOI or an
explicit building selection (Option B, AOI-scoped —
``docs/OptionB_AOI_Integration_Plan.md``).

Mirrors ``processors/damage_triage.py`` (the whole-layer path) almost
exactly — same ``Model``-as-unit-of-work shape, same Azure Batch
submission/poll/finalize state machine — but differs at the two points
that matter:

* ``_execute_inference`` resolves the building shortlist via
  ``hastegeo.core.utils.postgis_query.buildings_in_aoi`` (an
  ``ST_Intersects`` query against ``hg_buildings``, scoped to
  ``model.dtfAoi``/``model.dtfBuildingIds``) instead of handing the whole
  layer's cached footprints file to the batch task.
* ``_update_results_from_job`` PATCHes the matched rows in ``hg_buildings``
  (``postgis_query.update_building_predictions``) rather than repainting a
  full-layer raster — every other building's row, and any earlier
  whole-layer ``dtf`` model's raster/gpkg, are left untouched. New
  predictions always land at ``review_status="pending_review"`` (never
  auto-``"accepted"``) — see ``docs/OptionB_AOI_Integration_Plan.md``
  "Decide: human review or fully automated?".

Runs on the same ``damage-triage-aoi-infer`` training-image entrypoint
(``hastegeo.workflows.damage_triage_aoi_infer``, registered next to the
whole-layer path's ``damage-triage-infer`` in ``hastelib/pyproject.toml``
and ``docker/training/Dockerfile``), driven by the queue named
``config.queue_config["dtf_aoi_queue_name"]`` (a separate queue from the
whole-layer path's ``"dtf_queue_name"``, wired in
``api/hastefuncqueues/function_app.py``'s
``GetRunDTFAoiInferenceQueueTrigger``).
"""

import json

from ..config import ArtifactTypes, Config
from ..data_layer.unified import UnifiedDataLayer
from ..models.projects import ImageLayer, Model, TrainingJob
from ..runners.unified_runner import UnifiedRunner
from ..utils import postgis_query
from ..utils.data import extract_from_url
from ..utils.logs import Logger
from ..utils.metadata import MetadataUtils
from ..utils.queues import AzureQueueHandler
from .damage_triage import _download_hf_checkpoint

BATCH_JOB_WORKDIR = "AZ_BATCH_TASK_WORKING_DIR"
DTF_AOI_PREFIX = "dtfaoi"


class DTFAoiInferencePreprocessor:
    def __init__(self, model: Model, config: Config = None):
        if config is None:
            config = Config()
        self.queue_client = AzureQueueHandler(
            config.queue_config["queue_connection_string"],
            config.queue_config["dtf_aoi_queue_name"],
            config.queue_config["queue_account_url"],
        )
        self.model_data = model
        self.config = config

    def send_to_queue(self):
        self.model_data.status = self.config.get_status_types().PENDING.value
        self.model_data.currentStep = 0
        self.model_data.progressPct = 0.0
        # 2 friendly steps: submit -> inference/finalize (same shape as the
        # whole-layer dtf path).
        self.model_data.totalSteps = 2
        self.queue_client.put_message(json.dumps(self.model_data.dict()))
        self.model_data.statusMessage = MetadataUtils.append_status_message(
            self.model_data.statusMessage,
            "Queued for AOI-scoped DamageTriageFormer inference",
        )
        return self.model_data


class DTFAoiInferencePostprocessor:
    def __init__(
        self,
        model: Model,
        image_layer: ImageLayer = None,
        config: Config = None,
    ):
        if config is None:
            config = Config()
        self.config = config
        self.model_data = model
        self.image_layer = image_layer
        self.storage = UnifiedDataLayer(
            storage_type=config.storage_type,
            partition_key=model.projectId,
            **config.storage_config,
        )
        self.logger = Logger.get_logger(__name__)
        self.runner = UnifiedRunner(
            runner_type=config.runner_type,
            config=self.config,
            pool_id=self.config.get_azure_batch_config()["training_pool_id"],
        )
        self.queue_client = AzureQueueHandler(
            config.queue_config["queue_connection_string"],
            config.queue_config["dtf_aoi_queue_name"],
            config.queue_config["queue_account_url"],
        )

    def process(self):
        self.logger.info(
            f"{self.__class__.__name__}.process: model "
            f"{self.model_data.modelId} status {self.model_data.status}"
        )

        if (
            self.model_data.status
            == self.config.get_status_types().PENDING.value
        ):
            self._update_progress(
                "Resolving buildings inside AOI", step=0
            )
            self.model_data = self._execute_inference()

        elif (
            self.model_data.status
            == self.config.get_status_types().IN_PROGRESS.value
        ):
            task_status = self.runner.get_task_status(
                job_id=self.model_data.dtfAoiJob.jobId,
                task_id=self.model_data.dtfAoiJob.taskId,
            )
            self.logger.info(
                f"Task status for DTF-AOI model {self.model_data.modelId} "
                f"is {task_status}"
            )

            if task_status == self.config.get_status_types().COMPLETED.value:
                self.model_data.status = task_status
                self.model_data.dtfAoiJob.status = task_status
                self.model_data.dtfAoiJob.completedDate = (
                    MetadataUtils.get_timestamp()
                )
                self._update_results_from_job()
                for log in self._get_friendly_logs():
                    if log[1] not in self.model_data.statusMessage:
                        self._update_progress(log[1], timestamp=log[0])
                self.runner.cleanup_task(
                    job_id=self.model_data.dtfAoiJob.jobId,
                    task_id=self.model_data.dtfAoiJob.taskId,
                )

            elif task_status == self.config.get_status_types().FAILED.value:
                self.model_data.status = task_status
                self.model_data.dtfAoiJob.status = task_status
                self.model_data.dtfAoiJob.completedDate = (
                    MetadataUtils.get_timestamp()
                )
                for log in self._get_friendly_logs():
                    if log[1] not in self.model_data.statusMessage:
                        self._update_progress(log[1], timestamp=log[0])
                self._update_progress(
                    "AOI-scoped DamageTriageFormer job failed",
                    step=self.model_data.currentStep,
                )
                self.runner.cleanup_task(
                    job_id=self.model_data.dtfAoiJob.jobId,
                    task_id=self.model_data.dtfAoiJob.taskId,
                )
            else:
                self.model_data.status = task_status
                self.model_data.dtfAoiJob.status = task_status
                self.queue_client.put_message(
                    json.dumps(self.model_data.dict())
                )

        return self.model_data

    def _execute_inference(self):
        try:
            buildings = postgis_query.buildings_in_aoi(
                image_layer_id=self.model_data.imageLayerId,
                aoi_geojson=self.model_data.dtfAoi,
                building_ids=self.model_data.dtfBuildingIds,
            )
            if not buildings:
                raise ValueError(
                    "No buildings found for the given AOI/selection "
                    "(hg_buildings has no matching rows for this image "
                    "layer — has imagery prep finished loading footprints "
                    "into PostGIS yet?)."
                )
            self._update_progress(
                f"{len(buildings)} buildings resolved from AOI", step=0
            )

            input_files = self._create_dtf_aoi_config(buildings)
            command = (
                f'"mkdir -p ${BATCH_JOB_WORKDIR} '
                f"&& cd ${BATCH_JOB_WORKDIR} "
                f"&& damage-triage-aoi-infer --config "
                f'${BATCH_JOB_WORKDIR}/{input_files["config"]["file_path"]}'
                '"'
            )
            job_id = self.config.get_azure_batch_config()[
                "training_batch_job_id"
            ][:64]
            task_id = f"{DTF_AOI_PREFIX}-{MetadataUtils.generate_id()}"
            output_prefix = (
                f"{MetadataUtils.hash_string(self.model_data.projectId)}"
                f"/{task_id}"
            )
            self.runner.add_task(
                job_id=job_id,
                task_id=task_id,
                output_prefix=output_prefix,
                resource_files_for_upload=input_files,
                file_pattern=f"${BATCH_JOB_WORKDIR}/outputs/*.*",
                command=command,
                image_name=self.config.get_azure_batch_config()[
                    "docker_image"
                ],
            )
            self.model_data.dtfAoiJob = TrainingJob(
                jobId=job_id,
                taskId=task_id,
                modelId=self.model_data.modelId,
                projectId=self.model_data.projectId,
                status=self.config.get_status_types().IN_PROGRESS.value,
                creationDate=MetadataUtils.get_timestamp(),
            )
            self.model_data.status = (
                self.config.get_status_types().IN_PROGRESS.value
            )
            self._update_progress(
                "AOI-scoped DamageTriageFormer inference submitted with "
                f"task id {task_id}",
                step=1,
            )
            self.queue_client.put_message(json.dumps(self.model_data.dict()))
        except Exception as e:
            self.logger.error(
                "Error executing AOI-scoped DTF inference for model "
                f"{self.model_data.modelId}: {e}",
                stack_info=True,
            )
            self.model_data.status = (
                self.config.get_status_types().FAILED.value
            )
            self._update_progress(
                f"AOI-scoped DamageTriageFormer inference failed: {e}",
                step=self.model_data.currentStep,
            )
        return self.model_data

    def _create_dtf_aoi_config(self, buildings: list):
        filename_pattern = (
            rf"{MetadataUtils.hash_string(self.model_data.projectId)}/(.*)\?+"
        )
        plain_url_pattern = r"(.*)\?+"

        imagery_url = self.image_layer.postEventMosaicCogImageryUrl
        if not imagery_url:
            raise ValueError("Image layer has no post-event mosaic COG.")

        self.logger.info(
            "Downloading DamageTriageFormer checkpoint from HF..."
        )
        checkpoint_local_path, config_local_path = _download_hf_checkpoint()

        imagery_fn = (
            f"inputs/{extract_from_url(imagery_url, filename_pattern)}"
        )
        checkpoint_fn = "inputs/best_model.pth"
        checkpoint_config_fn = "inputs/dtf_config.json"

        gpkg_name = (
            ArtifactTypes.BUILDING_DTF_AOI_PREDICTIONS_GPKG.value.substitute(
                modelName=self.model_data.modelId
            )
            + ".gpkg"
        )

        dtf_aoi_config = {
            "project_id": self.model_data.projectId,
            "image_layer_id": self.model_data.imageLayerId,
            "model_id": self.model_data.modelId,
            "output_dir": "outputs",
            "buildings": buildings,
            "files": {
                "imagery": imagery_fn,
                "predictions_gpkg": gpkg_name,
                "checkpoint": checkpoint_fn,
                "checkpoint_config": checkpoint_config_fn,
            },
            "checkpoint_cache": {
                "local_checkpoint": checkpoint_local_path,
                "local_config": config_local_path,
            },
            "pipeline": {
                "crop_size": 1024,
                "pad_px": 32,
            },
        }
        self.storage.save(
            identifier=self.model_data.modelId,
            data=dtf_aoi_config,
            data_type=self.config.get_metadata_types().DTF_AOI_CONFIG.value,
            data_format="json",
        )
        config_filepath = self.storage.get_file_remote_path(
            self.model_data.modelId,
            self.config.get_metadata_types().DTF_AOI_CONFIG.value,
            data_format="json",
        )

        input_files = {
            "config": {
                "http_url": extract_from_url(
                    config_filepath, plain_url_pattern
                ),
                "file_path": f"inputs/{extract_from_url(config_filepath, filename_pattern)}",
            },
            "imagery": {
                "http_url": extract_from_url(
                    imagery_url, plain_url_pattern
                ),
                "file_path": imagery_fn,
            },
        }
        return input_files

    def _update_results_from_job(self):
        content = self.runner.get_filecontent_from_task(
            job_id=self.model_data.dtfAoiJob.jobId,
            task_id=self.model_data.dtfAoiJob.taskId,
            filename="damage_triage_aoi_manifest.json",
        )
        if not content:
            raise FileNotFoundError(
                "AOI-scoped DamageTriageFormer manifest not found for "
                f"model {self.model_data.modelId}"
            )
        manifest = json.loads(content)
        building_rows = manifest.get("buildings", [])

        # Write results back to hg_buildings — the polygon geometry column
        # is never touched, only attributes: plan.pdf's "the polygon never
        # changed, only its attribute changed."
        classified_rows = [r for r in building_rows if r.get("class", -1) != -1]
        if classified_rows:
            postgis_query.update_building_predictions(
                image_layer_id=self.model_data.imageLayerId,
                model_id=self.model_data.modelId,
                rows=[
                    {
                        "building_id": r["building_id"],
                        "class": r["class"],
                        "class_name": r["class_name"],
                        "confidence": r["confidence"],
                        "mask_url": self._artifact_url(
                            r.get("mask_url", ""), "png"
                        ),
                        "crop_url": self._artifact_url(
                            r.get("crop_url", ""), "png"
                        ),
                    }
                    for r in classified_rows
                ],
                review_status="pending_review",
            )

        gpkg_filename = manifest.get("predictions_gpkg_filename", "")
        self.model_data.gpkgUrl = self._artifact_url(gpkg_filename, "gpkg")
        self._update_progress(
            f"Classified {manifest.get('num_classified', '?')}/"
            f"{manifest.get('num_buildings', '?')} buildings in AOI",
            step=self.model_data.totalSteps,
        )

    def _artifact_url(self, filename: str, data_format: str = "gpkg") -> str:
        if not filename:
            return ""
        return self.storage.get_file_remote_path(
            identifier=filename,
            extra_partition_keys=f"{self.model_data.dtfAoiJob.taskId}",
            data_format=data_format,
        )

    def _get_friendly_logs(self):
        content = self.runner.get_filecontent_from_task(
            job_id=self.model_data.dtfAoiJob.jobId,
            task_id=self.model_data.dtfAoiJob.taskId,
            filename="damage_triage_aoi_friendly.log",
        )
        logs = []
        if content:
            for record in content.splitlines():
                if not record:
                    continue
                parts = record.split("|", 1)
                if len(parts) == 2:
                    logs.append((parts[0], parts[1]))
        return logs

    def _update_progress(
        self, message: str, step: int = None, timestamp: str = None
    ):
        if step is not None:
            self.model_data.currentStep = int(step)
        else:
            self.model_data.currentStep += 1
        total = int(self.model_data.totalSteps or 1)
        self.model_data.progressPct = round(
            int(self.model_data.currentStep) / total * 100, 2
        )
        self.model_data.statusMessage = MetadataUtils.append_status_message(
            self.model_data.statusMessage, message, timestamp=timestamp
        )
