# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DamageTriageFormer inference job: queue + execute the DTF workflow
(Option B — HASTE's map/UI/report shell, DamageTriageFormer's 5-class
typology model for the actual prediction).

Mirrors ``processors/embedding.py``: a ``Model`` with ``modelType="dtf"`` is
the unit of work. The preprocessor enqueues it; the postprocessor submits a
task to the training docker image running ``damage-triage-infer`` (the
workflow module ``damage_triage_infer.py``), polls it, and on completion
writes the per-building predictions GeoPackage URL onto ``model.gpkgUrl`` —
the same field the embedding sub-model uses — so the existing
Validation/Assessment reports and ``GetModelArtifact?kind=gpkg`` download
work unchanged.

Checkpoint is auto-downloaded from Hugging Face on first run and cached locally.
"""

import json
from pathlib import Path

from ..config import ArtifactTypes, Config
from ..data_layer.unified import UnifiedDataLayer
from ..models.projects import ImageLayer, Model, TrainingJob
from ..runners.unified_runner import UnifiedRunner
from ..utils.data import extract_from_url
from ..utils.logs import Logger
from ..utils.metadata import MetadataUtils
from ..utils.queues import AzureQueueHandler

# Do not prefix with '$'. Replaced at runtime with the task working directory.
BATCH_JOB_WORKDIR = "AZ_BATCH_TASK_WORKING_DIR"
DTF_PREFIX = "dtf"
DTF_HF_MODEL_ID = "Ymx1025/DamageTriageFormer-model"


def _download_hf_checkpoint():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    checkpoint_path = cache_dir / "best_model.pth"
    config_path = cache_dir / "dtf_config.json"

    if not checkpoint_path.exists():
        Logger.get_logger(__name__).info(
            f"Downloading DamageTriageFormer checkpoint from {DTF_HF_MODEL_ID}..."
        )
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=DTF_HF_MODEL_ID,
                filename="best_model.pth",
                cache_dir=str(cache_dir),
            )
        )

    if not config_path.exists():
        Logger.get_logger(__name__).info(
            f"Downloading DamageTriageFormer config from {DTF_HF_MODEL_ID}..."
        )
        config_path = Path(
            hf_hub_download(
                repo_id=DTF_HF_MODEL_ID,
                filename="config.json",
                cache_dir=str(cache_dir),
            )
        )

    return str(checkpoint_path), str(config_path)


class DTFInferencePreprocessor:
    def __init__(self, model: Model, config: Config = None):
        if config is None:
            config = Config()
        self.queue_client = AzureQueueHandler(
            config.queue_config["queue_connection_string"],
            config.queue_config["dtf_queue_name"],
            config.queue_config["queue_account_url"],
        )
        self.model_data = model
        self.config = config

    def send_to_queue(self):
        self.model_data.status = self.config.get_status_types().PENDING.value
        self.model_data.currentStep = 0
        self.model_data.progressPct = 0.0
        # 2 friendly steps: submit -> inference/finalize.
        self.model_data.totalSteps = 2
        self.queue_client.put_message(json.dumps(self.model_data.dict()))
        self.model_data.statusMessage = MetadataUtils.append_status_message(
            self.model_data.statusMessage, "Queued for DamageTriageFormer inference"
        )
        return self.model_data


class DTFInferencePostprocessor:
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
            config.queue_config["dtf_queue_name"],
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
            self._update_progress("Submitting DamageTriageFormer job", step=0)
            self.model_data = self._execute_inference()

        elif (
            self.model_data.status
            == self.config.get_status_types().IN_PROGRESS.value
        ):
            task_status = self.runner.get_task_status(
                job_id=self.model_data.dtfJob.jobId,
                task_id=self.model_data.dtfJob.taskId,
            )
            self.logger.info(
                f"Task status for DTF model {self.model_data.modelId} "
                f"is {task_status}"
            )

            if task_status == self.config.get_status_types().COMPLETED.value:
                self.model_data.status = task_status
                self.model_data.dtfJob.status = task_status
                self.model_data.dtfJob.completedDate = (
                    MetadataUtils.get_timestamp()
                )
                self._update_results_from_job()
                for log in self._get_friendly_logs():
                    if log[1] not in self.model_data.statusMessage:
                        self._update_progress(log[1], timestamp=log[0])
                self.runner.cleanup_task(
                    job_id=self.model_data.dtfJob.jobId,
                    task_id=self.model_data.dtfJob.taskId,
                )

            elif task_status == self.config.get_status_types().FAILED.value:
                self.model_data.status = task_status
                self.model_data.dtfJob.status = task_status
                self.model_data.dtfJob.completedDate = (
                    MetadataUtils.get_timestamp()
                )
                for log in self._get_friendly_logs():
                    if log[1] not in self.model_data.statusMessage:
                        self._update_progress(log[1], timestamp=log[0])
                self._update_progress(
                    "DamageTriageFormer job failed",
                    step=self.model_data.currentStep,
                )
                self.runner.cleanup_task(
                    job_id=self.model_data.dtfJob.jobId,
                    task_id=self.model_data.dtfJob.taskId,
                )
            else:
                self.model_data.status = task_status
                self.model_data.dtfJob.status = task_status
                self.queue_client.put_message(
                    json.dumps(self.model_data.dict())
                )

        return self.model_data

    def _execute_inference(self):
        try:
            input_files = self._create_dtf_config()
            command = (
                f'"mkdir -p ${BATCH_JOB_WORKDIR} '
                f"&& cd ${BATCH_JOB_WORKDIR} "
                f"&& damage-triage-infer --config "
                f'${BATCH_JOB_WORKDIR}/{input_files["config"]["file_path"]}'
                '"'
            )
            job_id = self.config.get_azure_batch_config()[
                "training_batch_job_id"
            ][:64]
            task_id = f"{DTF_PREFIX}-{MetadataUtils.generate_id()}"
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
            self.model_data.dtfJob = TrainingJob(
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
                f"DamageTriageFormer inference submitted with task id {task_id}",
                step=1,
            )
            self.queue_client.put_message(json.dumps(self.model_data.dict()))
        except Exception as e:
            self.logger.error(
                f"Error executing DTF inference for model "
                f"{self.model_data.modelId}: {e}",
                stack_info=True,
            )
            self.model_data.status = (
                self.config.get_status_types().FAILED.value
            )
            self._update_progress(
                f"DamageTriageFormer inference failed: {e}",
                step=self.model_data.currentStep,
            )
        return self.model_data

    def _create_dtf_config(self):
        filename_pattern = (
            rf"{MetadataUtils.hash_string(self.model_data.projectId)}/(.*)\?+"
        )
        plain_url_pattern = r"(.*)\?+"

        imagery_url = self.image_layer.postEventMosaicCogImageryUrl
        footprints_url = self.image_layer.buildingFootprintsUrl
        if not imagery_url:
            raise ValueError("Image layer has no post-event mosaic COG.")
        if not footprints_url:
            raise ValueError("Image layer has no building footprints.")

        self.logger.info("Downloading DamageTriageFormer checkpoint from HF...")
        checkpoint_local_path, config_local_path = _download_hf_checkpoint()

        imagery_fn = (
            f"inputs/{extract_from_url(imagery_url, filename_pattern)}"
        )
        footprints_fn = (
            f"inputs/{extract_from_url(footprints_url, filename_pattern)}"
        )
        checkpoint_fn = f"inputs/best_model.pth"
        checkpoint_config_fn = f"inputs/dtf_config.json"

        gpkg_name = (
            ArtifactTypes.BUILDING_DTF_PREDICTIONS_GPKG.value.substitute(
                modelName=self.model_data.modelId
            )
            + ".gpkg"
        )

        dtf_config = {
            "project_id": self.model_data.projectId,
            "image_layer_id": self.model_data.imageLayerId,
            "model_id": self.model_data.modelId,
            "output_dir": "outputs",
            "files": {
                "imagery": imagery_fn,
                "footprints": footprints_fn,
                "predictions_gpkg": gpkg_name,
                "checkpoint": checkpoint_fn,
                "checkpoint_config": checkpoint_config_fn,
            },
            "checkpoint_cache": {
                "local_checkpoint": checkpoint_local_path,
                "local_config": config_local_path,
            },
            "pipeline": {
                "tile_size": 1024,
            },
        }
        self.storage.save(
            identifier=self.model_data.modelId,
            data=dtf_config,
            data_type=self.config.get_metadata_types().DTF_CONFIG.value,
            data_format="json",
        )
        config_filepath = self.storage.get_file_remote_path(
            self.model_data.modelId,
            self.config.get_metadata_types().DTF_CONFIG.value,
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
                "http_url": extract_from_url(imagery_url, plain_url_pattern),
                "file_path": imagery_fn,
            },
            "footprints": {
                "http_url": extract_from_url(
                    footprints_url, plain_url_pattern
                ),
                "file_path": footprints_fn,
            },
        }
        return input_files

    def _update_results_from_job(self):
        content = self.runner.get_filecontent_from_task(
            job_id=self.model_data.dtfJob.jobId,
            task_id=self.model_data.dtfJob.taskId,
            filename="damage_triage_manifest.json",
        )
        if not content:
            raise FileNotFoundError(
                f"DamageTriageFormer manifest not found for model "
                f"{self.model_data.modelId}"
            )
        manifest = json.loads(content)
        gpkg_filename = manifest.get("predictions_gpkg_filename", "")
        self.model_data.gpkgUrl = self._artifact_url(gpkg_filename, "gpkg")
        # Map-ready rasters (see write_visualizer_rasters in
        # damage_triage_infer.py). predictedDamageLayerUrl is the same
        # Model field the embedding sub-model's inference.py sets, so
        # GetVisualizerResults / the Visualizer map work for a DTF model row
        # unchanged; it derives the *_predictions.tif URL from this one by
        # string-replacing the suffix, so both filenames must share the same
        # basename prefix (guaranteed — both come from write_visualizer_rasters).
        visualizer_filename = manifest.get("visualizer_tif_filename", "")
        self.model_data.predictedDamageLayerUrl = self._artifact_url(
            visualizer_filename, "tif"
        )
        self._update_progress(
            f"Classified {manifest.get('num_classified', '?')}/"
            f"{manifest.get('num_buildings', '?')} buildings",
            step=self.model_data.totalSteps,
        )

    def _artifact_url(self, filename: str, data_format: str = "gpkg") -> str:
        if not filename:
            return ""
        return self.storage.get_file_remote_path(
            identifier=filename,
            extra_partition_keys=f"{self.model_data.dtfJob.taskId}",
            data_format=data_format,
        )

    def _get_friendly_logs(self):
        content = self.runner.get_filecontent_from_task(
            job_id=self.model_data.dtfJob.jobId,
            task_id=self.model_data.dtfJob.taskId,
            filename="damage_triage_friendly.log",
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
