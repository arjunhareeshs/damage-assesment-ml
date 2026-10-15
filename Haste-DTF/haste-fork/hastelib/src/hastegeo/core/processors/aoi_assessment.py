# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Orchestrator for large-AOI batched damage assessments.

One ``hg_assessments`` row is one job. Each queue message advances it by
one step, and the processor re-enqueues itself (with the queue's default
visibility delay) until a terminal status is reached — the same
poll-by-queue pattern as ``processors/damage_triage_aoi.py``, but with all
job state durable in PostGIS instead of in the queue message:

    submitted -> discovering -> tiling -> queued -> running
              -> aggregating -> completed | completed_with_errors
    (any step) -> failed | rejected;  cancelling -> cancelled

* discovering: imagery (``hg_imagery``) and buildings (``hg_buildings``)
  intersecting the exact AOI polygon; pins the checkpoint digest.
* tiling: per TIF, 1024 px windows holding >=1 eligible building
  (``region_windows.select_units``), grouped into batches.
* running: up to ``max_concurrent_batches`` runner tasks
  (``damage-triage-region-infer``), each holding a lease; finished tasks'
  ``evidence.jsonl`` is upserted into ``hg_prediction_evidence``; failed or
  expired tasks are retried up to ``max_attempts``.
* aggregating: one result per building (``region_aggregate.aggregate``)
  upserted into ``hg_damage_assessments``.

Runs in the queue Functions app, which has no torch: import nothing from
``hastegeo.workflows.damage_triage_infer`` here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rasterio import Affine
from shapely import STRtree
from shapely.geometry import shape

from ..config import Config
from ..data_layer.unified import UnifiedDataLayer
from ..models.projects import (
    AoiAssessmentStatus,
    BatchStatus,
    BuildingResultStatus,
)
from ..runners.unified_runner import UnifiedRunner
from ..utils import assessment_store, postgis_query
from ..utils.dtf_constants import DTF_IM_SIZE, compute_model_digest
from ..utils.dtf_preprocess import PREPROCESSING_VERSION
from ..utils.imagery_catalog import imagery_for_aoi
from ..utils.logs import Logger
from ..utils.metadata import MetadataUtils
from ..utils.queues import AzureQueueHandler
from ..utils.region_aggregate import aggregate
from ..utils.region_windows import group_batches, select_units

BATCH_JOB_WORKDIR = "AZ_BATCH_TASK_WORKING_DIR"
ASSESSMENT_TASK_PREFIX = "aoiast"
EVIDENCE_FILENAME = "evidence.jsonl"
MANIFEST_FILENAME = "region_manifest.json"

_CANCEL_STATES = (
    AoiAssessmentStatus.CANCELLING.value,
    AoiAssessmentStatus.CANCELLED.value,
)


class AssessmentConfigError(ValueError):
    """A required assessment parameter is missing or invalid."""


def _cfg_value(cfg: Dict[str, Any], key: str, env_name: str) -> Optional[str]:
    """Resolved value: the assessment's stored config first (captured at
    submit time), then the current environment."""
    val = cfg.get(key)
    if val in (None, ""):
        val = os.getenv(env_name)
    return None if val in (None, "") else str(val)


def cfg_int(cfg: Dict[str, Any], key: str, env_name: str, required: bool) -> Optional[int]:
    val = _cfg_value(cfg, key, env_name)
    if val is None:
        if required:
            raise AssessmentConfigError(
                f"{env_name} is not set; set it (see docker/.env.example) "
                "from a benchmark on your hardware."
            )
        return None
    parsed = int(val)
    if parsed <= 0:
        raise AssessmentConfigError(f"{env_name} must be positive, got {parsed}")
    return parsed


def cfg_float(cfg: Dict[str, Any], key: str, env_name: str) -> Optional[float]:
    val = _cfg_value(cfg, key, env_name)
    return None if val is None else float(val)


def resolve_checkpoint_digest() -> Optional[str]:
    """sha256 of the checkpoint this deployment will run, or None if the
    checkpoint is not visible from this container."""
    path = os.getenv("DTF_CHECKPOINT_PATH")
    if path and os.path.exists(path):
        return compute_model_digest(path)
    return None


def _uri_filename(uri: str) -> str:
    return os.path.basename(urlparse(uri).path)


def _uri_without_query(uri: str) -> str:
    return uri.split("?", 1)[0]


class AoiAssessmentPreprocessor:
    """Enqueues an assessment for asynchronous processing."""

    def __init__(self, assessment_id: str, config: Optional[Config] = None):
        if config is None:
            config = Config()
        self.config = config
        self.assessment_id = assessment_id
        self.queue_client = AzureQueueHandler(
            config.queue_config["queue_connection_string"],
            config.queue_config["assessment_queue_name"],
            config.queue_config["queue_account_url"],
        )

    def send_to_queue(self) -> Dict[str, Any]:
        msg = {
            "assessment_id": self.assessment_id,
            "action": "process",
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        }
        # Visible immediately: the first step should start right away.
        self.queue_client.put_message(json.dumps(msg), visibility_timeout=0)
        return msg


class AoiAssessmentPostprocessor:
    """Advances one assessment by one state-machine step per call."""

    def __init__(
        self,
        assessment_id: str,
        config: Optional[Config] = None,
        conn=None,
    ):
        if config is None:
            config = Config()
        self.config = config
        self.assessment_id = assessment_id
        self.conn = conn
        self.logger = Logger.get_logger(__name__)

        self.queue_client = AzureQueueHandler(
            config.queue_config["queue_connection_string"],
            config.queue_config["assessment_queue_name"],
            config.queue_config["queue_account_url"],
        )
        self.runner = UnifiedRunner(
            runner_type=config.runner_type,
            config=self.config,
            pool_id=self.config.get_azure_batch_config()["training_pool_id"],
        )
        self._storage = None

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def process(self) -> Dict[str, Any]:
        ast = assessment_store.get_assessment(self.assessment_id, conn=self.conn)
        if not ast:
            self.logger.error("Assessment %s not found", self.assessment_id)
            return {}

        status = ast["status"]
        self.logger.info(
            "Processing assessment %s with status %s", self.assessment_id, status
        )
        try:
            if status == AoiAssessmentStatus.CANCELLING.value:
                self._handle_cancelling(ast)
            elif status in (
                AoiAssessmentStatus.SUBMITTED.value,
                AoiAssessmentStatus.VALIDATING.value,
                AoiAssessmentStatus.DISCOVERING.value,
            ):
                self._handle_discovering(ast)
            elif status == AoiAssessmentStatus.TILING.value:
                self._handle_tiling(ast)
            elif status in (
                AoiAssessmentStatus.QUEUED.value,
                AoiAssessmentStatus.RUNNING.value,
            ):
                self._handle_running(ast)
            elif status == AoiAssessmentStatus.AGGREGATING.value:
                self._handle_aggregating(ast)
            # terminal states: nothing to do (a duplicate message)
        except AssessmentConfigError as exc:
            self.logger.error("Assessment %s misconfigured: %s", self.assessment_id, exc)
            self._set_status(AoiAssessmentStatus.FAILED.value, error=str(exc))
        except Exception as exc:
            self.logger.error(
                "Assessment %s failed in status %s: %s",
                self.assessment_id, status, exc, exc_info=True,
            )
            self._set_status(
                AoiAssessmentStatus.FAILED.value,
                error=f"{status} step failed: {exc}",
            )

        return assessment_store.get_assessment(self.assessment_id, conn=self.conn) or {}

    def _set_status(self, status: str, **kwargs) -> None:
        """Write a status unless a cancel request arrived meanwhile."""
        if status not in _CANCEL_STATES:
            current = assessment_store.get_assessment(self.assessment_id, conn=self.conn)
            if current and current["status"] in _CANCEL_STATES:
                self.logger.info(
                    "Assessment %s is %s; not overwriting with %s",
                    self.assessment_id, current["status"], status,
                )
                return
        assessment_store.update_assessment_status(
            self.assessment_id, status, conn=self.conn, **kwargs
        )

    def _re_enqueue(self) -> None:
        msg = {
            "assessment_id": self.assessment_id,
            "action": "continue",
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        }
        # Default visibility delay (AzureQueueHandler.put_message) paces polling.
        self.queue_client.put_message(json.dumps(msg))

    def _storage_for(self, project_id: str) -> UnifiedDataLayer:
        if self._storage is None:
            self._storage = UnifiedDataLayer(
                storage_type=self.config.storage_type,
                partition_key=project_id,
                **self.config.storage_config,
            )
        return self._storage

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------
    def _handle_discovering(self, ast: Dict[str, Any]) -> None:
        self._set_status(AoiAssessmentStatus.DISCOVERING.value)
        image_layer_id = ast["image_layer_id"]
        aoi = ast["aoi_geometry"]

        imagery = imagery_for_aoi(image_layer_id, aoi, conn=self.conn)
        if not imagery:
            self._set_status(
                AoiAssessmentStatus.FAILED.value,
                error="no imagery covers this AOI",
                eligible_buildings=0,
                total_units=0,
            )
            return

        n_buildings = postgis_query.count_buildings_in_aoi(
            image_layer_id=image_layer_id, aoi_geojson=aoi, conn=self.conn
        )
        if n_buildings == 0:
            self.logger.info("Zero buildings in AOI for %s", self.assessment_id)
            self._set_status(
                AoiAssessmentStatus.COMPLETED.value,
                eligible_buildings=0,
                total_units=0,
                completed_units=0,
                failed_units=0,
            )
            return

        digest = ast.get("model_digest") or resolve_checkpoint_digest()
        if digest is None:
            self.logger.warning(
                "Checkpoint not visible from the queue worker (DTF_CHECKPOINT_PATH); "
                "assessment %s runs without model digest pinning",
                self.assessment_id,
            )
        self._set_status(
            AoiAssessmentStatus.TILING.value,
            eligible_buildings=n_buildings,
            model_digest=digest,
        )
        self._re_enqueue()

    def _handle_tiling(self, ast: Dict[str, Any]) -> None:
        cfg = ast.get("config") or {}
        windows_per_batch = cfg_int(
            cfg, "windows_per_batch", "HASTE_REGION_WINDOWS_PER_BATCH", required=True
        )
        max_units = cfg_int(cfg, "max_units", "HASTE_AOI_MAX_UNITS", required=True)

        image_layer_id = ast["image_layer_id"]
        aoi = ast["aoi_geometry"]
        aoi_geom = shape(aoi)

        imagery = imagery_for_aoi(image_layer_id, aoi, conn=self.conn)
        buildings = assessment_store.eligible_buildings_with_overlap(
            image_layer_id, aoi, conn=self.conn
        )
        b_geoms = [shape(b["geometry"]) for b in buildings]
        tree = STRtree(b_geoms) if b_geoms else None

        all_units: List[Dict[str, Any]] = []
        for img in imagery:
            if not img.get("transform"):
                raise ValueError(
                    f"Imagery {img['imagery_id']} has no stored geotransform; "
                    "re-run imagery preparation for this layer."
                )
            if tree is None:
                break
            # Pre-filter to buildings inside this TIF before the exact
            # per-window work in select_units.
            candidates = tree.query(shape(img["footprint"]), predicate="intersects")
            if len(candidates) == 0:
                continue
            all_units.extend(
                select_units(
                    imagery_id=img["imagery_id"],
                    raster_width=img["width"],
                    raster_height=img["height"],
                    raster_transform=Affine(*img["transform"]),
                    aoi_geom=aoi_geom,
                    eligible_buildings=[
                        {"building_id": buildings[i]["building_id"], "geometry": b_geoms[i]}
                        for i in sorted(candidates)
                    ],
                    tile_size=DTF_IM_SIZE,
                    raster_crs=img["crs"],
                    assessment_id=self.assessment_id,
                )
            )

        if len(all_units) > max_units:
            self._set_status(
                AoiAssessmentStatus.REJECTED.value,
                error=(
                    f"AOI needs {len(all_units)} processing windows, above the "
                    f"limit of {max_units} (HASTE_AOI_MAX_UNITS). Draw a smaller AOI."
                ),
                total_units=len(all_units),
            )
            return

        if not all_units:
            # Buildings exist but no catalogued imagery covers any of them:
            # aggregation records each as insufficient_data.
            self._set_status(AoiAssessmentStatus.AGGREGATING.value, total_units=0)
            self._re_enqueue()
            return

        batches = group_batches(
            all_units, windows_per_batch=windows_per_batch, assessment_id=self.assessment_id
        )
        # Batches first: hg_processing_units.batch_id references hg_batches.
        assessment_store.insert_batches(batches, self.assessment_id, conn=self.conn)
        assessment_store.insert_processing_units(all_units, self.assessment_id, conn=self.conn)

        self._set_status(AoiAssessmentStatus.QUEUED.value, total_units=len(all_units))
        self._re_enqueue()

    def _handle_running(self, ast: Dict[str, Any]) -> None:
        cfg = ast.get("config") or {}
        max_concurrent = cfg_int(
            cfg, "max_concurrent_batches", "HASTE_REGION_MAX_CONCURRENT_BATCHES", required=True
        )
        max_attempts = cfg_int(cfg, "max_attempts", "HASTE_REGION_MAX_ATTEMPTS", required=True)
        lease_seconds = cfg_int(cfg, "lease_seconds", "HASTE_REGION_LEASE_SECONDS", required=True)

        assessment_store.expire_leases(self.assessment_id, max_attempts=max_attempts, conn=self.conn)

        for b in assessment_store.get_batches(self.assessment_id, conn=self.conn):
            if b["status"] != BatchStatus.RUNNING.value or not b.get("runner_task_id"):
                continue
            job_id, task_id = b.get("runner_job_id"), b["runner_task_id"]
            task_status = self.runner.get_task_status(job_id=job_id, task_id=task_id)
            if task_status == self.config.get_status_types().COMPLETED.value:
                self._ingest_batch_outputs(b, ast, max_attempts)
                self.runner.cleanup_task(job_id=job_id, task_id=task_id)
            elif task_status == self.config.get_status_types().FAILED.value:
                new_status = assessment_store.requeue_or_fail_batch(
                    b["batch_id"], max_attempts, f"runner task {task_id} failed", conn=self.conn
                )
                self.logger.warning(
                    "Batch %s task %s failed; batch now %s", b["batch_id"], task_id, new_status
                )
                self.runner.cleanup_task(job_id=job_id, task_id=task_id)
            else:
                assessment_store.renew_batch_lease(
                    b["batch_id"], lease_seconds=lease_seconds, conn=self.conn
                )

        batches = assessment_store.get_batches(self.assessment_id, conn=self.conn)
        running = [b for b in batches if b["status"] == BatchStatus.RUNNING.value]
        pending = [b for b in batches if b["status"] == BatchStatus.PENDING.value]
        for b in pending[: max(0, max_concurrent - len(running))]:
            self._start_batch(b, ast, cfg, lease_seconds, max_attempts)

        counts = assessment_store.unit_status_counts(self.assessment_id, conn=self.conn)
        batches = assessment_store.get_batches(self.assessment_id, conn=self.conn)
        active = any(
            b["status"] in (BatchStatus.PENDING.value, BatchStatus.RUNNING.value)
            for b in batches
        )
        progress = dict(
            completed_units=counts.get("completed", 0),
            failed_units=counts.get("failed", 0),
        )
        if active:
            self._set_status(AoiAssessmentStatus.RUNNING.value, **progress)
            self._re_enqueue()
        elif not any(b["status"] == BatchStatus.COMPLETED.value for b in batches):
            self._set_status(
                AoiAssessmentStatus.FAILED.value, error="All batches failed", **progress
            )
        else:
            self._set_status(AoiAssessmentStatus.AGGREGATING.value, **progress)
            self._re_enqueue()

    def _build_batch_config(
        self, ast: Dict[str, Any], cfg: Dict[str, Any], batch_id: str
    ) -> Dict[str, Any]:
        units = assessment_store.get_batch_units(batch_id, conn=self.conn)
        if not units:
            raise ValueError(f"Batch {batch_id} has no units")
        file_uris = {u["file_uri"] for u in units}
        if len(file_uris) != 1:
            raise ValueError(f"Batch {batch_id} spans {len(file_uris)} TIFs; expected 1")
        file_uri = file_uris.pop()

        geometries = assessment_store.get_building_geometries(
            ast["image_layer_id"],
            {b["building_id"] for u in units for b in u["buildings"]},
            conn=self.conn,
        )
        for u in units:
            u["buildings"] = [
                {**b, "geometry": geometries[b["building_id"]]}
                for b in u["buildings"]
                if b["building_id"] in geometries
            ]

        return {
            "assessment_id": self.assessment_id,
            "batch_id": batch_id,
            "imagery_id": units[0]["imagery_id"],
            "image_layer_id": ast["image_layer_id"],
            "output_dir": "outputs",
            "file_uri": file_uri,
            "files": {"imagery": f"inputs/{_uri_filename(file_uri)}"},
            "units": [
                {k: u[k] for k in ("unit_id", "col_off", "row_off", "width", "height", "kind", "buildings")}
                for u in units
            ],
            "normalization": _cfg_value(cfg, "normalization", "HASTE_DTF_NORMALIZATION") or "imagenet",
            "masks_per_forward": cfg_int(
                cfg, "masks_per_forward", "HASTE_REGION_MASKS_PER_FORWARD", required=False
            ),
            "min_valid_pixel_fraction": cfg_float(
                cfg, "min_valid_pixel_fraction", "HASTE_REGION_MIN_VALID_FRACTION"
            ),
            "expected_model_digest": ast.get("model_digest"),
            "preprocessing_version": PREPROCESSING_VERSION,
        }

    def _start_batch(
        self,
        batch: Dict[str, Any],
        ast: Dict[str, Any],
        cfg: Dict[str, Any],
        lease_seconds: int,
        max_attempts: int,
    ) -> None:
        batch_id = batch["batch_id"]
        if not assessment_store.claim_batch_lease(batch_id, lease_seconds, conn=self.conn):
            return  # another worker took it

        try:
            batch_config = self._build_batch_config(ast, cfg, batch_id)
            storage = self._storage_for(ast["project_id"])
            data_type = self.config.get_metadata_types().AOI_ASSESSMENT_CONFIG.value
            storage.save(
                identifier=batch_id, data=batch_config, data_type=data_type, data_format="json"
            )
            config_url = storage.get_file_remote_path(batch_id, data_type, data_format="json")
            config_rel = f"inputs/{_uri_filename(config_url)}"
            input_files = {
                "config": {"http_url": _uri_without_query(config_url), "file_path": config_rel},
                "imagery": {
                    "http_url": _uri_without_query(batch_config["file_uri"]),
                    "file_path": batch_config["files"]["imagery"],
                },
            }

            task_id = f"{ASSESSMENT_TASK_PREFIX}-{MetadataUtils.generate_id()}"
            job_id = self.config.get_azure_batch_config()["training_batch_job_id"][:64]
            command = (
                f'"mkdir -p ${BATCH_JOB_WORKDIR} '
                f"&& cd ${BATCH_JOB_WORKDIR} "
                f"&& damage-triage-region-infer --config ${BATCH_JOB_WORKDIR}/{config_rel}"
                '"'
            )
            self.runner.add_task(
                job_id=job_id,
                task_id=task_id,
                output_prefix=f"{MetadataUtils.hash_string(ast['project_id'])}/{task_id}",
                resource_files_for_upload=input_files,
                file_pattern=f"${BATCH_JOB_WORKDIR}/outputs/*.*",
                command=command,
                image_name=self.config.get_azure_batch_config()["docker_image"],
            )
            assessment_store.update_batch_status(
                batch_id,
                BatchStatus.RUNNING.value,
                runner_job_id=job_id,
                runner_task_id=task_id,
                conn=self.conn,
            )
        except Exception as exc:
            self.logger.error("Failed to start batch %s: %s", batch_id, exc, exc_info=True)
            assessment_store.requeue_or_fail_batch(
                batch_id, max_attempts, f"task submission failed: {exc}", conn=self.conn
            )

    def _ingest_batch_outputs(
        self, batch: Dict[str, Any], ast: Dict[str, Any], max_attempts: int
    ) -> None:
        batch_id = batch["batch_id"]
        job_id, task_id = batch.get("runner_job_id"), batch["runner_task_id"]

        manifest_text = self.runner.get_filecontent_from_task(
            job_id=job_id, task_id=task_id, filename=MANIFEST_FILENAME
        )
        if not manifest_text:
            assessment_store.requeue_or_fail_batch(
                batch_id, max_attempts, f"{MANIFEST_FILENAME} missing from task output",
                conn=self.conn,
            )
            return
        manifest = json.loads(manifest_text)

        expected = ast.get("model_digest")
        if expected and manifest.get("model_digest") != expected:
            assessment_store.update_batch_status(
                batch_id,
                BatchStatus.FAILED.value,
                error=(
                    f"model digest {manifest.get('model_digest')} differs from the "
                    f"assessment's pinned {expected}"
                ),
                conn=self.conn,
            )
            return

        evidence_text = self.runner.get_filecontent_from_task(
            job_id=job_id, task_id=task_id, filename=EVIDENCE_FILENAME
        ) or ""
        rows = []
        for line in evidence_text.splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["assessment_id"] = self.assessment_id
                rows.append(rec)
        if rows:
            assessment_store.upsert_prediction_evidence(
                self.assessment_id, rows, conn=self.conn
            )
        assessment_store.apply_unit_results(
            batch_id, manifest.get("units", []), conn=self.conn
        )
        assessment_store.update_batch_status(
            batch_id, BatchStatus.COMPLETED.value, conn=self.conn
        )
        self.logger.info("Batch %s ingested: %d evidence rows", batch_id, len(rows))

    def _handle_aggregating(self, ast: Dict[str, Any]) -> None:
        image_layer_id = ast["image_layer_id"]
        buildings = assessment_store.eligible_buildings_with_overlap(
            image_layer_id, ast["aoi_geometry"], conn=self.conn
        )
        failed_ids = set(
            assessment_store.buildings_without_completed_units(self.assessment_id, conn=self.conn)
        )
        for b in buildings:
            if b["building_id"] in failed_ids:
                b["status"] = BuildingResultStatus.FAILED.value

        results = aggregate(
            evidence_rows=assessment_store.get_evidence(self.assessment_id, conn=self.conn),
            eligible_buildings=buildings,
            config=ast.get("config") or {},
            assessment_id=self.assessment_id,
            image_layer_id=image_layer_id,
            model_digest=ast.get("model_digest"),
            preprocessing_version=ast.get("preprocessing_version"),
        )
        assessment_store.upsert_damage_assessments(
            self.assessment_id, results, conn=self.conn
        )

        counts = assessment_store.unit_status_counts(self.assessment_id, conn=self.conn)
        n_failed_units = counts.get("failed", 0)
        n_failed_buildings = sum(
            1 for r in results if r["result_status"] == BuildingResultStatus.FAILED.value
        )
        final = (
            AoiAssessmentStatus.COMPLETED_WITH_ERRORS.value
            if n_failed_units or n_failed_buildings
            else AoiAssessmentStatus.COMPLETED.value
        )
        self._set_status(
            final,
            error=(
                f"{n_failed_units} processing window(s) failed; "
                f"{n_failed_buildings} building(s) could not be scored"
                if final == AoiAssessmentStatus.COMPLETED_WITH_ERRORS.value
                else None
            ),
            completed_units=counts.get("completed", 0),
            failed_units=n_failed_units,
        )
        self.logger.info("Assessment %s finished: %s", self.assessment_id, final)

    def _handle_cancelling(self, ast: Dict[str, Any]) -> None:
        for b in assessment_store.get_batches(self.assessment_id, conn=self.conn):
            if b["status"] == BatchStatus.RUNNING.value and b.get("runner_task_id"):
                try:
                    self.runner.cleanup_task(
                        job_id=b.get("runner_job_id"), task_id=b["runner_task_id"]
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Could not stop task %s: %s", b["runner_task_id"], exc
                    )
                assessment_store.update_batch_status(
                    b["batch_id"], BatchStatus.FAILED.value,
                    error="assessment cancelled", conn=self.conn,
                )
            elif b["status"] == BatchStatus.PENDING.value:
                assessment_store.update_batch_status(
                    b["batch_id"], BatchStatus.SKIPPED.value,
                    error="assessment cancelled", conn=self.conn,
                )
        assessment_store.update_assessment_status(
            self.assessment_id, AoiAssessmentStatus.CANCELLED.value, conn=self.conn
        )
