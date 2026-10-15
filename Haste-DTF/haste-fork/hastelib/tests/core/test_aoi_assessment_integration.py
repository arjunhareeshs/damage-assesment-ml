# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""End-to-end test of the large-AOI assessment against a real PostGIS.

Runs the real SQL (db/migrations/0000-0002), the real orchestrator state
machine and the real ``damage_triage_region_infer.run_batch_inference``
on a synthetic projected (UTM) GeoTIFF. Only three things are faked: the
queue, the task runner (executes the batch in-process instead of in a
container) and the model forward pass (deterministic probabilities, so no
checkpoint is needed).

Skipped unless a disposable PostGIS with the migrations applied is
reachable through HASTE_TEST_POSTGRES_* variables, e.g.:

    docker run -d --rm --name haste-aoi-test-pg -p 55432:5432 \\
      -e POSTGRES_USER=haste -e POSTGRES_PASSWORD=haste-test -e POSTGRES_DB=haste \\
      -v "$PWD/../db/migrations:/docker-entrypoint-initdb.d:ro" postgis/postgis:16-3.4
    HASTE_TEST_POSTGRES_HOST=localhost HASTE_TEST_POSTGRES_PORT=55432 \\
      HASTE_TEST_POSTGRES_PASSWORD=haste-test PYTHONPATH=src pytest \\
      tests/core/test_aoi_assessment_integration.py

The test truncates the hg_* tables: never point it at a real database.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PG_HOST = os.getenv("HASTE_TEST_POSTGRES_HOST")

try:  # torch is required by the region workflow module
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

LAYER = "layer-int-1"
PROJECT = "proj-int-1"
CLASS_FOR_TEST = 2  # every scored building is predicted "Total Roof Damage"


def _fake_probs(n):
    probs = np.full((n, 5), 0.05, dtype=np.float32)
    probs[:, CLASS_FOR_TEST] = 0.8
    return probs


class FakeQueue:
    messages = []

    def __init__(self, *args, **kwargs):
        pass

    def put_message(self, message, visibility_timeout=30):
        FakeQueue.messages.append(json.loads(message))


class FakeStorage:
    root = None

    def __init__(self, *args, **kwargs):
        pass

    def save(self, identifier, data_type, data=None, data_format="json", **kwargs):
        path = os.path.join(FakeStorage.root, f"{data_type}_{identifier}.{data_format}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def get_file_remote_path(self, identifier=None, data_type=None, data_format="json", **kwargs):
        # A query string like a SAS token, which the processor must strip.
        return os.path.join(FakeStorage.root, f"{data_type}_{identifier}.{data_format}") + "?sig=x"


class FakeRunner:
    """Runs the batch synchronously in-process, like a container would."""

    root = None
    fail_next = 0
    tasks = {}

    def __init__(self, *args, **kwargs):
        pass

    def add_task(self, job_id, task_id, resource_files_for_upload, command, **kwargs):
        from hastegeo.workflows import damage_triage_region_infer as region

        task_dir = os.path.join(FakeRunner.root, task_id)
        os.makedirs(os.path.join(task_dir, "inputs"), exist_ok=True)
        for spec in resource_files_for_upload.values():
            shutil.copy(spec["http_url"], os.path.join(task_dir, spec["file_path"]))
        config_rel = resource_files_for_upload["config"]["file_path"]
        assert config_rel in command
        with open(os.path.join(task_dir, config_rel), encoding="utf-8") as f:
            batch_config = json.load(f)

        if FakeRunner.fail_next > 0:
            FakeRunner.fail_next -= 1
            FakeRunner.tasks[task_id] = ("failed", task_dir)
            return
        with patch.object(region, "WORKDIR", task_dir), patch.object(
            region, "build_dtf_model", return_value=(object(), True)
        ), patch.object(
            region, "compute_model_digest", return_value="digest-test"
        ), patch.object(
            region, "score_window", side_effect=lambda **kw: _fake_probs(len(kw["masks"]))
        ), patch.object(region, "_find_local_checkpoint", return_value=("ckpt", "cfg")), \
                patch.object(region.os.path, "exists", side_effect=lambda p: True if p in ("ckpt", "cfg") else os.path.lexists(p)):
            region.run_batch_inference(batch_config)
        FakeRunner.tasks[task_id] = ("completed", task_dir)

    def get_task_status(self, job_id, task_id):
        from hastegeo.core.config import Config

        state = FakeRunner.tasks.get(task_id, ("running",))[0]
        if state == "completed":
            return Config.get_status_types().COMPLETED.value
        if state == "failed":
            return Config.get_status_types().FAILED.value
        return Config.get_status_types().IN_PROGRESS.value

    def get_filecontent_from_task(self, job_id, task_id, filename, as_chunk=False):
        path = os.path.join(FakeRunner.tasks[task_id][1], "outputs", filename)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    def cleanup_task(self, job_id, task_id):
        pass


@unittest.skipUnless(PG_HOST, "HASTE_TEST_POSTGRES_HOST not set (needs a disposable PostGIS)")
@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestAoiAssessmentEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.update(
            {
                "POSTGRES_HOST": PG_HOST,
                "POSTGRES_PORT": os.getenv("HASTE_TEST_POSTGRES_PORT", "5432"),
                "POSTGRES_DATABASE": os.getenv("HASTE_TEST_POSTGRES_DATABASE", "haste"),
                "POSTGRES_USER": os.getenv("HASTE_TEST_POSTGRES_USER", "haste"),
                "POSTGRES_PASSWORD": os.getenv("HASTE_TEST_POSTGRES_PASSWORD", ""),
                "POSTGRES_SSLMODE": "disable",
            }
        )

    def setUp(self):
        import rasterio
        from rasterio.transform import from_origin

        from hastegeo.core.utils.imagery_catalog import describe_raster, register_imagery_record
        from hastegeo.core.utils.postgis_connect import get_connection

        self.tmp = tempfile.mkdtemp()
        FakeStorage.root = self.tmp
        FakeRunner.root = self.tmp
        FakeRunner.fail_next = 0
        FakeRunner.tasks = {}
        FakeQueue.messages = []

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE hg_damage_assessments, hg_prediction_evidence, hg_unit_buildings, "
                "hg_processing_units, hg_batches, hg_assessments, hg_imagery, hg_buildings"
            )
        conn.commit()

        # 2048 x 2048 px at 1 m in UTM 14N (NAIP-like projected imagery).
        # The right-most 300 columns are black (nodata collar).
        self.origin_x, self.origin_y = 500000.0, 3900000.0
        transform = from_origin(self.origin_x, self.origin_y, 1.0, 1.0)
        data = np.full((3, 2048, 2048), 120, dtype=np.uint8)
        data[:, :, 1748:] = 0
        self.tif = os.path.join(self.tmp, "post_event.tif")
        with rasterio.open(
            self.tif, "w", driver="GTiff", width=2048, height=2048, count=3,
            dtype="uint8", crs="EPSG:32614", transform=transform,
        ) as dst:
            dst.write(data)

        self.imagery = register_imagery_record(
            LAYER, PROJECT, self.tif, describe_raster(self.tif), conn=conn
        )

        # Buildings, defined in pixel space then stored in EPSG:4326.
        def px_box(c0, r0, c1, r1):
            return self._utm_box_to_4326(
                self.origin_x + c0, self.origin_y - r1, self.origin_x + c1, self.origin_y - r0
            )

        self.buildings = {
            "inside": px_box(100, 100, 130, 130),       # wholly in window (0,0)
            "straddle": px_box(1010, 200, 1040, 230),   # crosses the col 1024 edge
            "boundary": px_box(1500, 1500, 1540, 1540), # cut by the AOI edge
            "outside": px_box(1900, 100, 1930, 130),    # outside the AOI
            "nodata": px_box(1760, 400, 1790, 430),     # inside AOI, on black pixels
        }
        with conn.cursor() as cur:
            for b_id, geom in self.buildings.items():
                cur.execute(
                    "INSERT INTO hg_buildings (building_id, image_layer_id, project_id, geometry, "
                    "footprint_source, footprint_version) VALUES "
                    "(%s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 4326), 'test', 'v1')",
                    (b_id, LAYER, PROJECT, geom.wkt),
                )
        conn.commit()
        conn.close()

        # AOI: pixel cols 0..1800, rows 0..1520 (cuts through "boundary").
        self.aoi = px_box(0, 0, 1800, 1520)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _utm_box_to_4326(minx, miny, maxx, maxy):
        import pyproj
        from shapely.geometry import box
        from shapely.ops import transform

        to_4326 = pyproj.Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)
        return transform(to_4326.transform, box(minx, miny, maxx, maxy))

    def _create(self, assessment_id, **cfg_overrides):
        from shapely.geometry import mapping

        from hastegeo.core.utils import assessment_store

        cfg = {
            "max_units": 50,
            "windows_per_batch": 2,
            "masks_per_forward": None,
            "max_concurrent_batches": 1,
            "max_attempts": 2,
            "lease_seconds": 600,
            "min_valid_pixel_fraction": 0.5,
            "normalization": "imagenet",
        }
        cfg.update(cfg_overrides)
        assessment_store.create_assessment(
            {
                "assessment_id": assessment_id,
                "project_id": PROJECT,
                "image_layer_id": LAYER,
                "aoi_geometry": mapping(self.aoi),
                "aoi_area_km2": 1.0,
                "config": cfg,
            }
        )

    def _run(self, assessment_id, max_steps=40):
        import hastegeo.core.processors.aoi_assessment as aoi_proc

        with patch.object(aoi_proc, "AzureQueueHandler", FakeQueue), patch.object(
            aoi_proc, "UnifiedRunner", FakeRunner
        ), patch.object(aoi_proc, "UnifiedDataLayer", FakeStorage), patch.object(
            aoi_proc, "resolve_checkpoint_digest", return_value="digest-test"
        ):
            for _ in range(max_steps):
                ast = aoi_proc.AoiAssessmentPostprocessor(assessment_id).process()
                if ast["status"] in (
                    "completed", "completed_with_errors", "failed", "cancelled", "rejected"
                ):
                    return ast
        self.fail(f"assessment did not finish in {max_steps} steps: {ast['status']}")

    def _results(self, assessment_id):
        from hastegeo.core.utils.postgis_connect import get_connection

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT building_id, damage_class, damage_class_name, result_status, "
                "boundary_intersecting, damage_percentage, model_digest, footprint_version "
                "FROM hg_damage_assessments WHERE assessment_id = %s",
                (assessment_id,),
            )
            rows = {r[0]: r[1:] for r in cur.fetchall()}
        conn.close()
        return rows

    def test_full_assessment(self):
        from hastegeo.core.utils import assessment_store

        self._create("ast-0001")
        ast = self._run("ast-0001")

        self.assertEqual(ast["status"], "completed", ast.get("error"))
        self.assertEqual(ast["eligible_buildings"], 4)
        self.assertEqual(ast["model_digest"], "digest-test")
        self.assertEqual(ast["failed_units"], 0)

        rows = self._results("ast-0001")
        # exact AOI eligibility: one row per eligible building, none outside
        self.assertEqual(set(rows), {"inside", "straddle", "boundary", "nodata"})
        for b_id in ("inside", "straddle", "boundary"):
            cls, name, status, _, pct, digest, fp_version = rows[b_id]
            self.assertEqual((cls, name, status), (CLASS_FOR_TEST, "Total Roof Damage", "completed"))
            self.assertIsNone(pct)
            self.assertEqual(digest, "digest-test")
            self.assertEqual(fp_version, "v1")
        self.assertTrue(rows["boundary"][3])
        self.assertFalse(rows["inside"][3])
        # nodata is never reported as Undamaged
        self.assertEqual(rows["nodata"][:3], (None, None, "insufficient_data"))

        # the straddling building was also scored whole from a centred window
        evidence = [e for e in assessment_store.get_evidence("ast-0001") if e["building_id"] == "straddle"]
        self.assertGreaterEqual(len(evidence), 2)
        self.assertTrue(any(e["coverage_fraction"] >= 0.9999 for e in evidence))

        # re-ingesting evidence does not duplicate rows
        before = len(assessment_store.get_evidence("ast-0001"))
        assessment_store.upsert_prediction_evidence("ast-0001", assessment_store.get_evidence("ast-0001"))
        self.assertEqual(len(assessment_store.get_evidence("ast-0001")), before)

        # a second assessment over the same AOI leaves the first untouched
        self._create("ast-0002")
        self._run("ast-0002")
        self.assertEqual(self._results("ast-0001"), rows)
        history = assessment_store.get_building_history(LAYER, "inside")
        self.assertEqual({h["assessment_id"] for h in history}, {"ast-0001", "ast-0002"})

        stats = assessment_store.get_assessment_statistics("ast-0001")
        self.assertEqual(stats["class_counts"], {"Total Roof Damage": 3})
        self.assertEqual(stats["status_counts"]["insufficient_data"], 1)

        feats = assessment_store.get_assessment_map_data(
            "ast-0001", bbox=self.aoi.bounds, zoom=16, min_zoom=14, max_features=2
        )
        self.assertTrue(feats["truncated"])
        self.assertEqual(len(feats["features"]), 2)

    def test_failed_task_is_retried(self):
        FakeRunner.fail_next = 1  # first task fails, its retry succeeds
        self._create("ast-0003", max_attempts=2)
        ast = self._run("ast-0003")
        self.assertEqual(ast["status"], "completed", ast.get("error"))
        self.assertEqual(self._results("ast-0003")["inside"][2], "completed")

    def test_exhausted_retries_end_completed_with_errors(self):
        FakeRunner.fail_next = 1
        self._create("ast-0004", max_attempts=1)
        ast = self._run("ast-0004")
        self.assertEqual(ast["status"], "completed_with_errors")
        statuses = {r[2] for r in self._results("ast-0004").values()}
        self.assertIn("failed", statuses)

    def test_unit_limit_rejects(self):
        self._create("ast-0005", max_units=1)
        ast = self._run("ast-0005")
        self.assertEqual(ast["status"], "rejected")
        self.assertIn("HASTE_AOI_MAX_UNITS", ast["error"])

    def test_missing_required_parameter_fails_clearly(self):
        self._create("ast-0006", windows_per_batch=None)
        with patch.dict(os.environ, {"HASTE_REGION_WINDOWS_PER_BATCH": ""}):
            ast = self._run("ast-0006")
        self.assertEqual(ast["status"], "failed")
        self.assertIn("HASTE_REGION_WINDOWS_PER_BATCH", ast["error"])

    def test_cancel(self):
        from hastegeo.core.utils import assessment_store

        self._create("ast-0007")
        assessment_store.update_assessment_status("ast-0007", "cancelling")
        ast = self._run("ast-0007")
        self.assertEqual(ast["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
