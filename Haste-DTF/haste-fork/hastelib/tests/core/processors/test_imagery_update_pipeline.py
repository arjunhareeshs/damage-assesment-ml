# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Imagery updates through the pipeline: job config, results, failure,
building results and the catalogue (mocked storage, queue and database)."""

import unittest
from unittest.mock import MagicMock, patch

from hastegeo.core.config import Config
from hastegeo.core.models.projects import ImageLayer, ImageryPreprocessJob


class _Stop(Exception):
    pass


def _processor(layer):
    with patch("hastegeo.core.processors.imagery.UnifiedDataLayer", autospec=True), \
            patch("hastegeo.core.processors.imagery.UnifiedRunner", autospec=True), \
            patch("hastegeo.core.processors.imagery.AzureQueueHandler", autospec=True):
        from hastegeo.core.processors.imagery import ImageryPostProcessor

        return ImageryPostProcessor(image_data=layer)


def _layer(**kw):
    base = dict(
        imageLayerId="layer-1", projectId="proj-1",
        postEventImageryUrls=["https://h/a.tif", "https://h/b.tif"],
        sourceTypePostEvent="rgb/no_processing",
        status=Config().get_status_types().IN_PROGRESS.value,
        totalSteps=4,  # set by queue_for_processing on every real queued layer
        preprocessJob=ImageryPreprocessJob(jobId="j", taskId="t", imageLayerId="layer-1", projectId="proj-1"),
    )
    base.update(kw)
    return ImageLayer(**base)


REPORT = {
    "existing": [
        {"id": 0, "url": "https://h/a.tif", "action": "retire"},
        {"id": 1, "url": "https://h/b.tif", "action": "keep_under_new"},
    ],
    "incoming": [{
        "id": "new0", "url": "https://h/new.tif", "file": "new.tif", "area_km2": 0.5,
        "new_area_share": 0.0,
        "corresponds_to": [{"id": 0, "share_of_new": 0.8, "share_of_existing": 1.0},
                           {"id": 1, "share_of_new": 0.2, "share_of_existing": 0.3}],
        "registration": {"applied": True, "dx_m": -1.2, "dy_m": 0.7},
    }],
    "retired_urls": ["https://h/a.tif"],
    "new_urls": ["https://h/new.tif"],
    "updated_area": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
    "updated_area_km2": 0.5,
}


class TestJobConfig(unittest.TestCase):
    def _config(self, layer):
        proc = _processor(layer)
        captured = {}

        def save(**kwargs):
            captured.update(kwargs)
            raise _Stop()

        proc.storage.save = MagicMock(side_effect=save)
        with self.assertRaises(_Stop):
            proc._execute_image_preprocess()
        return captured["data"]

    def test_update_urls_reach_the_job(self):
        cfg = self._config(_layer(pendingImageryUpdateUrls=["https://h/new.tif"]))
        self.assertEqual(cfg["imagery_update"], {"new_post_event_imagery_urls": ["https://h/new.tif"]})
        # The current imagery stays the base the update is applied to.
        self.assertEqual(cfg["post_event_imagery_urls"], ["https://h/a.tif", "https://h/b.tif"])

    def test_normal_run_has_no_update(self):
        self.assertNotIn("imagery_update", self._config(_layer()))


class TestApplyUpdate(unittest.TestCase):
    @patch("hastegeo.core.processors.imagery.postgis_load.mark_imagery_updated",
           return_value={"buildings": 12, "results_out_of_date": 5})
    def test_urls_history_and_buildings(self, mark):
        layer = _layer(pendingImageryUpdateUrls=["https://h/new.tif"])
        proc = _processor(layer)
        proc._apply_imagery_update(REPORT)
        self.assertEqual(layer.postEventImageryUrls, ["https://h/b.tif", "https://h/new.tif"])
        self.assertEqual(layer.retiredPostEventImageryUrls, ["https://h/a.tif"])
        self.assertIsNone(layer.pendingImageryUpdateUrls)
        entry = layer.imageryUpdates[-1]
        self.assertEqual(entry["newUrls"], ["https://h/new.tif"])
        self.assertEqual(entry["retiredUrls"], ["https://h/a.tif"])
        self.assertEqual(entry["keptUnderNewUrls"], ["https://h/b.tif"])
        self.assertEqual(entry["resultsOutOfDate"], 5)
        # Each new file names the existing files it corresponds to, by URL.
        self.assertEqual(
            [c["url"] for c in entry["files"][0]["corresponds_to"]],
            ["https://h/a.tif", "https://h/b.tif"],
        )
        mark.assert_called_once_with("layer-1", REPORT["updated_area"])
        # The result is still a valid ImageLayer (it is saved as JSON).
        ImageLayer(**layer.dict())

    @patch("hastegeo.core.processors.imagery.postgis_load.mark_imagery_updated",
           side_effect=RuntimeError("db down"))
    def test_database_error_does_not_lose_the_update(self, _):
        layer = _layer(pendingImageryUpdateUrls=["https://h/new.tif"])
        _processor(layer)._apply_imagery_update(REPORT)
        self.assertEqual(layer.postEventImageryUrls, ["https://h/b.tif", "https://h/new.tif"])


class TestFailedUpdate(unittest.TestCase):
    def test_layer_keeps_working_imagery(self):
        statuses = Config().get_status_types()
        layer = _layer(pendingImageryUpdateUrls=["https://h/new.tif"])
        proc = _processor(layer)
        proc.runner.get_task_status.return_value = statuses.FAILED.value
        proc.runner.get_filecontent_from_task.return_value = None
        out = proc.process()
        self.assertEqual(out.status, statuses.COMPLETED.value)
        self.assertEqual(out.postEventImageryUrls, ["https://h/a.tif", "https://h/b.tif"])
        self.assertIsNone(out.pendingImageryUpdateUrls)
        self.assertIn("keeps its previous imagery", out.statusMessage)

    def test_failed_first_processing_still_fails(self):
        statuses = Config().get_status_types()
        proc = _processor(_layer())
        proc.runner.get_task_status.return_value = statuses.FAILED.value
        proc.runner.get_filecontent_from_task.return_value = None
        self.assertEqual(proc.process().status, statuses.FAILED.value)


def _conn():
    connection, cursor = MagicMock(), MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection, cursor


class TestBuildingResultsKept(unittest.TestCase):
    def _load(self, preserve):
        import os
        import tempfile

        import geopandas as gpd
        from shapely.geometry import box

        from hastegeo.core.utils import postgis_load

        gdf = gpd.GeoDataFrame({"id": ["b1", "b2"], "subtype": [None, None], "class": [None, None]},
                               geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs=4326)
        conn, cur = _conn()
        with tempfile.TemporaryDirectory() as tmp, patch("psycopg2.extras.execute_values"):
            path = os.path.join(tmp, "x.gpkg")
            gdf.to_file(path, driver="GPKG")
            postgis_load.load_footprints_into_postgis(_layer(), path, conn=conn,
                                                      preserve_results=preserve)
        return [c.args for c in cur.execute.call_args_list]

    def test_load_only_removes_buildings_that_are_gone(self):
        for preserve in (True, False):
            calls = self._load(preserve)
            delete = next(a for a in calls if a[0].strip().startswith("DELETE"))
            self.assertIn("NOT EXISTS", delete[0])
            self.assertEqual(delete[1], ("layer-1",))

    def test_results_are_flagged_not_wiped(self):
        calls = self._load(preserve=False)
        upsert = next(a for a in calls if "INSERT INTO hg_buildings" in a[0])
        self.assertIn("imagery_updated_at", upsert[0])
        self.assertNotIn("damage_class", upsert[0])

    def test_mark_imagery_updated_counts_out_of_date_results(self):
        from hastegeo.core.utils.postgis_load import mark_imagery_updated

        conn, cur = _conn()
        cur.fetchall.return_value = [(True,), (False,), (True,)]
        out = mark_imagery_updated("layer-1", REPORT["updated_area"], conn=conn)
        self.assertEqual(out, {"buildings": 3, "results_out_of_date": 2})
        sql = " ".join(c.args[0] for c in cur.execute.call_args_list)
        self.assertIn("imagery_updated_at", sql)
        conn.commit.assert_called_once()


class TestCatalogue(unittest.TestCase):
    def test_deactivate_keeps_rows(self):
        from hastegeo.core.utils import imagery_catalog

        conn, cur = _conn()
        cur.rowcount = 4
        self.assertEqual(imagery_catalog.deactivate_layer_imagery("layer-1", conn=conn), 4)
        sql = " ".join(c.args[0] for c in cur.execute.call_args_list)
        self.assertIn("SET active = FALSE", sql)
        self.assertNotIn("DELETE", sql)

    def test_assessments_read_only_active_rows(self):
        from hastegeo.core.utils import imagery_catalog

        conn, cur = _conn()
        cur.fetchall.return_value = []
        imagery_catalog.imagery_for_aoi("layer-1", REPORT["updated_area"], conn=conn)
        self.assertIn("AND active", cur.execute.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()
