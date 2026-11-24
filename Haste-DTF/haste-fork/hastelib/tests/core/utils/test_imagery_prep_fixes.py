# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Imagery prep / footprint / matching fixes: valid-area extraction, mixed-CRS
mosaics, catalogue footprints, window coverage, oversized buildings, the
runner download cache, map endpoints and the footprint-load failure path."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, shape

UTM = "EPSG:32615"


def _write(path, data, transform, crs=UTM, nodata=None):
    count = data.shape[0]
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
        count=count, dtype=data.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


class TestExtractAoiPolygon(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.transform = from_origin(500000, 3300000, 1.0, 1.0)  # 1 m pixels

    def _path(self, name="a.tif"):
        return os.path.join(self.tmp.name, name)

    def test_decimated_read_still_covers_the_image(self):
        from hastegeo.core.utils.aoi import extract_aoi_polygon

        data = np.full((3, 600, 600), 120, dtype=np.uint8)
        p = _write(self._path(), data, self.transform)
        calls = []
        original = rasterio.io.DatasetReader.read

        def spy(self, *a, **k):
            calls.append(k.get("out_shape"))
            return original(self, *a, **k)

        with patch.object(rasterio.io.DatasetReader, "read", spy):
            poly = extract_aoi_polygon(p, max_dim=64)
        # every read was decimated to at most 64 px on a side
        self.assertTrue(calls)
        for shape_ in calls:
            self.assertIsNotNone(shape_)
            self.assertLessEqual(max(shape_[-2:]), 64)
        self.assertEqual(poly.geom_type, "Polygon")

    def test_declared_nodata_is_honoured(self):
        from hastegeo.core.utils.aoi import extract_aoi_polygon

        data = np.full((3, 200, 200), 7, dtype=np.uint8)
        data[:, :, 100:] = 255  # right half is nodata=255, not 0
        p = _write(self._path(), data, self.transform, nodata=255)
        poly = extract_aoi_polygon(p, simplify_tolerance_meters=0.5)
        # In 4326 the left half is roughly half the width of the full image.
        full = extract_aoi_polygon(
            _write(self._path("full.tif"), np.full((3, 200, 200), 7, np.uint8), self.transform),
            simplify_tolerance_meters=0.5,
        )
        self.assertAlmostEqual(poly.area / full.area, 0.5, delta=0.03)

    def test_large_gap_kept_as_hole_small_gap_filled(self):
        from hastegeo.core.utils.aoi import extract_aoi_polygon

        data = np.full((3, 400, 400), 90, dtype=np.uint8)
        data[:, 100:200, 100:200] = 0  # 100x100 m gap = 10,000 m^2 -> hole
        data[:, 300:310, 300:310] = 0  # 10x10 m speck -> filled
        p = _write(self._path(), data, self.transform)
        poly = extract_aoi_polygon(p, simplify_tolerance_meters=0.5)
        self.assertEqual(len(poly.interiors), 1)


class TestMixedCrsMosaic(unittest.TestCase):
    def test_tile_in_another_crs_is_reprojected_not_misplaced(self):
        from hastegeo.core.utils.imagery import ImageryUtils
        from rasterio.warp import transform_bounds

        with tempfile.TemporaryDirectory() as tmp:
            a = _write(os.path.join(tmp, "a.tif"), np.full((3, 100, 100), 50, np.uint8),
                       from_origin(500000, 3300000, 1.0, 1.0))
            # Neighbour 100 m east, delivered in EPSG:4326.
            west, south, east, north = transform_bounds(
                UTM, "EPSG:4326", 500100, 3299900, 500200, 3300000
            )
            px = (east - west) / 100
            b = _write(os.path.join(tmp, "b.tif"), np.full((3, 100, 100), 200, np.uint8),
                       from_origin(west, north, px, (north - south) / 100), crs="EPSG:4326")
            out = ImageryUtils.mosaic_imagery(
                [a, b], os.path.join(tmp, "m.tif"), "-of GTiff"
            )
            with rasterio.open(out) as m:
                self.assertEqual(m.crs.to_string(), UTM)
                self.assertGreater(m.bounds.right, 500190)  # b placed east of a
                self.assertLess(m.bounds.right, 500260)

    def test_file_without_crs_is_refused(self):
        from hastegeo.core.utils.imagery import ImageryUtils

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "nocrs.tif")
            with rasterio.open(p, "w", driver="GTiff", width=10, height=10, count=3,
                               dtype="uint8") as dst:
                dst.write(np.ones((3, 10, 10), np.uint8))
            with self.assertRaisesRegex(ValueError, "coordinate reference system"):
                ImageryUtils.mosaic_imagery([p], os.path.join(tmp, "m.tif"), "-of GTiff")


class TestCatalogueFootprint(unittest.TestCase):
    def test_footprint_excludes_nodata_collar(self):
        from hastegeo.core.utils.imagery_catalog import describe_raster

        with tempfile.TemporaryDirectory() as tmp:
            data = np.zeros((3, 200, 200), np.uint8)
            data[:, :, :100] = 80  # only the west half holds imagery
            p = _write(os.path.join(tmp, "a.tif"), data, from_origin(500000, 3300000, 1, 1))
            meta = describe_raster(p)
        fp = shape(meta["footprint"])
        from rasterio.warp import transform_bounds

        w, s, e, n = transform_bounds(UTM, "EPSG:4326", 500000, 3299800, 500200, 3300000)
        self.assertLess(fp.area, 0.6 * box(w, s, e, n).area)


class TestSelectUnits(unittest.TestCase):
    def _bldg(self, x0, y0, x1, y1):
        import pyproj
        from shapely.ops import transform

        to4326 = pyproj.Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform
        return transform(to4326, box(x0, y0, x1, y1))

    def test_oversized_building_gets_a_whole_building_window(self):
        from hastegeo.core.utils.region_windows import select_units

        t = from_origin(500000, 3303000, 1.0, 1.0)
        big = self._bldg(500100, 3300200, 501600, 3301700)  # 1500 m square
        aoi = self._bldg(499000, 3299000, 504000, 3304000)
        units = select_units("img", 3000, 3000, t, aoi, [{"building_id": "big", "geometry": big}],
                             raster_crs=UTM)
        over = [u for u in units if u["kind"] == "building_oversize"]
        self.assertEqual(len(over), 1)
        self.assertGreater(over[0]["width"], 1024)
        self.assertAlmostEqual(over[0]["buildings"][0]["coverage_fraction"], 1.0, places=3)

    def test_coverage_measured_in_raster_crs(self):
        from hastegeo.core.utils.region_windows import select_units

        t = from_origin(500000, 3302048, 1.0, 1.0)
        # Building straddles the x=1024 px window edge: 30 m in window 0, 10 m in window 1.
        b = self._bldg(500994, 3301500, 501034, 3301510)
        aoi = self._bldg(499000, 3299000, 504000, 3304000)
        units = select_units("img", 2048, 2048, t, aoi, [{"building_id": "b", "geometry": b}],
                             raster_crs=UTM)
        grid = {u["col_off"]: u["buildings"][0]["coverage_fraction"]
                for u in units if u["kind"] == "grid"}
        self.assertAlmostEqual(grid[0], 0.75, places=3)
        self.assertAlmostEqual(grid[1024], 0.25, places=3)


try:
    import azure.storage.blob  # noqa: F401

    _HAS_AZURE = True
except ImportError:  # the runner imports the Azure SDK at module load
    _HAS_AZURE = False


@unittest.skipUnless(_HAS_AZURE, "azure-storage-blob not installed")
def _has_module(name):
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(
    _has_module("docker") and _has_module("azure.storage.queue"),
    "LocalRunner needs the docker and azure-storage-queue SDKs",
)
class TestRunnerCache(unittest.TestCase):
    def test_one_download_shared_by_tasks(self):
        from pathlib import Path

        from hastegeo.core.runners.local import LocalRunner

        with tempfile.TemporaryDirectory() as tmp:
            runner = LocalRunner.__new__(LocalRunner)
            runner.blob_client = object()
            runner.cache_dir = Path(tmp) / "cache"
            runner.cache_max_age_s = 3600.0
            calls = []

            def fake_download(url, target):
                calls.append(url)
                Path(target).write_bytes(b"tif-bytes")

            runner._download_blob_to = fake_download
            url = "http://azurite:10000/devstoreaccount1/data/x/mosaic.tif"
            t1, t2 = Path(tmp) / "t1" / "a.tif", Path(tmp) / "t2" / "a.tif"
            t1.parent.mkdir()
            t2.parent.mkdir()
            runner._fetch_cached(url, t1)
            runner._fetch_cached(url, t2)
            self.assertEqual(calls, [url])
            self.assertEqual(t2.read_bytes(), b"tif-bytes")


def _conn():
    connection, cursor = MagicMock(), MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection, cursor


class TestMapEndpoints(unittest.TestCase):
    def test_classified_buildings_bounded_and_ordered(self):
        from hastegeo.core.utils.postgis_query import get_classified_buildings_geojson

        conn, cur = _conn()
        row = ("b", json.dumps({"type": "Point", "coordinates": [0, 0]}), 3, "x", 0.9,
               "pending_review", "ast-1", True)
        cur.fetchall.return_value = [row, row, row]
        out = get_classified_buildings_geojson("L", bbox=(0, 0, 1, 1), limit=2, conn=conn)
        sql, params = cur.execute.call_args.args
        self.assertIn("ST_MakeEnvelope", sql)
        self.assertIn("ORDER BY damage_class DESC", sql)
        self.assertEqual(params[-1], 3)  # limit + 1 to detect truncation
        self.assertTrue(out["truncated"])
        self.assertEqual(len(out["features"]), 2)
        props = out["features"][0]["properties"]
        self.assertEqual(props["assessment_id"], "ast-1")
        self.assertTrue(props["result_out_of_date"])

    def test_map_data_is_ordered_and_carries_review(self):
        from hastegeo.core.utils import assessment_store

        conn, cur = _conn()
        assessment_store._REVIEW_SCHEMA_READY = True
        cur.fetchall.return_value = [(
            "b", 2, "n", 0.8, "completed", False,
            json.dumps({"type": "Point", "coordinates": [0, 0]}),
            "accepted", "me", None, "ast-1", "L",
        )]
        out = assessment_store.get_assessment_map_data("ast-1", bbox=(0, 0, 1, 1), zoom=18, conn=conn)
        sql = cur.execute.call_args.args[0]
        self.assertIn("ORDER BY d.damage_class DESC", sql)
        p = out["features"][0]["properties"]
        self.assertEqual((p["review_status"], p["assessment_id"]), ("accepted", "ast-1"))

    def test_clusters_group_by_centroid_not_polygon(self):
        """Snapping a footprint polygon to the grid collapses it to EMPTY and
        put every building in one cluster; the centroid must be snapped."""
        from hastegeo.core.utils import assessment_store

        conn, cur = _conn()
        assessment_store._REVIEW_SCHEMA_READY = True
        cur.fetchall.return_value = []
        assessment_store.get_assessment_map_data("ast-1", bbox=(0, 0, 1, 1), zoom=10, conn=conn)
        sql, params = cur.execute.call_args.args
        self.assertIn("ST_SnapToGrid(ST_Centroid(b.geometry)", sql)
        self.assertAlmostEqual(params[-2], 360.0 / 2 ** 12)

    def test_history_carries_review(self):
        from hastegeo.core.utils import assessment_store

        conn, cur = _conn()
        assessment_store._REVIEW_SCHEMA_READY = True
        cur.fetchall.return_value = [("ast-1", "n", None, 1, "x", 0.5, "completed", "u",
                                      "rejected", "me", None)]
        h = assessment_store.get_building_history("L", "b", conn=conn)
        self.assertEqual(h[0]["review_status"], "rejected")


class TestFootprintLoadFailureIsReported(unittest.TestCase):
    def test_db_failure_is_recorded_on_layer_not_only_logged(self):
        from hastegeo.core.config import Config
        from hastegeo.core.models.projects import ImageLayer, ImageryPreprocessJob

        with patch("hastegeo.core.processors.imagery.UnifiedDataLayer", autospec=True), \
                patch("hastegeo.core.processors.imagery.UnifiedRunner", autospec=True), \
                patch("hastegeo.core.processors.imagery.AzureQueueHandler", autospec=True):
            from hastegeo.core.processors.imagery import ImageryPostProcessor

            layer = ImageLayer(
                imageLayerId="L", projectId="P",
                status=Config().get_status_types().COMPLETED.value,
                preprocessJob=ImageryPreprocessJob(jobId="j", taskId="t", imageLayerId="L", projectId="P"),
            )
            proc = ImageryPostProcessor(image_data=layer)
        proc.runner.get_filecontent_from_task.return_value = [b"gpkg"]
        with patch("hastegeo.core.processors.imagery.postgis_load.load_footprints_into_postgis",
                   side_effect=RuntimeError("db down")):
            err = proc._load_footprints_into_postgis("fp.gpkg")
        self.assertIn("db down", err)

        with patch("hastegeo.core.processors.imagery.postgis_load.load_footprints_into_postgis",
                   return_value=12):
            self.assertEqual(proc._load_footprints_into_postgis("fp.gpkg"), "")
        self.assertEqual(proc.image_data.buildingCount, 12)


if __name__ == "__main__":
    unittest.main()
