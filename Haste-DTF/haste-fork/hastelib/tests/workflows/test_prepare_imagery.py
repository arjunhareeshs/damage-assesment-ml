# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the imageryprep workflow's building-footprint step.

The full workflow exercises GDAL/rasterio against real GeoTIFFs, so these
tests focus on the new ``ImageryWorkflow.download_building_footprints()``
step in isolation by mocking out the AOI extractor and the subprocess that
runs the actual Overture download. The goals:

- (a) the prefix/filename uses the BUILDING_FOOTPRINTS ArtifactType template
  substituted with the layer's IDs;
- (b) the subprocess invocation passes the right CLI args;
- (c) any subprocess failure (non-zero exit, timeout, SIGSEGV) does not
  raise out of ``download_building_footprints`` — instead the failure is
  captured on ``building_footprints_error`` so the parent workflow can
  still finish writing the manifest, and ImageryPostProcessor can mark
  the image layer FAILED in the UI with that message;
- (d) the step records an error when no post-event mosaic is present
  rather than silently no-oping;
- (e) the AOI polygon is persisted as the valid-area-mask GeoJSON file
  so the UI can offer it as a downloadable artifact.
"""

import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import shapely.geometry
from hastegeo.workflows.prepare_imagery import ImageryWorkflow


def _fake_polygon():
    """A small shapely polygon to stand in for extract_aoi_polygon()."""
    return shapely.geometry.Polygon(
        [
            (-156.70, 20.87),
            (-156.66, 20.87),
            (-156.66, 20.89),
            (-156.70, 20.89),
        ]
    )


class TestDownloadBuildingFootprintsStep(unittest.TestCase):
    """Unit tests for ImageryWorkflow.download_building_footprints()."""

    def _make_workflow(self, dst_directory):
        wf = ImageryWorkflow(
            project_id="proj-1",
            image_layer_id="layer-9",
            dst_directory=dst_directory,
        )
        # Pretend process_post_event ran and populated the mosaic path.
        # The path doesn't have to be a real COG because we mock the AOI
        # extractor; it just needs to exist so the existence check passes.
        mosaic = os.path.join(dst_directory, "fake_post_event_mosaic.tif")
        with open(mosaic, "wb") as f:
            f.write(b"")
        wf.mosaic_post_event_tif_filepath = mosaic
        return wf

    @patch("subprocess.run")
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_writes_gpkg_with_layer_keyed_filename(
        self, mock_polygon, mock_run
    ):
        mock_polygon.return_value = _fake_polygon()

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            expected_path = os.path.join(
                tmp, "building_footprints_proj-1_layer-9.gpkg"
            )

            # Have the "subprocess" actually create the output file so the
            # path-existence check inside download_building_footprints passes.
            def fake_run(cmd, **_):
                with open(expected_path, "wb") as f:
                    f.write(b"GPKG-stub")
                return MagicMock(returncode=0, stdout="42\n", stderr="")

            mock_run.side_effect = fake_run

            wf.download_building_footprints()

            self.assertEqual(wf.building_footprints_path, expected_path)
            self.assertEqual(wf.building_footprints_error, "")
            mock_polygon.assert_called_once_with(
                wf.mosaic_post_event_tif_filepath
            )
            # Subprocess called with the expected CLI shape
            self.assertTrue(mock_run.called)
            cmd = mock_run.call_args.args[0]
            self.assertIn("hastegeo.core.utils.footprints", cmd)
            # Bbox derived from the polygon's bounds
            self.assertIn("--bbox=-156.7,20.87,-156.66,20.89", cmd)
            self.assertIn("--output-path", cmd)
            self.assertIn(expected_path, cmd)
            self.assertIn("--overwrite", cmd)
            # The AOI geojson was saved before the subprocess ran, so
            # its path should be forwarded to the downloader for
            # corner-cropping the bbox-only Overture result.
            mask_path = os.path.join(
                tmp, "valid_area_mask_proj-1_layer-9.geojson"
            )
            self.assertIn(f"--aoi-geojson={mask_path}", cmd)

    @patch("subprocess.run")
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_writes_valid_area_mask_geojson(self, mock_polygon, mock_run):
        # The AOI polygon should be persisted to a GeoJSON file using the
        # VALID_AREA_MASK prefix so the UI can offer it for download.
        polygon = _fake_polygon()
        mock_polygon.return_value = polygon
        mock_run.return_value = MagicMock(
            returncode=0, stdout="0\n", stderr=""
        )

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            expected_mask = os.path.join(
                tmp, "valid_area_mask_proj-1_layer-9.geojson"
            )
            # Subprocess "succeeds" by creating the gpkg.
            with open(
                os.path.join(tmp, "building_footprints_proj-1_layer-9.gpkg"),
                "wb",
            ) as f:
                f.write(b"GPKG-stub")

            wf.download_building_footprints()

            self.assertEqual(wf.valid_area_mask_path, expected_mask)
            self.assertEqual(wf.valid_area_mask_error, "")
            self.assertTrue(os.path.exists(expected_mask))
            with open(expected_mask) as f:
                fc = json.load(f)
            self.assertEqual(fc["type"], "FeatureCollection")
            self.assertEqual(len(fc["features"]), 1)
            self.assertEqual(fc["features"][0]["geometry"]["type"], "Polygon")

    @patch("subprocess.run")
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_mask_survives_footprint_subprocess_failure(
        self, mock_polygon, mock_run
    ):
        # Mask is written before the subprocess runs, so a subprocess
        # failure should still leave the mask in place.
        mock_polygon.return_value = _fake_polygon()
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="boom"
        )

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.download_building_footprints()

            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("exit code 1", wf.building_footprints_error)
            # Mask was extracted/saved before the subprocess ran.
            self.assertTrue(wf.valid_area_mask_path)
            self.assertTrue(os.path.exists(wf.valid_area_mask_path))
            self.assertEqual(wf.valid_area_mask_error, "")

    @patch("subprocess.run")
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_subprocess_nonzero_exit_records_error(
        self, mock_polygon, mock_run
    ):
        mock_polygon.return_value = _fake_polygon()
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="boom",
        )

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.download_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("exit code 1", wf.building_footprints_error)
            self.assertIn("boom", wf.building_footprints_error)

    @patch("subprocess.run")
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_subprocess_segfault_records_error(self, mock_polygon, mock_run):
        mock_polygon.return_value = _fake_polygon()
        # SIGSEGV manifests as returncode -11 from subprocess.run
        mock_run.return_value = MagicMock(
            returncode=-11,
            stdout="",
            stderr="qemu: uncaught target signal 11",
        )

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.download_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("exit code -11", wf.building_footprints_error)

    @patch("subprocess.run")
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_subprocess_timeout_records_error(self, mock_polygon, mock_run):
        mock_polygon.return_value = _fake_polygon()
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["python"], timeout=1
        )

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.download_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("timed out", wf.building_footprints_error)

    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    def test_aoi_failure_records_error(self, mock_polygon):
        mock_polygon.side_effect = RuntimeError("rasterio could not open")
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.download_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("area-of-interest", wf.building_footprints_error)
            # The same AOI failure also bubbles into the mask error so
            # the UI can surface a clear cause for the missing mask.
            self.assertEqual(wf.valid_area_mask_path, "")
            self.assertIn("area-of-interest", wf.valid_area_mask_error)

    def test_skips_when_no_post_event_mosaic_records_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = ImageryWorkflow(
                project_id="proj-1",
                image_layer_id="layer-9",
                dst_directory=tmp,
            )
            # No mosaic path set
            wf.download_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("post-event mosaic", wf.building_footprints_error)
            self.assertEqual(wf.valid_area_mask_path, "")
            self.assertIn("post-event mosaic", wf.valid_area_mask_error)


class TestProcessUserBuildingFootprintsStep(unittest.TestCase):
    """Unit tests for ImageryWorkflow.process_user_building_footprints().

    Mirrors TestDownloadBuildingFootprintsStep but for the user-supplied
    GPKG path. Covers:
      - happy path: validates the URL, derives AOI, calls the clip helper,
        and writes the BUILDING_FOOTPRINTS-templated output;
      - URL re-validation inside the workflow (defense-in-depth);
      - download failure / clip failure / no-mosaic / AOI extract failure
        all soft-failed (captured on building_footprints_error);
      - cross-host redirect refused;
      - max-bytes cap enforced.
    """

    _GOOD_URL = "https://example.blob.core.windows.net/c/footprints.gpkg?sv=...&sig=..."

    def _make_workflow(self, dst_directory, *, with_mosaic=True):
        wf = ImageryWorkflow(
            project_id="proj-1",
            image_layer_id="layer-9",
            dst_directory=dst_directory,
            user_building_footprints_url=self._GOOD_URL,
        )
        if with_mosaic:
            mosaic = os.path.join(dst_directory, "fake_post_event_mosaic.tif")
            with open(mosaic, "wb") as f:
                f.write(b"")
            wf.mosaic_post_event_tif_filepath = mosaic
        return wf

    def _expected_output(self, tmp):
        return os.path.join(tmp, "building_footprints_proj-1_layer-9.gpkg")

    @patch(
        "hastegeo.workflows.prepare_imagery.ImageryWorkflow."
        "_download_user_footprints_to"
    )
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.footprints.clip_and_normalize_user_footprints")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_happy_path_writes_clipped_gpkg(
        self,
        mock_validate,
        mock_clip,
        mock_aoi,
        mock_download,
    ):
        mock_aoi.return_value = MagicMock()  # shapely polygon stand-in
        mock_clip.return_value = 17  # feature count

        def fake_download(url, output_path, **_):
            with open(output_path, "wb") as f:
                f.write(b"GPKG-stub")

        mock_download.side_effect = fake_download

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.process_user_building_footprints()

            self.assertEqual(
                wf.building_footprints_path,
                self._expected_output(tmp),
            )
            self.assertEqual(wf.building_footprints_error, "")
            mock_validate.assert_called_once_with(self._GOOD_URL)
            mock_aoi.assert_called_once_with(wf.mosaic_post_event_tif_filepath)
            mock_clip.assert_called_once()
            kwargs = mock_clip.call_args.kwargs
            self.assertEqual(kwargs["output_path"], self._expected_output(tmp))
            self.assertTrue(kwargs["overwrite"])

    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_invalid_url_records_error_without_download(
        self, mock_validate, mock_aoi
    ):
        mock_validate.side_effect = ValueError("not allowlisted")
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.process_user_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("not allowed", wf.building_footprints_error)
            self.assertIn("not allowlisted", wf.building_footprints_error)
            # AOI extraction must not run if validation already failed.
            mock_aoi.assert_not_called()

    def test_skips_when_no_post_event_mosaic_records_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp, with_mosaic=False)
            wf.process_user_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("post-event mosaic", wf.building_footprints_error)
            # Mask should mirror the same failure so the UI surfaces it.
            self.assertEqual(wf.valid_area_mask_path, "")
            self.assertIn("post-event mosaic", wf.valid_area_mask_error)

    @patch(
        "hastegeo.workflows.prepare_imagery.ImageryWorkflow."
        "_download_user_footprints_to"
    )
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.footprints.clip_and_normalize_user_footprints")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_writes_valid_area_mask_geojson(
        self, mock_validate, mock_clip, mock_aoi, mock_download
    ):
        # User-footprints path must persist the AOI polygon as the
        # valid-area mask, same as the Overture path. Without this the
        # UI's "Download Valid Area Mask" stays greyed out whenever a
        # custom footprints URL is used.
        polygon = shapely.geometry.Polygon(
            [(-1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 1.0)]
        )
        mock_aoi.return_value = polygon
        mock_clip.return_value = 5

        def fake_download(url, output_path, **_):
            with open(output_path, "wb") as f:
                f.write(b"GPKG-stub")

        mock_download.side_effect = fake_download

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            expected_mask = os.path.join(
                tmp, "valid_area_mask_proj-1_layer-9.geojson"
            )
            wf.process_user_building_footprints()

            self.assertEqual(wf.valid_area_mask_path, expected_mask)
            self.assertEqual(wf.valid_area_mask_error, "")
            self.assertTrue(os.path.exists(expected_mask))
            with open(expected_mask) as f:
                fc = json.load(f)
            self.assertEqual(fc["type"], "FeatureCollection")
            self.assertEqual(len(fc["features"]), 1)
            self.assertEqual(fc["features"][0]["geometry"]["type"], "Polygon")

    @patch(
        "hastegeo.workflows.prepare_imagery.ImageryWorkflow."
        "_download_user_footprints_to"
    )
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_mask_survives_user_footprint_download_failure(
        self, mock_validate, mock_aoi, mock_download
    ):
        # Mask is written *before* the user-footprint download is
        # attempted, so a download failure should still leave the mask
        # in place — same invariant as the Overture path.
        polygon = shapely.geometry.Polygon(
            [(-1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 1.0)]
        )
        mock_aoi.return_value = polygon
        mock_download.side_effect = RuntimeError("HTTP 500")

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.process_user_building_footprints()

            # Footprints failed but mask survived
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("download", wf.building_footprints_error.lower())
            self.assertTrue(wf.valid_area_mask_path)
            self.assertTrue(os.path.exists(wf.valid_area_mask_path))
            self.assertEqual(wf.valid_area_mask_error, "")

    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_aoi_failure_records_error(self, mock_validate, mock_aoi):
        mock_aoi.side_effect = RuntimeError("rasterio could not open")
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.process_user_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("area-of-interest", wf.building_footprints_error)
            # AOI failure should also bubble into the mask error so the
            # UI can surface a clear cause for the missing mask.
            self.assertEqual(wf.valid_area_mask_path, "")
            self.assertIn("area-of-interest", wf.valid_area_mask_error)

    @patch(
        "hastegeo.workflows.prepare_imagery.ImageryWorkflow."
        "_download_user_footprints_to"
    )
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_download_failure_records_error(
        self, mock_validate, mock_aoi, mock_download
    ):
        mock_aoi.return_value = MagicMock()
        mock_download.side_effect = RuntimeError("HTTP 500")
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.process_user_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("download", wf.building_footprints_error.lower())
            self.assertIn("HTTP 500", wf.building_footprints_error)

    @patch(
        "hastegeo.workflows.prepare_imagery.ImageryWorkflow."
        "_download_user_footprints_to"
    )
    @patch("hastegeo.core.utils.aoi.extract_aoi_polygon")
    @patch("hastegeo.core.utils.footprints.clip_and_normalize_user_footprints")
    @patch("hastegeo.core.utils.url_allowlist.validate_footprint_url")
    def test_clip_failure_records_error(
        self,
        mock_validate,
        mock_clip,
        mock_aoi,
        mock_download,
    ):
        mock_aoi.return_value = MagicMock()

        def fake_download(url, output_path, **_):
            with open(output_path, "wb") as f:
                f.write(b"GPKG-stub")

        mock_download.side_effect = fake_download
        mock_clip.side_effect = ValueError("No features intersect the AOI")
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            wf.process_user_building_footprints()
            self.assertEqual(wf.building_footprints_path, "")
            self.assertIn("process", wf.building_footprints_error.lower())
            self.assertIn(
                "No features intersect the AOI",
                wf.building_footprints_error,
            )

    def test_download_refuses_cross_host_redirect(self):
        from unittest.mock import MagicMock as MM

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            output_path = os.path.join(tmp, "stage.gpkg")

            redirect_resp = MM()
            redirect_resp.status_code = 302
            redirect_resp.headers = {"Location": "https://evil.example/x"}

            with patch("requests.get", return_value=redirect_resp) as mock_get:
                with self.assertRaises(RuntimeError) as cm:
                    wf._download_user_footprints_to(
                        self._GOOD_URL,
                        output_path,
                        max_bytes=10_000,
                        timeout_seconds=30,
                    )
                self.assertIn("Redirects", str(cm.exception))
                kwargs = mock_get.call_args.kwargs
                self.assertFalse(kwargs.get("allow_redirects", True))

    def test_download_enforces_max_bytes(self):
        # The download streams (``iter_content``) and counts bytes as they
        # arrive, so the cap holds even when Content-Length is absent.
        from unittest.mock import MagicMock as MM

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            output_path = os.path.join(tmp, "stage.gpkg")

            resp = MM()
            resp.status_code = 200
            resp.raise_for_status = MM(return_value=None)
            resp.headers = {}  # no Content-Length: only the streamed count can catch it
            resp.iter_content = MM(return_value=[b"\x00" * 600, b"\x00" * 600])  # 1200 > 1000 cap

            with patch("requests.get", return_value=resp):
                with self.assertRaises(RuntimeError) as cm:
                    wf._download_user_footprints_to(
                        self._GOOD_URL,
                        output_path,
                        max_bytes=1000,
                        timeout_seconds=30,
                    )
                self.assertIn("max size", str(cm.exception))
                # File must not have been written when the size check fails
                self.assertFalse(os.path.exists(output_path))

    def test_download_writes_file_within_cap(self):
        # Sanity: a normal in-cap response actually writes the bytes
        # to disk.
        from unittest.mock import MagicMock as MM

        with tempfile.TemporaryDirectory() as tmp:
            wf = self._make_workflow(tmp)
            output_path = os.path.join(tmp, "stage.gpkg")

            payload = b"GPKG-stub-bytes"
            resp = MM()
            resp.status_code = 200
            resp.raise_for_status = MM(return_value=None)
            resp.headers = {"Content-Length": str(len(payload))}
            resp.iter_content = MM(return_value=[payload[:5], payload[5:]])

            with patch("requests.get", return_value=resp) as mock_get:
                wf._download_user_footprints_to(
                    self._GOOD_URL,
                    output_path,
                    max_bytes=10_000,
                    timeout_seconds=30,
                )
                # File was written with exactly the response bytes.
                with open(output_path, "rb") as f:
                    self.assertEqual(f.read(), payload)
                # And we called requests.get with the right safety knobs.
                kwargs = mock_get.call_args.kwargs
                self.assertFalse(kwargs.get("allow_redirects", True))
                self.assertEqual(kwargs.get("timeout"), 30)


class TestAlignBuildingFootprintsStep(unittest.TestCase):
    """align_building_footprints must never fail the imagery layer."""

    def _workflow(self, tmp):
        return ImageryWorkflow(
            project_id="proj-1", image_layer_id="layer-9", dst_directory=tmp
        )

    def test_skipped_without_footprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._workflow(tmp).align_building_footprints()
            self.assertFalse(result["applied"])

    def test_soft_fails_on_unreadable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = self._workflow(tmp)
            wf.building_footprints_path = os.path.join(tmp, "footprints.gpkg")
            wf.mosaic_post_event_tif_filepath = os.path.join(tmp, "mosaic.tif")
            for path in (wf.building_footprints_path, wf.mosaic_post_event_tif_filepath):
                with open(path, "wb") as f:
                    f.write(b"not a real file")
            result = wf.align_building_footprints()  # must not raise
            self.assertFalse(result["applied"])
            self.assertIn("alignment failed", result["reason"])
            # The footprints file is left exactly as it was.
            with open(wf.building_footprints_path, "rb") as f:
                self.assertEqual(f.read(), b"not a real file")

    def test_result_is_json_serialisable(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            json.dumps(self._workflow(tmp).align_building_footprints())


class TestImageryUpdateStep(unittest.TestCase):
    """Imagery-update mode of imageryprep: download, match, align, plan."""

    def test_match_files_to_urls_by_name(self):
        from hastegeo.workflows.prepare_imagery import _match_files_to_urls

        urls = ["https://h/c/a.tif?sig=1", "https://h/c/b%20x.tif", "https://h/c/b.tif"]
        files = ["/w/b.tif", "/w/a.tif", "/w/b x.tif", "/w/zzz.tif"]
        self.assertEqual(
            _match_files_to_urls(files, urls),
            # The original URL comes back, signature included.
            ["https://h/c/b.tif", "https://h/c/a.tif?sig=1", "https://h/c/b%20x.tif", None],
        )

    def test_update_retires_covered_tile_and_feeds_the_mosaic(self):
        import shutil

        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        from hastegeo.core.utils.imagery import ImageryUtils

        rng = np.random.default_rng(3)
        scene = np.full((3, 600, 1200), 60, np.uint8)
        for _ in range(150):
            r, c = rng.integers(0, 560, 1)[0], rng.integers(0, 1160, 1)[0]
            scene[:, r:r + rng.integers(10, 40), c:c + rng.integers(10, 40)] = rng.integers(90, 250)

        def tif(path, col0, width, shift_x=0.0):
            with rasterio.open(path, "w", driver="GTiff", width=width, height=600, count=3,
                               dtype="uint8", crs="EPSG:32617",
                               transform=from_origin(500000 + col0 * 0.5 + shift_x, 3000000, 0.5, 0.5)) as d:
                d.write(scene[:, :, col0:col0 + width])

        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            tif(os.path.join(src, "left.tif"), 0, 600)
            tif(os.path.join(src, "right.tif"), 600, 600)
            tif(os.path.join(src, "fresh.tif"), 0, 700, shift_x=1.5)  # new, misregistered

            def fake_download(urls, dst_directory):
                os.makedirs(dst_directory, exist_ok=True)
                paths = []
                for url in reversed(urls):  # the real downloader reorders too
                    name = url.rsplit("/", 1)[1]
                    shutil.copy(os.path.join(src, name), os.path.join(dst_directory, name))
                    paths.append(os.path.join(dst_directory, name))
                return paths

            wf = ImageryWorkflow(
                project_id="proj-1", image_layer_id="layer-9", dst_directory=out,
                post_event_urls=["https://h/left.tif", "https://h/right.tif"],
            )
            # is_gtiff and the JPEG preview use ``with gdal.Open(...)``, which
            # needs GDAL >= 3.8 (the imageryprep image ships 3.9); stand in for
            # those two so the test also runs on older local GDAL builds.
            def fake_preview(tif_path, out_path, *a, **k):
                open(out_path, "wb").close()
                return out_path

            with patch("hastegeo.workflows.prepare_imagery._download_imagery", side_effect=fake_download), \
                    patch("hastegeo.workflows.prepare_imagery._are_all_valid_tif_files", return_value=True), \
                    patch.object(ImageryUtils, "convert_tif_to_jpeg", side_effect=fake_preview):
                result = wf.download_post_event_for_update(["https://h/fresh.tif"])

            self.assertEqual(result.retired_urls, ["https://h/left.tif"])
            self.assertEqual(result.active_urls, ["https://h/right.tif", "https://h/fresh.tif"])
            self.assertEqual(wf.post_event_raw_paths, result.mosaic_paths)
            self.assertEqual(len(wf.post_event_raw_paths), 2)
            self.assertEqual(len(wf.post_event_preview_paths), 2)  # right + fresh
            for p in wf.post_event_preview_paths:
                self.assertTrue(os.path.exists(p))
                self.assertEqual(os.path.dirname(p), out)  # previews are uploaded
            self.assertTrue(result.report["incoming"][0]["registration"]["applied"])
            # The aligned files mosaic without error, on one grid.
            mosaic = ImageryUtils.mosaic_imagery(
                wf.post_event_raw_paths, os.path.join(out, "m.tif"), "-of COG"
            )
            with rasterio.open(mosaic) as m:
                self.assertEqual(m.count, 3)
                self.assertEqual(m.width, 1200)


if __name__ == "__main__":
    unittest.main()
