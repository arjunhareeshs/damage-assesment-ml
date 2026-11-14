import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from hastegeo.workflows.damage_triage_aoi_infer import run_damage_triage_aoi
from hastegeo.workflows.damage_triage_infer import build_dtf_model, resolve_device


def _square(x, y, size=0.001):
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
        ],
    }


class TestOffRasterBuildings(unittest.TestCase):
    def test_only_buildings_on_the_raster_reach_the_model(self):
        model = mock.MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.crop_mode = False
        model.return_value = (None, "damage_out")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "image.tif")
            with rasterio.open(
                path, "w", driver="GTiff", width=256, height=256, count=3,
                dtype="uint8", crs="EPSG:4326",
                transform=from_origin(0.0, 0.0256, 0.0001, 0.0001),
            ) as dst:
                dst.write(np.full((3, 256, 256), 100, dtype=np.uint8))
            buildings = [
                {"building_id": "on", "geometry": _square(0.01, 0.01)},
                {"building_id": "east", "geometry": _square(0.5, 0.01)},
                {"building_id": "west", "geometry": _square(-0.5, 0.01)},
            ]
            with mock.patch(
                "hastegeo.workflows.damage_triage_aoi_infer.decode_gated_predictions",
                return_value=np.array([[0.9, 0.1, 0.0, 0.0, 0.0]]),
            ):
                results = run_damage_triage_aoi(
                    path, buildings, None, None, tmp,
                    preloaded_model=(model, True), crop_size=256,
                )
        self.assertEqual(model.call_count, 1)
        self.assertEqual([r["class"] for r in results], [0, -1, -1])


class TestCheckpointValidation(unittest.TestCase):
    def test_crop_checkpoint_refused_by_tile_workflows(self):
        # torch.load is stubbed so the test does not depend on real tensor
        # serialization, which other tests in the suite can leave mocked.
        with mock.patch.object(
            torch, "load", return_value={"gate.weight": object()}
        ):
            with self.assertRaisesRegex(RuntimeError, "crop-mode"):
                build_dtf_model("best.pth", "unused.json", torch.device("cpu"))


class TestResolveDevice(unittest.TestCase):
    def test_explicit_cuda_refuses_cpu_fallback(self):
        with mock.patch.dict(os.environ, {"DTF_DEVICE": "cuda"}), mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            with self.assertRaises(RuntimeError):
                resolve_device()

    def test_auto_falls_back_to_cpu(self):
        with mock.patch.dict(os.environ, {"DTF_DEVICE": "auto"}), mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            self.assertEqual(resolve_device().type, "cpu")


if __name__ == "__main__":
    unittest.main()
