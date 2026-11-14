"""Crop-mode scoring, condition-based batch sizing and unit prefetching in the
large-AOI region runner. Uses stub models, so it needs no checkpoint or GPU."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from hastegeo.workflows import damage_triage_infer as infer
from hastegeo.workflows import damage_triage_region_infer as region

N_CLASSES = 5


class StubCropModel:
    """Per-sample deterministic probabilities, so results cannot depend on how
    crops are batched. Records the batch sizes it was called with."""

    crop_mode = True
    use_gated_head = True

    def __init__(self, oom_above=None, oom_single_calls=0):
        self.calls = []
        self.oom_above = oom_above
        self.oom_single_calls = oom_single_calls
        self.device_moves = []

    def parameters(self):
        return iter([torch.zeros(1)])

    def to(self, device):
        self.device_moves.append(str(device))
        return self

    def __call__(self, x):
        n = x.shape[0]
        self.calls.append(n)
        if self.oom_above is not None and n > self.oom_above:
            raise torch.cuda.OutOfMemoryError("stub")
        if n == 1 and self.oom_single_calls > 0:
            self.oom_single_calls -= 1
            raise torch.cuda.OutOfMemoryError("stub")
        mean = x.mean(dim=(1, 2, 3))
        logits = torch.stack([mean * (i + 1) for i in range(N_CLASSES)], dim=1)
        return torch.softmax(logits, dim=1)


def _tile(seed=0):
    return np.random.default_rng(seed).integers(
        0, 255, (3, 1024, 1024), dtype=np.uint8
    )


def _boxes(n):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        x, y = (int(v) for v in rng.integers(0, 900, 2))
        out.append((x, y, x + 40 + i % 30, y + 40 + i % 30))
    return out


class TestCropBox(unittest.TestCase):
    def test_margin_is_added_around_the_footprint(self):
        mask = np.zeros((1024, 1024), bool)
        mask[200:240, 100:140] = True
        self.assertEqual(infer.crop_box_from_mask(mask, 1024, 1024), (68, 168, 172, 272))

    def test_box_is_clipped_to_the_real_window(self):
        mask = np.zeros((1024, 1024), bool)
        mask[10:30, 600:640] = True
        # a 700 px wide edge window: the box must not reach into padding
        self.assertEqual(infer.crop_box_from_mask(mask, 700, 500), (568, 0, 672, 62))

    def test_empty_mask_has_no_box(self):
        self.assertIsNone(infer.crop_box_from_mask(np.zeros((8, 8), bool), 8, 8))


class TestBatchSizeFromHardware(unittest.TestCase):
    def test_cpu_uses_a_small_fixed_batch(self):
        cpu = torch.device("cpu")
        self.assertEqual(infer.resolve_crops_per_forward(cpu), infer.CROP_CPU_PER_FORWARD)
        self.assertEqual(infer.resolve_crops_per_forward(cpu, upper=2), 2)

    def test_cuda_batch_follows_free_memory_in_powers_of_two(self):
        cuda = torch.device("cuda")
        gib = 1024**3
        self.assertEqual(infer.resolve_crops_per_forward(cuda, free_bytes=100 * gib), 64)
        self.assertEqual(infer.resolve_crops_per_forward(cuda, free_bytes=1 * gib), 32)
        self.assertEqual(infer.resolve_crops_per_forward(cuda, free_bytes=300 * 2**20), 8)
        self.assertEqual(infer.resolve_crops_per_forward(cuda, free_bytes=10 * 2**20), 1)

    def test_configured_upper_bound_always_wins(self):
        cuda = torch.device("cuda")
        self.assertEqual(
            infer.resolve_crops_per_forward(cuda, upper=16, free_bytes=100 * 1024**3), 16
        )
        self.assertEqual(
            infer.resolve_crops_per_forward(cuda, upper=20, free_bytes=100 * 1024**3), 16
        )


class TestScoreCrops(unittest.TestCase):
    cpu = torch.device("cpu")

    def test_no_boxes_gives_empty_result(self):
        out = infer.score_crops(StubCropModel(), _tile(), [], self.cpu)
        self.assertEqual(out.shape, (0, N_CLASSES))

    def test_chunks_follow_crops_per_forward_and_keep_order(self):
        model, tile, boxes = StubCropModel(), _tile(), _boxes(10)
        out = infer.score_crops(model, tile, boxes, self.cpu, crops_per_forward=4)
        self.assertEqual(model.calls, [4, 4, 2])
        singles = np.vstack(
            [infer.score_crops(StubCropModel(), tile, [b], self.cpu, 1) for b in boxes]
        )
        np.testing.assert_allclose(out, singles, atol=1e-6)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_batch_size_does_not_change_results(self):
        tile, boxes = _tile(), _boxes(9)
        ref = infer.score_crops(StubCropModel(), tile, boxes, self.cpu, crops_per_forward=9)
        for size in (1, 2, 5, 64):
            got = infer.score_crops(StubCropModel(), tile, boxes, self.cpu, crops_per_forward=size)
            np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_out_of_memory_halves_the_batch_and_loses_nothing(self):
        model, tile, boxes = StubCropModel(oom_above=3), _tile(), _boxes(10)
        out = infer.score_crops(model, tile, boxes, self.cpu, crops_per_forward=8)
        ref = infer.score_crops(StubCropModel(), tile, boxes, self.cpu, crops_per_forward=10)
        np.testing.assert_allclose(out, ref, atol=1e-6)
        self.assertEqual(model.calls[:3], [8, 4, 2])
        self.assertLessEqual(max(model.calls[3:]), 2)

    def test_single_crop_out_of_memory_falls_back_to_cpu(self):
        model, tile, boxes = StubCropModel(oom_single_calls=1), _tile(), _boxes(2)
        out = infer.score_crops(model, tile, boxes, self.cpu, crops_per_forward=1)
        self.assertEqual(out.shape, (2, N_CLASSES))
        self.assertIn("cpu", model.device_moves)


def _square(col0, row0, size, transform):
    """GeoJSON square covering exactly pixels [col0, col0+size) x [row0, row0+size)
    of a raster in EPSG:4326 (inset half a pixel so all_touched adds nothing)."""
    x0, y0 = transform * (col0 + 0.5, row0 + 0.5)
    x1, y1 = transform * (col0 + size - 0.5, row0 + size - 0.5)
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


class TestRegionRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.transform = from_origin(0.0, 0.1024, 0.0001, 0.0001)
        self.image = os.path.join(self.tmp.name, "image.tif")
        data = np.random.default_rng(3).integers(50, 250, (3, 1024, 2048), dtype=np.uint8)
        with rasterio.open(
            self.image, "w", driver="GTiff", width=2048, height=1024, count=3,
            dtype="uint8", crs="EPSG:4326", transform=self.transform,
        ) as dst:
            dst.write(data)
        self.ckpt = os.path.join(self.tmp.name, "best.pth")
        self.cfg = os.path.join(self.tmp.name, "config.json")
        for path in (self.ckpt, self.cfg):
            with open(path, "wb") as f:
                f.write(b"x")

    def config(self, masks_per_forward=None):
        def b(bid, col, row, size):
            return {
                "building_id": bid,
                "coverage_fraction": 1.0,
                "geometry": _square(col, row, size, self.transform),
            }

        def unit(uid, col_off, buildings):
            return {
                "unit_id": uid, "col_off": col_off, "row_off": 0,
                "width": 1024, "height": 1024, "kind": "grid", "buildings": buildings,
            }

        return {
            "batch_id": "b1", "assessment_id": "a1", "imagery_id": "i1",
            "output_dir": "outputs", "image_path": self.image,
            "checkpoint_path": self.ckpt, "checkpoint_config_path": self.cfg,
            "normalization": "imagenet", "masks_per_forward": masks_per_forward,
            "min_valid_pixel_fraction": 0.5,
            "units": [
                unit("u1", 0, [b("A", 100, 200, 40), b("B", 500, 600, 50), b("tiny", 700, 700, 4)]),
                unit("u2", 1024, [b("C", 1124, 300, 60)]),
                unit("u3", 0, []),
            ],
        }

    def run_region(self, model, score_crops=None, score_window=None, prefetch="2", **cfg_kw):
        patches = [
            mock.patch.object(region, "WORKDIR", self.tmp.name),
            mock.patch.object(region, "build_dtf_model", return_value=(model, True)),
            mock.patch.object(region, "resolve_device", return_value=torch.device("cpu")),
            mock.patch.dict(os.environ, {"HASTE_REGION_PREFETCH_UNITS": prefetch}),
        ]
        if score_crops is not None:
            patches.append(mock.patch.object(region, "score_crops", side_effect=score_crops))
        if score_window is not None:
            patches.append(mock.patch.object(region, "score_window", side_effect=score_window))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        manifest = region.run_batch_inference(self.config(**cfg_kw))
        with open(os.path.join(self.tmp.name, "outputs", "evidence.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        return manifest, rows

    def test_crop_boxes_come_from_each_footprint_in_order(self):
        seen = []

        def fake(model, tile, boxes, device, crops_per_forward):
            seen.append((list(boxes), crops_per_forward))
            return np.tile([0.1, 0.2, 0.3, 0.2, 0.2], (len(boxes), 1))

        manifest, rows = self.run_region(StubCropModel(), score_crops=fake)
        # A: pixels cols 100-139, rows 200-239 -> grown by 32 px
        self.assertEqual(seen[0][0], [(68, 168, 172, 272), (468, 568, 582, 682)])
        # C is in the second window, whose own pixel origin is col 1024
        self.assertEqual(seen[1][0], [(68, 268, 192, 392)])
        self.assertEqual([r["building_id"] for r in rows], ["A", "B", "C"])
        self.assertNotIn("tiny", [r["building_id"] for r in rows])  # under 30 px
        self.assertEqual(manifest["evidence_count"], 3)
        self.assertEqual(manifest["completed_units"], 3)
        self.assertEqual(manifest["failed_units"], 0)

    def test_batch_size_comes_from_the_hardware_rule_and_config_cap(self):
        sizes = []

        def fake(model, tile, boxes, device, crops_per_forward):
            sizes.append(crops_per_forward)
            return np.full((len(boxes), N_CLASSES), 0.2)

        self.run_region(StubCropModel(), score_crops=fake)
        self.assertEqual(set(sizes), {infer.CROP_CPU_PER_FORWARD})  # CPU rule
        sizes.clear()
        self.run_region(StubCropModel(), score_crops=fake, masks_per_forward=2)
        self.assertEqual(set(sizes), {2})

    def test_prefetch_depth_does_not_change_results(self):
        outputs = []
        for depth in ("1", "2", "5"):
            _, rows = self.run_region(StubCropModel(), prefetch=depth)
            outputs.append(rows)
            patch_stack = []  # fresh patches each run (cleanups stop at teardown)
            del patch_stack
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(len(outputs[0]), 3)

    def test_a_failing_unit_is_reported_and_the_rest_still_run(self):
        calls = {"n": 0}

        def flaky(model, tile, boxes, device, crops_per_forward):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return np.full((len(boxes), N_CLASSES), 0.2)

        manifest, rows = self.run_region(StubCropModel(), score_crops=flaky)
        self.assertEqual(manifest["failed_units"], 1)
        self.assertEqual(manifest["completed_units"], 2)
        self.assertEqual([r["building_id"] for r in rows], ["C"])

    def test_mask_mode_models_keep_the_original_path(self):
        masks_seen = []

        def fake(**kw):
            masks_seen.append(kw["masks"].shape)
            return np.full((len(kw["masks"]), N_CLASSES), 0.2)

        manifest, rows = self.run_region(object(), score_window=fake)
        self.assertEqual(masks_seen, [(3, 1024, 1024), (1, 1024, 1024)])
        self.assertEqual(manifest["evidence_count"], 4)  # tiny (16 px) is kept here


if __name__ == "__main__":
    unittest.main()
