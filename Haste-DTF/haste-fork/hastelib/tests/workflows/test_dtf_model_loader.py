"""The single DamageTriageFormer loading path and preprocessing contract."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from hastegeo.core.utils.dtf_preprocess import (
    DEFAULT_SPEC,
    PreprocessingSpec,
    load_preprocessing_spec,
    resolve_normalization,
)
from hastegeo.workflows import damage_triage_infer as infer
from hastegeo.workflows.damage_triage_infer import (
    build_dtf_model,
    read_checkpoint_config,
    resolve_device,
    selftest_model,
)

MODEL_CARD = {
    "architecture": {
        "mask_pooling": False,
        "class_names": list(infer.DTF_CATEGORIES),
    },
    "preprocessing": {
        "resize": [224, 224],
        "crop_margin_pixels": 32,
        "minimum_footprint_area_pixels": 30,
        "color_space": "RGB",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    },
}


def _write(tmp, name, obj):
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


class TestPreprocessingSpec(unittest.TestCase):
    def test_default_is_imagenet(self):
        self.assertEqual(DEFAULT_SPEC.mean, (0.485, 0.456, 0.406))
        self.assertEqual(DEFAULT_SPEC.input_size, 224)

    def test_spec_read_from_model_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = load_preprocessing_spec(_write(tmp, "c.json", MODEL_CARD))
        self.assertEqual(spec.source, "config")
        self.assertEqual(spec.crop_margin_px, 32)
        self.assertEqual(spec.min_footprint_px, 30)

    def test_missing_file_or_block_falls_back_to_default(self):
        self.assertIs(load_preprocessing_spec("/nope.json"), DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIs(
                load_preprocessing_spec(_write(tmp, "c.json", {"use_gated_head": 1})),
                DEFAULT_SPEC,
            )

    def test_malformed_block_raises(self):
        bad = {"preprocessing": {"image_std": [0.2, 0.0, 0.2]}}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            load_preprocessing_spec(_write(tmp, "c.json", bad))
        bad = {"preprocessing": {"resize": [224, 256]}}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            load_preprocessing_spec(_write(tmp, "c.json", bad))

    def test_normalize_matches_imagenet_formula(self):
        x = np.full((3, 2, 2), 128, dtype=np.uint8)
        out = DEFAULT_SPEC.normalize(x)
        expect = (128 / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        np.testing.assert_allclose(out[:, 0, 0], expect, rtol=1e-5)

    def test_custom_spec_changes_output_and_fingerprint(self):
        other = PreprocessingSpec(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        x = np.full((3, 2, 2), 255, dtype=np.uint8)
        np.testing.assert_allclose(other.normalize(x), 1.0)
        self.assertNotEqual(other.fingerprint, DEFAULT_SPEC.fingerprint)

    def test_only_imagenet_normalization_is_accepted(self):
        self.assertEqual(resolve_normalization(None), "imagenet")
        self.assertEqual(resolve_normalization("ImageNet"), "imagenet")
        with self.assertRaises(ValueError):
            resolve_normalization("unit")

    def test_score_window_rejects_other_policies(self):
        with self.assertRaises(ValueError):
            infer.score_window(
                mock.MagicMock(), True, np.zeros((3, 8, 8), np.uint8),
                np.ones((1, 8, 8), np.float32), torch.device("cpu"),
                normalization="unit",
            )


class TestCheckpointConfig(unittest.TestCase):
    def test_model_card_crop_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = read_checkpoint_config(_write(tmp, "c.json", MODEL_CARD))
        self.assertEqual(info["schema"], "model_card")
        self.assertTrue(info["crop_mode"])

    def test_training_dump_is_mask_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = read_checkpoint_config(
                _write(tmp, "c.json", {"use_gated_head": True, "sfp_stride": 8})
            )
        self.assertEqual(info["schema"], "training_dump")
        self.assertFalse(info["crop_mode"])

    def test_unrecognised_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            read_checkpoint_config(_write(tmp, "c.json", {"hello": 1}))


class TestCheckpointValidation(unittest.TestCase):
    def test_crop_checkpoint_refused_by_tile_workflows(self):
        with mock.patch.object(
            torch, "load", return_value={"gate.weight": object()}
        ):
            with self.assertRaisesRegex(RuntimeError, "crop-mode"):
                build_dtf_model("best.pth", "unused.json", torch.device("cpu"))

    def test_weights_are_loaded_tensor_only(self):
        with mock.patch.object(
            torch, "load", return_value={"gate.weight": object()}
        ) as loader:
            with self.assertRaises(RuntimeError):
                build_dtf_model("best.pth", "unused.json", torch.device("cpu"))
        self.assertIs(loader.call_args.kwargs.get("weights_only"), True)

    def test_unsafe_pickle_is_refused_not_retried(self):
        with mock.patch.object(torch, "load", side_effect=Exception("pickle")):
            with self.assertRaisesRegex(RuntimeError, "safely"):
                build_dtf_model("best.pth", "unused.json", torch.device("cpu"))

    def test_config_that_contradicts_the_weights_is_an_error(self):
        # Model card says crop-mode, weights are mask-pooling (no gate.weight).
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write(tmp, "c.json", MODEL_CARD)
            with mock.patch.object(
                torch, "load", return_value={"backbone.x": object()}
            ):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    build_dtf_model(
                        "best.pth", cfg, torch.device("cpu"), allow_crop_mode=True
                    )

    def test_digest_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write(tmp, "c.json", MODEL_CARD)
            ckpt = os.path.join(tmp, "best.pth")
            open(ckpt, "wb").write(b"x")
            with mock.patch.object(
                torch, "load", return_value={"gate.weight": object()}
            ):
                with self.assertRaisesRegex(RuntimeError, "digest"):
                    build_dtf_model(
                        ckpt, cfg, torch.device("cpu"),
                        allow_crop_mode=True, expected_digest="0" * 64,
                    )

    def test_card_sha_mismatch_is_an_error(self):
        card = dict(MODEL_CARD, weights_sha256="1" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write(tmp, "c.json", card)
            ckpt = os.path.join(tmp, "best.pth")
            open(ckpt, "wb").write(b"x")
            with mock.patch.object(
                torch, "load", return_value={"gate.weight": object()}
            ):
                with self.assertRaisesRegex(RuntimeError, "weights_sha256"):
                    build_dtf_model(
                        ckpt, cfg, torch.device("cpu"), allow_crop_mode=True
                    )


class _Probe(torch.nn.Module):
    crop_mode = True

    def __init__(self, out):
        super().__init__()
        self.out = out
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.as_tensor(self.out, dtype=torch.float32).repeat(len(x), 1)


class TestSelfTest(unittest.TestCase):
    def test_valid_output_passes(self):
        selftest_model(_Probe([[0.2] * 5]), torch.device("cpu"))

    def test_bad_outputs_fail(self):
        for out in ([[0.5] * 5], [[float("nan")] * 5], [[0.5, 0.5]]):
            with self.assertRaises(RuntimeError):
                selftest_model(_Probe(out), torch.device("cpu"))


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
