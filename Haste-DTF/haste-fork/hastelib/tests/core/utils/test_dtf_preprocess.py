# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np

from hastegeo.core.utils.dtf_preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_VERSION,
    normalize_tile,
)


class TestDtfPreprocess(unittest.TestCase):
    def test_version_string(self):
        self.assertEqual(PREPROCESSING_VERSION, "1.0.0")

    def test_unit_normalization(self):
        tile = np.full((3, 64, 64), 255, dtype=np.uint8)
        norm = normalize_tile(tile, mode="unit")
        self.assertEqual(norm.dtype, np.float32)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

        tile_zero = np.zeros((3, 64, 64), dtype=np.uint8)
        norm_zero = normalize_tile(tile_zero, mode="unit")
        np.testing.assert_allclose(norm_zero, 0.0, atol=1e-5)

    def test_imagenet_normalization_chw(self):
        # Shape (3, 32, 32)
        tile = np.zeros((3, 32, 32), dtype=np.uint8)
        norm = normalize_tile(tile, mode="imagenet")
        self.assertEqual(norm.shape, (3, 32, 32))
        self.assertEqual(norm.dtype, np.float32)

        # Expected: (0 - mean) / std
        expected_ch0 = -IMAGENET_MEAN[0] / IMAGENET_STD[0]
        expected_ch1 = -IMAGENET_MEAN[1] / IMAGENET_STD[1]
        expected_ch2 = -IMAGENET_MEAN[2] / IMAGENET_STD[2]

        np.testing.assert_allclose(norm[0], expected_ch0, atol=1e-5)
        np.testing.assert_allclose(norm[1], expected_ch1, atol=1e-5)
        np.testing.assert_allclose(norm[2], expected_ch2, atol=1e-5)

    def test_imagenet_normalization_hwc(self):
        # Shape (32, 32, 3)
        tile = np.full((32, 32, 3), 255, dtype=np.uint8)
        norm = normalize_tile(tile, mode="imagenet")
        self.assertEqual(norm.shape, (32, 32, 3))

        # Expected: (1 - mean) / std
        expected_ch0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_ch1 = (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_ch2 = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

        np.testing.assert_allclose(norm[:, :, 0], expected_ch0, atol=1e-5)
        np.testing.assert_allclose(norm[:, :, 1], expected_ch1, atol=1e-5)
        np.testing.assert_allclose(norm[:, :, 2], expected_ch2, atol=1e-5)

    def test_invalid_mode_raises(self):
        tile = np.zeros((3, 10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            normalize_tile(tile, mode="invalid_mode")

    def test_invalid_shape_raises(self):
        tile = np.zeros((2, 10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            normalize_tile(tile, mode="imagenet")


if __name__ == "__main__":
    unittest.main()
