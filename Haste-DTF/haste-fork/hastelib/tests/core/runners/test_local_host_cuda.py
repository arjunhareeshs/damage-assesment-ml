# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Host-CUDA mount and GPU-fallback helpers of LocalRunner."""

import importlib.util
import logging
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

if importlib.util.find_spec("docker") is None:  # pragma: no cover
    raise unittest.SkipTest("docker SDK not installed")
# The helpers under test don't touch Azure. Where the SDK isn't installed
# (bare dev hosts), stub it only for the duration of this import so the
# stubs never leak into other test modules.
# Stub each missing module on its own: a partial install (e.g. only
# azure-storage-blob, pulled in by adlfs) must still import cleanly.
def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


_stubs = {}
for _name, _attrs in (
    ("azure", ()),
    ("azure.storage", ()),
    ("azure.storage.blob", ("BlobClient", "BlobServiceClient")),
    ("azure.storage.queue", ("QueueServiceClient",)),
):
    if _missing(_name):  # pragma: no cover
        _mod = types.ModuleType(_name)
        _mod.__path__ = []
        for _attr in _attrs:
            setattr(_mod, _attr, MagicMock())
        _stubs[_name] = _mod
with patch.dict(sys.modules, _stubs):
    local = __import__("hastegeo.core.runners.local", fromlist=["_"])
LOG = logging.getLogger("test")


class TestHostCudaVolume(unittest.TestCase):
    def test_unset_means_no_mount(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(local._host_cuda_volume(LOG), {})

    def test_absolute_path_mounted_read_only(self):
        with patch.dict(os.environ, {"HASTE_HOST_CUDA_DIR": "/home/u/nvidia"}):
            self.assertEqual(
                local._host_cuda_volume(LOG),
                {"/home/u/nvidia": {"bind": "/opt/host-cuda/nvidia", "mode": "ro"}},
            )

    def test_relative_path_rejected(self):
        with patch.dict(os.environ, {"HASTE_HOST_CUDA_DIR": "nvidia"}):
            self.assertEqual(local._host_cuda_volume(LOG), {})


class TestGpuImages(unittest.TestCase):
    def test_only_model_image_gets_gpu_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            for image in ("haste-training", "haste-training:latest",
                          "myacr.azurecr.io/haste-training:v2"):
                self.assertTrue(local._image_wants_gpu(image), image)
            for image in ("haste-imageryprep:latest", "haste-imageryprep", "", None):
                self.assertFalse(local._image_wants_gpu(image), image)

    def test_list_is_configurable(self):
        with patch.dict(os.environ, {"HASTE_GPU_IMAGES": "haste-training, haste-imageryprep"}):
            self.assertTrue(local._image_wants_gpu("haste-imageryprep:latest"))

    def test_gpu_switch(self):
        for value, expected in (("1", True), ("true", True), ("0", False), ("", False)):
            with patch.dict(os.environ, {"HASTE_ENABLE_GPU": value}):
                self.assertEqual(local._gpu_enabled(), expected, value)


class TestGpuAttachError(unittest.TestCase):
    def test_recognises_gpu_attach_failures(self):
        for msg in (
            'could not select device driver "" with capabilities: [[gpu]]',
            "nvidia-container-cli: initialization error: driver not loaded",
            "unknown or invalid runtime name: nvidia",
        ):
            self.assertTrue(local._is_gpu_attach_error(Exception(msg)), msg)

    def test_other_errors_are_not_swallowed(self):
        for msg in ("No such image: haste-training", "network haste_default not found"):
            self.assertFalse(local._is_gpu_attach_error(Exception(msg)), msg)


if __name__ == "__main__":
    unittest.main()
