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
_stubs = {}
if importlib.util.find_spec("azure") is None:  # pragma: no cover
    for name in ("azure", "azure.storage", "azure.storage.blob", "azure.storage.queue"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        _stubs[name] = mod
    _stubs["azure.storage.blob"].BlobClient = MagicMock()
    _stubs["azure.storage.blob"].BlobServiceClient = MagicMock()
    _stubs["azure.storage.queue"].QueueServiceClient = MagicMock()
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
