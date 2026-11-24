# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""ArtifactProcessor against a live blob store (Azurite).

The previous test here called ``ArtifactProcessor.zip()``, a method that no
longer exists (zipping now runs as a ``zip-artifacts`` task via
``submit_zip_job``). This covers the path the export endpoints use instead:
store a file, get a download URL, fetch it back.

Skipped unless BLOB_CONNECTION_STRING points at a disposable blob store, e.g.
Azurite: ``docker run -p 10000:10000 mcr.microsoft.com/azure-storage/azurite``.
"""
import os
import tempfile
import unittest
import urllib.request

CONN = os.getenv("BLOB_CONNECTION_STRING")


def _account_url(conn: str) -> str:
    parts = dict(p.split("=", 1) for p in conn.strip(";").split(";") if "=" in p)
    return parts["BlobEndpoint"]


@unittest.skipUnless(CONN, "BLOB_CONNECTION_STRING not set (needs a disposable blob store)")
class TestArtifactProcessorStorage(unittest.TestCase):
    def test_store_then_download(self):
        from unittest.mock import patch

        env = {
            "BLOB_CONNECTION_STRING": CONN,
            "AZURE_STORAGE_CONNECTION_STRING": CONN,
            "BLOB_ACCOUNT_URL": _account_url(CONN),
            "BLOB_CONTAINER": os.getenv("BLOB_CONTAINER", "data"),
            "ARTIFACT_STORAGE_TYPE": "blob",
            "QUEUE_ACCOUNT_URL": _account_url(CONN).replace(":10000", ":10001"),
        }
        with patch.dict(os.environ, env), \
                patch("hastegeo.core.processors.artifacts.AzureQueueHandler"), \
                patch("hastegeo.core.processors.artifacts.UnifiedRunner"):
            from hastegeo.core.processors.artifacts import ArtifactProcessor

            processor = ArtifactProcessor(partition_key="layer-artifact-test")
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
                f.write("building_id,damage_class\nb1,0\n")
                src = f.name
            try:
                processor.store_artifact(
                    artifact_name="export_test.csv", src_path=src, namespace="artifacts"
                )
                url = processor.get_download_url(
                    identifier="export_test.csv", extra_partition_keys="artifacts"
                )
            finally:
                os.unlink(src)
        self.assertTrue(url.startswith("http"), url)
        body = urllib.request.urlopen(url).read().decode()
        self.assertIn("b1,0", body)


class TestArtifactProcessorLazyClients(unittest.TestCase):
    """Storing an export must not build a task runner: a LocalRunner created
    in the API process fails (it needs /shared/azurite, which only the queue
    worker mounts), which turned every file export into a 500."""

    def test_runner_and_queue_built_only_when_used(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"RUNNER_TYPE": "local"}), \
                patch("hastegeo.core.processors.artifacts.UnifiedArtifactStorage"), \
                patch("hastegeo.core.processors.artifacts.UnifiedRunner") as runner, \
                patch("hastegeo.core.processors.artifacts.AzureQueueHandler") as queue:
            from hastegeo.core.processors.artifacts import ArtifactProcessor

            processor = ArtifactProcessor(partition_key="layer-1")
            processor.get_download_url(identifier="x.csv", extra_partition_keys="artifacts")
            runner.assert_not_called()
            queue.assert_not_called()
            self.assertIs(processor.runner, runner.return_value)
            self.assertIs(processor.queue_client, queue.return_value)
            runner.assert_called_once()
            queue.assert_called_once()


if __name__ == "__main__":
    unittest.main()
