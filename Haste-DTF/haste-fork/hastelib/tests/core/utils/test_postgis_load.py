# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for hastegeo.core.utils.postgis_load.

Exercises load_footprints_into_postgis() against a real GeoPackage (written
via geopandas, exactly the schema hastegeo.core.utils.footprints produces)
but a mocked psycopg2 connection/cursor, so no real Postgres is required.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import geopandas as gpd
import shapely.geometry

from hastegeo.core.utils.postgis_load import (
    compute_canonical_building_id,
    load_footprints_into_postgis,
    load_footprints_streaming,
)


def _fake_footprints_gpkg(path):
    gdf = gpd.GeoDataFrame(
        {
            "id": ["B101", "B102"],
            "subtype": ["residential", None],
            "class": [None, None],
            "geometry": [
                shapely.geometry.Polygon(
                    [(0, 0), (0, 1), (1, 1), (1, 0)]
                ),
                shapely.geometry.Polygon(
                    [(2, 2), (2, 3), (3, 3), (3, 2)]
                ),
            ],
        },
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG")


class TestCanonicalBuildingId(unittest.TestCase):
    def test_uses_unique_raw_id(self):
        poly = shapely.geometry.box(0, 0, 1, 1)
        seen = set()
        b_id = compute_canonical_building_id("RAW_123", poly, "test_source", seen)
        self.assertEqual(b_id, "RAW_123")
        self.assertIn("RAW_123", seen)

    def test_falls_back_to_sha1_when_no_raw_id(self):
        poly = shapely.geometry.box(0, 0, 1, 1)
        seen = set()
        b_id = compute_canonical_building_id(None, poly, "test_source", seen)
        self.assertEqual(len(b_id), 40)  # sha1 hex string
        self.assertIn(b_id, seen)

    def test_coordinate_rounding_to_7_decimals(self):
        # Two polygons differing only beyond 7 decimals should yield identical IDs
        poly1 = shapely.geometry.Polygon(
            [(0.12345671, 0.5), (0.12345671, 1.5), (1.1, 1.5), (1.1, 0.5)]
        )
        poly2 = shapely.geometry.Polygon(
            [(0.12345674, 0.5), (0.12345674, 1.5), (1.1, 1.5), (1.1, 0.5)]
        )
        id1 = compute_canonical_building_id(None, poly1, "ms")
        id2 = compute_canonical_building_id(None, poly2, "ms")
        self.assertEqual(id1, id2)


class TestLoadFootprintsStreaming(unittest.TestCase):
    def setUp(self):
        self.image_layer = SimpleNamespace(
            imageLayerId="layer-stream-1", projectId="project-stream-1"
        )

    def _mock_connection(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        return connection, cursor

    def test_streaming_loads_chunks_and_sets_source(self):
        connection, cursor = self._mock_connection()
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "footprints.gpkg")
            _fake_footprints_gpkg(gpkg_path)

            count = load_footprints_streaming(
                self.image_layer,
                gpkg_path,
                footprint_source="microsoft",
                footprint_version="2.0",
                chunk_size=1,
                conn=connection,
            )

        self.assertEqual(count, 2)
        delete_call = cursor.execute.call_args_list[0]
        self.assertIn("DELETE FROM hg_buildings", delete_call.args[0])
        self.assertEqual(delete_call.args[1], ("layer-stream-1",))
        # With chunk_size=1 and 2 features, executemany is called twice
        self.assertEqual(cursor.executemany.call_count, 2)
        connection.commit.assert_called_once()

    def test_streaming_jsonl_support(self):
        connection, cursor = self._mock_connection()
        import json
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "footprints.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": "J1",
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
                    "properties": {"subtype": "commercial"}
                }) + "\n")
                f.write(json.dumps({
                    "id": "J2",
                    "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]]]},
                    "properties": {"subtype": "residential"}
                }) + "\n")

            count = load_footprints_streaming(
                self.image_layer,
                jsonl_path,
                chunk_size=10,
                conn=connection,
            )

        self.assertEqual(count, 2)
        self.assertEqual(cursor.executemany.call_count, 1)


class TestLoadFootprintsIntoPostgis(unittest.TestCase):
    def setUp(self):
        self.image_layer = SimpleNamespace(
            imageLayerId="layer-1", projectId="project-1"
        )

    def _mock_connection(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        return connection, cursor

    def test_deletes_existing_rows_before_insert(self):
        connection, cursor = self._mock_connection()
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "footprints.gpkg")
            _fake_footprints_gpkg(gpkg_path)

            count = load_footprints_into_postgis(
                self.image_layer, gpkg_path, conn=connection
            )

        self.assertEqual(count, 2)
        delete_call = cursor.execute.call_args_list[0]
        self.assertIn("DELETE FROM hg_buildings", delete_call.args[0])
        self.assertEqual(delete_call.args[1], ("layer-1",))
        connection.commit.assert_called_once()

    def test_insert_rows_carry_building_id_and_layer_ids(self):
        connection, cursor = self._mock_connection()
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "footprints.gpkg")
            _fake_footprints_gpkg(gpkg_path)

            load_footprints_into_postgis(
                self.image_layer, gpkg_path, conn=connection
            )

        executemany_call = cursor.executemany.call_args
        self.assertIn("INSERT INTO hg_buildings", executemany_call.args[0])
        rows = executemany_call.args[1]
        self.assertEqual(len(rows), 2)
        building_ids = {row[0] for row in rows}
        self.assertEqual(building_ids, {"B101", "B102"})
        for row in rows:
            self.assertEqual(row[1], "layer-1")  # image_layer_id
            self.assertEqual(row[2], "project-1")  # project_id

    def test_does_not_close_injected_connection(self):
        connection, _cursor = self._mock_connection()
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "footprints.gpkg")
            _fake_footprints_gpkg(gpkg_path)
            load_footprints_into_postgis(
                self.image_layer, gpkg_path, conn=connection
            )
        connection.close.assert_not_called()

    def test_empty_gpkg_still_deletes_and_skips_insert(self):
        connection, cursor = self._mock_connection()
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "footprints.gpkg")
            gpd.GeoDataFrame(
                {"id": [], "subtype": [], "class": [], "geometry": []},
                crs="EPSG:4326",
            ).to_file(gpkg_path, driver="GPKG")

            count = load_footprints_into_postgis(
                self.image_layer, gpkg_path, conn=connection
            )

        self.assertEqual(count, 0)
        cursor.executemany.assert_not_called()
        connection.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()

