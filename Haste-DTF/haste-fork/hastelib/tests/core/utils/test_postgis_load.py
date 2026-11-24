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

    def _load(self, gdf=None, **kw):
        from unittest.mock import patch

        connection, cursor = self._mock_connection()
        staged = []
        with tempfile.TemporaryDirectory() as tmp, patch(
            "psycopg2.extras.execute_values",
            side_effect=lambda cur, sql, rows, **k: staged.extend(rows),
        ):
            gpkg_path = os.path.join(tmp, "footprints.gpkg")
            if gdf is None:
                _fake_footprints_gpkg(gpkg_path)
            else:
                gdf.to_file(gpkg_path, driver="GPKG")
            count = load_footprints_into_postgis(
                self.image_layer, gpkg_path, conn=connection, **kw
            )
        sql = [c.args[0] for c in cursor.execute.call_args_list]
        params = [c.args[1] if len(c.args) > 1 else None for c in cursor.execute.call_args_list]
        return count, staged, sql, params, connection

    def test_stages_every_building_and_commits_once(self):
        count, staged, _sql, _p, connection = self._load()
        self.assertEqual(count, 2)
        self.assertEqual({r[0] for r in staged}, {"B101", "B102"})
        connection.commit.assert_called_once()
        connection.close.assert_not_called()

    def test_never_wipes_results_only_removes_buildings_gone(self):
        _c, _s, sql, params, _conn = self._load()
        delete = next(s for s in sql if s.strip().startswith("DELETE"))
        self.assertIn("NOT EXISTS", delete)
        upsert = next(s for s in sql if "INSERT INTO hg_buildings" in s)
        self.assertIn("ON CONFLICT", upsert)
        # damage columns are never in the update list
        self.assertNotIn("damage_class =", upsert)

    def test_full_reprep_flags_results_stale_update_flags_changed_only(self):
        for preserve, expect in ((False, True), (True, False)):
            _c, _s, sql, params, _conn = self._load(preserve_results=preserve)
            i = next(i for i, s in enumerate(sql) if "INSERT INTO hg_buildings" in s)
            self.assertIn("imagery_updated_at", sql[i])
            self.assertIs(params[i][-1], expect)

    def test_source_and_version_recorded(self):
        _c, _s, sql, params, _conn = self._load(
            footprint_source="overture", footprint_version="2026-02-18.0"
        )
        i = next(i for i, s in enumerate(sql) if "INSERT INTO hg_buildings" in s)
        self.assertIn("overture", params[i])
        self.assertIn("2026-02-18.0", params[i])

    def test_missing_and_duplicate_ids_are_made_unique_and_stable(self):
        gdf = gpd.GeoDataFrame(
            {"id": [None, "A", "A"], "subtype": [None] * 3, "class": [None] * 3},
            geometry=[
                shapely.geometry.box(0, 0, 1, 1),
                shapely.geometry.box(2, 2, 3, 3),
                shapely.geometry.box(4, 4, 5, 5),
            ],
            crs="EPSG:4326",
        )
        _c, staged, *_ = self._load(gdf=gdf, footprint_source="user")
        ids = [r[0] for r in staged]
        self.assertEqual(len(set(ids)), 3)
        self.assertIn("A", ids)
        self.assertIn("A-2", ids)
        _c, staged2, *_ = self._load(gdf=gdf, footprint_source="user")
        self.assertEqual(ids, [r[0] for r in staged2])  # stable across loads

    def test_reads_in_chunks(self):
        count, staged, *_ = self._load(chunk_size=1)
        self.assertEqual(count, 2)
        self.assertEqual(len(staged), 2)

    def test_empty_gpkg_stages_nothing(self):
        empty = gpd.GeoDataFrame(
            {"id": [], "subtype": [], "class": []}, geometry=[], crs="EPSG:4326"
        )
        count, staged, *_ = self._load(gdf=empty)
        self.assertEqual(count, 0)
        self.assertEqual(staged, [])


if __name__ == "__main__":
    unittest.main()

