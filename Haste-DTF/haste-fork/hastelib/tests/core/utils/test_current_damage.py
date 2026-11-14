# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import datetime as dt
import json
import os
import tempfile
import unittest

import pyarrow.parquet as pq
import shapely
from shapely.geometry import Polygon, mapping

from hastegeo.core.utils import current_damage

# Real (non-rectangular) footprint taken from Texas.geojson.
L_SHAPE = Polygon(
    [(-96.560633, 32.834667), (-96.560567, 32.834684), (-96.560602, 32.834783),
     (-96.560751, 32.834746), (-96.560698, 32.834597), (-96.560616, 32.834618),
     (-96.560633, 32.834667)]
)
NOW = dt.datetime(2026, 11, 2, 10, 0, tzinfo=dt.timezone.utc)


class FakeCursor:
    def __init__(self, rows, log):
        self.rows, self.log, self.rowcount = rows, log, 7

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, rows=()):
        self.rows, self.log, self.commits, self.rollbacks = list(rows), [], 0, 0

    def cursor(self):
        return FakeCursor(self.rows, self.log)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


class TestColors(unittest.TestCase):
    def test_five_classes_have_distinct_colors(self):
        colors = [current_damage.damage_color(i) for i in range(5)]
        self.assertEqual(len(set(colors)), 5)
        self.assertEqual(colors[2], "#fb7050")

    def test_unknown_class_gets_fallback(self):
        self.assertEqual(current_damage.damage_color(None), "#b0bec5")
        self.assertEqual(current_damage.damage_color(9), "#b0bec5")


class TestPublish(unittest.TestCase):
    def test_only_completed_results_are_published_and_geometry_untouched(self):
        current_damage._SCHEMA_READY = False
        conn = FakeConn()
        n = current_damage.publish_current_results("ast-1", conn=conn)
        self.assertIn("ADD COLUMN IF NOT EXISTS last_assessment_id", conn.log[0][0])
        sql, params = conn.log[-1]
        self.assertEqual(n, 7)
        self.assertEqual(params, ("ast-1", "completed"))
        set_clause = sql.split("SET")[1].split("FROM")[0]
        self.assertNotIn("geometry", set_clause)
        self.assertIn("d.result_status = %s", sql)
        self.assertIn("b.updated_at <= d.updated_at", sql)
        self.assertEqual(conn.commits, 2)  # schema check + publish

    def test_rolls_back_on_error(self):
        class Boom(FakeConn):
            def cursor(self):
                raise RuntimeError("db down")

        conn = Boom()
        with self.assertRaises(RuntimeError):
            current_damage.publish_current_results("ast-1", conn=conn)
        self.assertEqual(conn.rollbacks, 1)


def _geojson_row():
    return ("b1", json.dumps(mapping(L_SHAPE)), 2, "Total Roof Damage",
            0.91, "completed", "ast-1", NOW)


class TestGeoJson(unittest.TestCase):
    def test_feature_keeps_exact_polygon_and_attributes(self):
        conn = FakeConn([_geojson_row()])
        out = current_damage.get_current_geojson("layer", conn=conn)
        f = out["features"][0]
        self.assertEqual(f["geometry"]["type"], "Polygon")
        self.assertEqual(shapely.from_geojson(json.dumps(f["geometry"])).wkt, L_SHAPE.wkt)
        self.assertEqual(len(f["geometry"]["coordinates"][0]), 7)  # not a 5-pt box
        p = f["properties"]
        self.assertEqual(p["damage_color"], "#fb7050")
        self.assertEqual(p["damage_class_name"], "Total Roof Damage")
        self.assertEqual(p["assessment_id"], "ast-1")
        self.assertIsNone(out["nextCursor"])

    def test_bbox_and_cursor_are_parameterised(self):
        conn = FakeConn([])
        current_damage.get_current_geojson(
            "layer", bbox=(-97, 30, -96, 31), limit=10, after_building_id="b0", conn=conn
        )
        sql, params = conn.log[0]
        self.assertIn("ST_MakeEnvelope", sql)
        self.assertEqual(params, ["layer", -97, 30, -96, 31, "b0", 10])
        self.assertIn("DISTINCT ON (building_id)", sql)

    def test_statewide_requires_bbox(self):
        with self.assertRaises(ValueError):
            current_damage.get_current_geojson(None, conn=FakeConn())
        conn = FakeConn([])
        current_damage.get_current_geojson(None, bbox=(-97, 30, -96, 31), conn=conn)
        self.assertNotIn("image_layer_id", conn.log[0][0])

    def test_full_page_returns_cursor(self):
        conn = FakeConn([_geojson_row()])
        out = current_damage.get_current_geojson("layer", limit=1, conn=conn)
        self.assertEqual(out["nextCursor"], "b1")


class TestGeoJsonText(unittest.TestCase):
    def test_text_payload_is_valid_and_matches_dict_payload(self):
        row = _geojson_row()
        as_dict = current_damage.get_current_geojson("layer", conn=FakeConn([row]))
        text = current_damage.get_current_geojson_text("layer", conn=FakeConn([row]))
        parsed = json.loads(text)
        self.assertEqual(parsed["features"], as_dict["features"])
        self.assertIsNone(parsed["nextCursor"])

    def test_empty_and_cursor(self):
        self.assertEqual(
            json.loads(current_damage.get_current_geojson_text("layer", conn=FakeConn([]))),
            {"type": "FeatureCollection", "features": [], "nextCursor": None},
        )
        full = current_damage.get_current_geojson_text("layer", limit=1, conn=FakeConn([_geojson_row()]))
        self.assertEqual(json.loads(full)["nextCursor"], "b1")


class TestParquet(unittest.TestCase):
    def test_roundtrip_preserves_polygon_exactly(self):
        row = ("b1", shapely.to_wkb(L_SHAPE), 3, "Partial Structural Collapse",
               0.8, "completed", "ast-2", NOW)
        conn = FakeConn([row])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.parquet")
            n = current_damage.export_geoparquet("layer", path, conn=conn)
            table = pq.read_table(path)
            meta = json.loads(table.schema.metadata[b"geo"])
        self.assertEqual(n, 1)
        got = shapely.from_wkb(table["geometry"][0].as_py())
        self.assertTrue(got.equals_exact(L_SHAPE, 0))
        self.assertEqual(table["damage_color"][0].as_py(), "#d32020")
        self.assertEqual(meta["columns"]["geometry"]["encoding"], "WKB")
        self.assertEqual(meta["columns"]["geometry"]["crs"]["id"]["code"], 4326)

    def test_auto_export_disabled_without_env(self):
        os.environ.pop("HASTE_CURRENT_DAMAGE_EXPORT_DIR", None)
        self.assertIsNone(current_damage.export_after_publish("layer", conn=FakeConn()))

    def test_auto_export_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["HASTE_CURRENT_DAMAGE_EXPORT_DIR"] = d

            class Boom(FakeConn):
                def cursor(self):
                    raise RuntimeError("x")

            try:
                self.assertIsNone(current_damage.export_after_publish("layer", conn=Boom()))
            finally:
                os.environ.pop("HASTE_CURRENT_DAMAGE_EXPORT_DIR")


if __name__ == "__main__":
    unittest.main()
