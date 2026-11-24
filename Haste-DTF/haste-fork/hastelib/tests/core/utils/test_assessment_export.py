"""Tabular / spatial exports of a large-AOI assessment and the review step."""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pyarrow.parquet as pq
from shapely.geometry import box

from hastegeo.core.utils import assessment_export as ex
from hastegeo.core.utils import assessment_store as store


def _rows():
    rows = []
    for i, (cls, review) in enumerate(
        [(0, "accepted"), (3, "pending_review"), (None, "unclassified"), (4, "rejected")]
    ):
        r = {c: None for c in ex.TABLE_COLUMNS}
        r.update(
            assessment_id="ast-1", image_layer_id="L", building_id=f"b{i}",
            damage_class=cls, confidence=0.8, review_status=review,
            result_status="completed" if cls is not None else "insufficient_data",
            boundary_intersecting=False, centroid_lon=1.0, centroid_lat=2.0,
            footprint_area_m2=60.0, _wkb=box(i, 0, i + 1, 1).wkb,
        )
        rows.append(r)
    return rows


META = {
    "assessment_id": "ast-1", "project_id": "p", "image_layer_id": "L",
    "name": "n", "status": "completed", "aoi_area_km2": 1.5,
    "aoi_geometry": box(0, 0, 4, 1).__geo_interface__, "model_digest": "d",
    "preprocessing_version": "1.0.0", "created_at": None, "completed_at": None,
}


class TestTable(unittest.TestCase):
    def test_columns_are_fixed_and_include_class_probabilities(self):
        self.assertEqual(len(ex.PROB_COLUMNS), 5)
        self.assertIn("p_undamaged", ex.PROB_COLUMNS)
        out = ex.table_rows(_rows())
        self.assertEqual(list(out[0].keys()), ex.TABLE_COLUMNS)
        self.assertNotIn("_wkb", out[0])

    def test_summary_counts_by_class_and_review(self):
        s = ex.summary_rows(_rows())
        total = s[-1]
        self.assertEqual(total["buildings"], 4)
        self.assertEqual(total["accepted"], 1)
        self.assertEqual(total["rejected"], 1)
        by = {r["damage_class_name"]: r for r in s}
        self.assertEqual(by["Partial Structural Collapse"]["pending_review"], 1)
        self.assertEqual(by["No result"]["buildings"], 1)

    def test_empty_summary_has_no_division_error(self):
        self.assertEqual(ex.summary_rows([])[-1]["buildings"], 0)

    def test_aoi_feature_carries_metadata(self):
        f = ex.aoi_feature(META)
        self.assertEqual(f["geometry"]["type"], "Polygon")
        self.assertEqual(f["properties"]["assessment_id"], "ast-1")


class TestFiles(unittest.TestCase):
    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.csv")
            ex.write_export("csv", META, _rows(), p)
            got = list(csv.DictReader(open(p)))
        self.assertEqual(len(got), 4)
        self.assertEqual(list(got[0].keys()), ex.TABLE_COLUMNS)

    def test_parquet_and_geoparquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.parquet"), os.path.join(tmp, "b.parquet")
            ex.write_export("parquet", META, _rows(), a)
            ex.write_export("geoparquet", META, _rows(), b)
            self.assertEqual(pq.read_table(a).num_rows, 4)
            self.assertNotIn("geometry", pq.read_table(a).column_names)
            self.assertIn("geometry", pq.read_table(b).column_names)
            self.assertIn(b"geo", pq.read_schema(b).metadata)

    def test_gpkg_has_buildings_aoi_and_summary_layers(self):
        import pyogrio

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.gpkg")
            ex.write_export("gpkg", META, _rows(), p)
            names = [l[0] for l in pyogrio.list_layers(p)]
        self.assertEqual(names, ["buildings", "aoi", "summary"])

    def test_unknown_file_format_rejected(self):
        with self.assertRaises(ValueError):
            ex.write_export("xlsx", META, _rows(), "x")

    def test_review_filter_validated(self):
        with self.assertRaises(ValueError):
            ex.fetch_results("ast-1", review="bogus", conn=MagicMock())


class TestReviewRules(unittest.TestCase):
    def test_only_completed_results_are_reviewable(self):
        self.assertEqual(store.initial_review_status("completed", True), "pending_review")
        self.assertEqual(store.initial_review_status("completed", False), "accepted")
        for s in ("uncertain", "insufficient_data", "failed"):
            self.assertEqual(store.initial_review_status(s, True), "unclassified")

    def test_invalid_status_and_bulk_without_assessment_rejected(self):
        for args in (
            dict(building_ids=["b"], review_status="unclassified"),
            dict(building_ids=["b"], review_status="nonsense"),
            dict(building_ids=None, review_status="accepted"),
        ):
            with self.assertRaises(ValueError):
                store.review_buildings("L", conn=MagicMock(), **args)

    def test_review_updates_both_tables_and_reports_count(self):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [("ast-1", "b1"), ("ast-1", "b2")]
        store._REVIEW_SCHEMA_READY = True
        n = store.review_buildings(
            "L", ["b1", "b2"], "accepted", assessment_id="ast-1",
            reviewed_by="me", conn=conn,
        )
        self.assertEqual(n, 2)
        sql = " ".join(c.args[0] for c in cur.execute.call_args_list)
        self.assertIn("UPDATE hg_damage_assessments", sql)
        self.assertIn("UPDATE hg_buildings", sql)
        conn.commit.assert_called()

    def test_no_match_skips_the_buildings_update(self):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = []
        store._REVIEW_SCHEMA_READY = True
        self.assertEqual(
            store.review_buildings("L", ["x"], "rejected", assessment_id="a", conn=conn), 0
        )
        sql = " ".join(c.args[0] for c in cur.execute.call_args_list)
        self.assertNotIn("UPDATE hg_buildings", sql)


if __name__ == "__main__":
    unittest.main()
