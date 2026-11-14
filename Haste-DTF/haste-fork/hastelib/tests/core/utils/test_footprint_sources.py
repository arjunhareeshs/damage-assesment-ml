# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for multi-format user footprint ingestion (gdb / geojson / ...)."""

import os
import shutil
import tempfile
import unittest
import zipfile

import geopandas as gpd
import pyogrio
import shapely
from hastegeo.core.utils.footprint_sources import (
    FootprintSourceError,
    discover_sources,
    read_footprints,
)
from hastegeo.core.utils.footprints import clip_and_normalize_user_footprints

AOI = shapely.box(-97.80, 30.20, -97.70, 30.30)


def _frame(ids, x0=-97.79, crs=4326, extra=None, z=False):
    geoms = [shapely.box(x0 + i * 0.01, 30.21, x0 + i * 0.01 + 0.005, 30.215)
             for i in range(len(ids))]
    data = {"geometry": geoms}
    data.update(extra or {})
    gdf = gpd.GeoDataFrame(data, crs=4326)
    return gdf.to_crs(crs) if crs != 4326 else gdf


def _write_gdb(frame, path):
    try:
        pyogrio.write_dataframe(frame, path, driver="OpenFileGDB",
                                layer="Bldgs")
    except Exception as exc:  # GDAL < 3.6 cannot write OpenFileGDB
        raise unittest.SkipTest(f"OpenFileGDB write unsupported: {exc}")


def _zip_dir_contents(src_dir, zip_path, prefix=""):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, src_dir)
                zf.write(full, os.path.join(prefix, rel))


class TestFootprintSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def p(self, name):
        return os.path.join(self.tmp, name)

    def test_geojson_without_crs_assumed_wgs84(self):
        path = self.p("b.geojson")
        _frame(["a", "b"], extra={"id": ["a", "b"]}).to_file(path, driver="GeoJSON")
        out = read_footprints(path, AOI)
        self.assertEqual(list(out["id"]), ["a", "b"])
        self.assertEqual(out.crs.to_epsg(), 4326)

    def test_gpkg_reprojected_and_bbox_filtered(self):
        gdf = _frame(["a", "far"], crs=3857, extra={"id": ["a", "far"]})
        gdf.loc[1, "geometry"] = shapely.box(1e7, 5e6, 1.0001e7, 5.0001e6)
        path = self.p("b.gpkg")
        gdf.to_file(path, driver="GPKG")
        out = read_footprints(path, AOI)
        self.assertEqual(list(out["id"]), ["a"])
        self.assertEqual(out.crs.to_epsg(), 4326)

    def test_gdb_folder_and_zip_layouts(self):
        gdb = self.p("Region.gdb")
        _write_gdb(_frame(["1", "2"], crs=3857, extra={"Bldg_ID": ["1", "2"],
                                                       "Simp_type": ["R", "C"]}), gdb)
        out = read_footprints(gdb, AOI)
        self.assertEqual(list(out["id"]), ["1", "2"])
        self.assertEqual(list(out["subtype"]), ["R", "C"])

        root_zip = self.p("Region.gdb.zip")  # GDB files at the archive root
        _zip_dir_contents(gdb, root_zip)
        self.assertEqual(len(read_footprints(root_zip, AOI)), 2)

        nested_zip = self.p("nested.zip")  # a Region.gdb/ folder inside
        _zip_dir_contents(gdb, nested_zip, prefix="Region.gdb")
        self.assertEqual(len(read_footprints(nested_zip, AOI)), 2)

    def test_folder_merges_regions_and_qualifies_duplicate_ids(self):
        d = self.p("regions")
        os.makedirs(d)
        for name, x0 in (("FPR01", -97.79), ("FPR02", -97.75)):
            _write_gdb(_frame(["1"], x0=x0, extra={"Bldg_ID": ["7"]}),
                       os.path.join(d, f"{name}.gdb"))
        out = read_footprints(d, AOI)
        self.assertEqual(sorted(out["id"]), ["FPR01-7", "FPR02-7"])

    def test_non_overlapping_source_skipped_and_bad_source_tolerated(self):
        d = self.p("mix")
        os.makedirs(d)
        _frame(["a"], extra={"id": ["a"]}).to_file(os.path.join(d, "ok.gpkg"), driver="GPKG")
        gpd.GeoDataFrame({"id": ["z"], "geometry": [shapely.box(10, 10, 11, 11)]},
                         crs=4326).to_file(os.path.join(d, "far.gpkg"), driver="GPKG")
        with open(os.path.join(d, "broken.geojson"), "w") as f:
            f.write("{not json")
        out = read_footprints(d, AOI)
        self.assertEqual(list(out["id"]), ["ok-a"])

    def test_same_dataset_in_two_formats_loaded_once(self):
        d = self.p("both")
        os.makedirs(d)
        gdf = _frame(["1"], extra={"Bldg_ID": ["5"]})
        _write_gdb(gdf, os.path.join(d, "FPR01.gdb"))
        gdf.to_parquet(os.path.join(d, "FPR01.parquet"), write_covering_bbox=True)
        self.assertEqual([s.kind for s in discover_sources(d)], ["parquet"])
        self.assertEqual(len(read_footprints(d, AOI)), 1)

    def test_single_duplicate_ids_made_unique(self):
        path = self.p("d.geojson")
        _frame(["a", "a"], extra={"id": ["a", "a"]}).to_file(path, driver="GeoJSON")
        self.assertEqual(list(read_footprints(path, AOI)["id"]), ["a", "a-2"])

    def test_geoparquet(self):
        path = self.p("b.parquet")
        _frame(["a"], extra={"id": ["a"]}).to_parquet(path, write_covering_bbox=True)
        self.assertEqual(list(read_footprints(path, AOI)["id"]), ["a"])

    def test_raster_and_junk_rejected(self):
        tif = self.p("img.tif")
        with open(tif, "wb") as f:
            f.write(b"II*\x00" + b"\x00" * 100)
        with self.assertRaises(FootprintSourceError):
            discover_sources(tif)
        junk = self.p("x.txt")
        open(junk, "w").write("hello")
        with self.assertRaises(FootprintSourceError):
            discover_sources(junk)

    def test_zip_without_footprints_and_unsafe_zip_rejected(self):
        z = self.p("empty.zip")
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("readme.txt", "x")
        with self.assertRaises(FootprintSourceError):
            discover_sources(z)
        bad = self.p("slip.zip")
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../evil.gpkg", "x")
        with self.assertRaises(FootprintSourceError):
            discover_sources(bad)

    def test_gdb_read_disables_spatial_index_and_restores_it(self):
        from unittest.mock import patch

        from hastegeo.core.utils import footprint_sources as fs

        gdb = self.p("Idx.gdb")
        _write_gdb(_frame(["1"], extra={"Bldg_ID": ["1"]}), gdb)
        calls = []
        real = pyogrio.set_gdal_config_options
        with patch.object(
            fs.pyogrio, "set_gdal_config_options",
            side_effect=lambda o: (calls.append(dict(o)), real(o))[1],
        ):
            self.assertEqual(len(read_footprints(gdb, AOI)), 1)
        self.assertEqual(
            calls,
            [{"OPENFILEGDB_USE_SPATIAL_INDEX": "NO"},
             {"OPENFILEGDB_USE_SPATIAL_INDEX": None}],
        )

    def test_clip_and_normalize_accepts_gdb_zip(self):
        gdb = self.p("R.gdb")
        _write_gdb(_frame(["1"], crs=3857, extra={"Bldg_ID": ["9"]}), gdb)
        z = self.p("R.gdb.zip")
        _zip_dir_contents(gdb, z)
        out_path = self.p("out.gpkg")
        n = clip_and_normalize_user_footprints(z, AOI, out_path)
        self.assertEqual(n, 1)
        out = gpd.read_file(out_path)
        self.assertEqual(list(out.columns), ["id", "subtype", "class", "geometry"])
        self.assertEqual(out["id"].iloc[0], "9")


if __name__ == "__main__":
    unittest.main()
