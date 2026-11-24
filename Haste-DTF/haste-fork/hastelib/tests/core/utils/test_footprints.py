# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for hastegeo.core.utils.footprints.

Originally lived in docker/training/code/bda/test_footprints.py; relocated
here when the Overture Maps client moved into hastegeo so the imageryprep
workflow could share it.
"""

import unittest
from unittest.mock import MagicMock, patch

from hastegeo.core.utils.footprints import (
    FALLBACK_RELEASE,
    _dataset_path,
    get_latest_release,
)


class TestGetLatestRelease(unittest.TestCase):
    """Tests for get_latest_release()."""

    def setUp(self):
        # Clear the lru_cache before each test so mocks take effect
        get_latest_release.cache_clear()

    def tearDown(self):
        get_latest_release.cache_clear()

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_returns_latest_release(self, mock_filesystem):
        """Should return the most recent release by lexicographic sort."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = [
            "release/2025-03-01.0",
            "release/2025-06-15.0",
            "release/2026-01-10.0",
            "release/2026-02-18.0",
        ]
        mock_filesystem.return_value = mock_fs

        result = get_latest_release()

        self.assertEqual(result, "2026-02-18.0")
        mock_filesystem.assert_called_once_with(
            "az", account_name="overturemapswestus2", anon=True
        )
        mock_fs.ls.assert_called_once_with("release/")

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_returns_latest_when_unordered(self, mock_filesystem):
        """Should sort correctly even if blob listing is unordered."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = [
            "release/2026-02-18.0",
            "release/2025-03-01.0",
            "release/2026-06-01.0",
            "release/2025-12-15.1",
        ]
        mock_filesystem.return_value = mock_fs

        result = get_latest_release()

        self.assertEqual(result, "2026-06-01.0")

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_filters_invalid_entries(self, mock_filesystem):
        """Should ignore entries that don't match the release name pattern."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = [
            "release/2025-06-15.0",
            "release/readme.txt",
            "release/.metadata",
            "release/2026-02-18.0",
        ]
        mock_filesystem.return_value = mock_fs

        result = get_latest_release()

        self.assertEqual(result, "2026-02-18.0")

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_fallback_on_empty_listing(self, mock_filesystem):
        """Should return FALLBACK_RELEASE when no valid releases are found."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = []
        mock_filesystem.return_value = mock_fs

        result = get_latest_release()

        self.assertEqual(result, FALLBACK_RELEASE)

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_fallback_on_exception(self, mock_filesystem):
        """Should return FALLBACK_RELEASE when blob listing raises."""
        mock_filesystem.side_effect = Exception("Network error")

        result = get_latest_release()

        self.assertEqual(result, FALLBACK_RELEASE)

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_handles_trailing_slashes(self, mock_filesystem):
        """Should handle entries with trailing slashes from blob listing."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = [
            "release/2025-06-15.0/",
            "release/2026-02-18.0/",
        ]
        mock_filesystem.return_value = mock_fs

        result = get_latest_release()

        self.assertEqual(result, "2026-02-18.0")


class TestDatasetPath(unittest.TestCase):
    """Tests for _dataset_path()."""

    def setUp(self):
        get_latest_release.cache_clear()

    def tearDown(self):
        get_latest_release.cache_clear()

    def test_explicit_release(self):
        """Should use the explicitly provided release version."""
        path = _dataset_path("building", release="2025-01-01.0")
        self.assertEqual(
            path, "release/2025-01-01.0/theme=buildings/type=building/"
        )

    @patch("hastegeo.core.utils.footprints.fsspec.filesystem")
    def test_dynamic_release(self, mock_filesystem):
        """Should resolve version dynamically when release is None."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = [
            "release/2026-03-01.0",
            "release/2026-06-15.0",
        ]
        mock_filesystem.return_value = mock_fs

        path = _dataset_path("building")

        self.assertEqual(
            path, "release/2026-06-15.0/theme=buildings/type=building/"
        )

    def test_all_type_theme_mappings(self):
        """Every type in type_theme_map should produce a valid path."""
        from hastegeo.core.utils.footprints import type_theme_map

        for overture_type, theme in type_theme_map.items():
            path = _dataset_path(overture_type, release="2026-01-01.0")
            self.assertEqual(
                path,
                f"release/2026-01-01.0/theme={theme}/type={overture_type}/",
            )


class TestDownloadBuildingFootprints(unittest.TestCase):
    """Tests for download_building_footprints()."""

    def test_rejects_non_gpkg_output_path(self):
        from hastegeo.core.utils.footprints import download_building_footprints

        with self.assertRaises(ValueError):
            download_building_footprints(
                bbox=(0, 0, 1, 1),
                output_path="/tmp/foo.geojson",
            )

    @staticmethod
    def _batches(gdf):
        """Stand-in for record_batch_reader: the frame as Arrow batches."""
        import pyarrow as pa

        table = pa.table(gdf.to_arrow(geometry_encoding="WKB"))
        return lambda *a, **k: iter(table.to_batches(max_chunksize=1))

    # download_building_footprints reads through _overture_batches (footer
    # pruning); these tests stand in for it with the frame's batches.

    def _run(self, gdf, **kw):
        import os
        import tempfile

        import geopandas as gpd
        from hastegeo.core.utils import footprints

        with patch.object(footprints, "_overture_batches", side_effect=self._batches(gdf)) as reader, \
                patch.object(footprints, "get_latest_release", return_value="2026-02-18.0"), \
                tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "out.gpkg")
            count = footprints.download_building_footprints(output_path=output, **kw)
            written = gpd.read_file(output)
        return count, written, reader

    def _frame(self, ids, boxes):
        import geopandas as gpd
        import shapely.geometry

        return gpd.GeoDataFrame(
            {"id": ids, "subtype": [""] * len(ids), "class": [""] * len(ids)},
            geometry=[shapely.geometry.box(*b) for b in boxes],
            crs="EPSG:4326",
        )

    def test_filters_to_polygons_and_writes(self):
        import geopandas as gpd
        import shapely.geometry

        gdf = gpd.GeoDataFrame(
            {"id": ["a", "b", "pt"], "subtype": [""] * 3, "class": [""] * 3},
            geometry=[
                shapely.geometry.box(0, 0, 1, 1),
                shapely.geometry.box(2, 2, 3, 3),
                shapely.geometry.Point(5, 5),
            ],
            crs="EPSG:4326",
        )
        count, written, reader = self._run(gdf, bbox=(0, 0, 10, 10))
        self.assertEqual(count, 2)
        self.assertEqual(set(written["id"]), {"a", "b"})
        self.assertEqual(written.crs.to_epsg(), 4326)
        self.assertEqual(reader.call_args.kwargs["release"], "2026-02-18.0")
        self.assertEqual(
            reader.call_args.kwargs["columns"],
            ["id", "geometry", "subtype", "class", "sources"],
        )
        # Provenance column is always written (None when Overture gives none).
        self.assertIn("source_dataset", written.columns)

    def test_geometry_source_dataset(self):
        from hastegeo.core.utils.footprints import _geometry_source_dataset

        self.assertEqual(
            _geometry_source_dataset([
                {"property": "/properties/height", "dataset": "OpenStreetMap"},
                {"property": "", "dataset": "Microsoft ML Buildings"},
            ]),
            "Microsoft ML Buildings",
        )
        self.assertEqual(
            _geometry_source_dataset([{"property": None, "dataset": "OpenStreetMap"}]),
            "OpenStreetMap",
        )
        self.assertIsNone(_geometry_source_dataset(None))
        self.assertIsNone(_geometry_source_dataset([]))

    def test_aoi_polygon_filters_buildings_outside_mask(self):
        """Footprints not intersecting the AOI polygon are dropped (the
        Overture query is bbox-only)."""
        import shapely.geometry

        gdf = self._frame(
            ["in-1", "in-2", "out-1"],
            [(1, 1, 1.5, 1.5), (4, 4, 4.5, 4.5), (7, 7, 7.5, 7.5)],
        )
        aoi = shapely.geometry.box(0.0, 0.0, 5.0, 10.0)
        count, written, _ = self._run(gdf, bbox=(0, 0, 10, 10), aoi_polygon=aoi)
        self.assertEqual(count, 2)
        self.assertEqual(set(written["id"]), {"in-1", "in-2"})

    def test_aoi_polygon_keeps_straddling_buildings_whole(self):
        """A footprint crossing the AOI boundary is kept whole, not clipped."""
        import shapely.geometry

        gdf = self._frame(["straddle"], [(4, 4, 6, 6)])
        aoi = shapely.geometry.box(0.0, 0.0, 5.0, 10.0)
        count, written, _ = self._run(gdf, bbox=(0, 0, 10, 10), aoi_polygon=aoi)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(written.geometry.iloc[0].area, 4.0)

    def test_disjoint_regions_read_in_one_pass(self):
        """Two imagery regions far apart are read in ONE scan (the release is
        scanned once) whose filter is both regions' bounding boxes."""
        import shapely.geometry

        gdf = self._frame(["a", "b"], [(0.1, 0.1, 0.2, 0.2), (50.1, 50.1, 50.2, 50.2)])
        aoi = shapely.geometry.MultiPolygon(
            [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(50, 50, 51, 51)]
        )
        count, written, reader = self._run(gdf, bbox=(0, 0, 51, 51), aoi_polygon=aoi)
        self.assertEqual(reader.call_count, 1)
        boxes = sorted(reader.call_args.args[1])  # one pass, both boxes
        self.assertEqual(boxes, [(0.0, 0.0, 1.0, 1.0), (50.0, 50.0, 51.0, 51.0)])
        self.assertEqual(count, 2)
        self.assertEqual(sorted(written["id"]), ["a", "b"])

    def test_multi_box_filter_is_an_or_of_boxes(self):
        """The combined filter keeps rows in either box and drops rows in
        neither (checked on a real Arrow table, not a mock)."""
        import pyarrow as pa
        import pyarrow.dataset as pads
        from unittest.mock import patch as _patch

        from hastegeo.core.utils import footprints

        table = pa.table({
            "id": ["in-a", "in-b", "between"],
            "bbox": pa.array(
                [{"xmin": 0.1, "ymin": 0.1, "xmax": 0.2, "ymax": 0.2},
                 {"xmin": 50.1, "ymin": 50.1, "xmax": 50.2, "ymax": 50.2},
                 {"xmin": 25.0, "ymin": 25.0, "xmax": 25.1, "ymax": 25.1}]),
            "geometry": pa.array([b"", b"", b""], pa.binary()),
        })
        captured = {}
        real = pads.dataset(table)

        def fake_dataset(*a, **k):
            return real

        class Spy:
            def to_batches(self, filter=None, columns=None):
                captured["rows"] = real.to_table(filter=filter).column("id").to_pylist()
                return iter([])

            schema = real.schema

        with _patch.object(footprints.ds, "dataset", return_value=Spy()), \
                _patch.object(footprints, "_dataset_path", return_value="x/"), \
                _patch.object(footprints.fsspec, "filesystem"), \
                _patch.object(footprints.fs, "PyFileSystem"):
            footprints.record_batch_reader(
                "building", [(0, 0, 1, 1), (50, 50, 51, 51)], release="r"
            )
        self.assertEqual(sorted(captured["rows"]), ["in-a", "in-b"])

    def test_overture_reader_prunes_row_groups_and_filters_rows(self):
        """_overture_batches on real parquet files (in-memory filesystem):
        row groups outside every box are never read, rows are filtered
        exactly, and two boxes are served in one pass."""
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq
        import shapely
        from unittest.mock import patch as _patch

        from hastegeo.core.utils import footprints

        mem = fsspec.filesystem("memory")
        root = "/overture/theme=buildings/type=building/"

        def write(name, rows):
            geoms = [shapely.box(x, y, x + 0.01, y + 0.01) for _, x, y in rows]
            table = pa.table({
                "id": [r[0] for r in rows],
                "geometry": pa.array([shapely.to_wkb(g) for g in geoms], pa.binary()),
                "bbox": pa.array([{"xmin": x, "ymin": y, "xmax": x + 0.01, "ymax": y + 0.01}
                                  for _, x, y in rows]),
                "subtype": ["residential"] * len(rows),
                "class": [None] * len(rows),
            })
            with mem.open(root + name, "wb") as f:
                pq.write_table(table, f, row_group_size=2)  # 2 rows per row group

        # file a: row group 0 near box A, row group 1 far away
        write("part-a.parquet", [("a1", 0.1, 0.1), ("a-edge", 0.95, 0.1), ("far1", 30, 30), ("far2", 31, 31)])
        # file b: only near box B; file c: nowhere near either box
        write("part-b.parquet", [("b1", 50.1, 50.1), ("b-out", 50.5, 52.0)])
        write("part-c.parquet", [("c1", -40, -40), ("c2", -41, -41)])
        reads = []
        orig = pq.ParquetFile.read_row_groups

        def spy(self, row_groups, *a, **k):
            reads.append(list(row_groups))
            return orig(self, row_groups, *a, **k)

        mem_open = mem.open
        with _patch.object(footprints.fsspec, "filesystem", return_value=mem), \
                _patch.object(footprints, "_dataset_path", return_value=root), \
                _patch.object(mem, "open", side_effect=lambda p, mode="rb", **k: mem_open(p, mode)), \
                _patch.object(pq.ParquetFile, "read_row_groups", spy):
            batches = list(footprints._overture_batches(
                "building", [(0, 0, 1, 1), (50, 50, 51, 51)], release="r",
                columns=["id", "geometry", "subtype", "class"],
            ))
        ids = sorted(i for b in batches for i in b.column("id").to_pylist())
        self.assertEqual(ids, ["a-edge", "a1", "b1"])  # b-out shares b1's row group but is outside
        self.assertEqual(sorted(reads), [[0], [0]])      # one row group from a, one from b, none from c
        self.assertEqual(batches[0].schema.names, ["id", "geometry", "subtype", "class"])
        self.assertEqual(
            batches[0].schema.field("geometry").metadata,
            {b"ARROW:extension:name": b"geoarrow.wkb"},
        )

    def test_nothing_found_still_writes_an_empty_file(self):
        gdf = self._frame([], [])
        count, written, _ = self._run(gdf, bbox=(0, 0, 1, 1))
        self.assertEqual(count, 0)
        self.assertEqual(len(written), 0)


class TestClipAndNormalizeUserFootprints(unittest.TestCase):
    """Tests for clip_and_normalize_user_footprints().

    Uses small in-memory GeoDataFrames written to temporary .gpkg files
    to exercise the real geopandas/fiona/shapely path. We deliberately
    avoid mocking these so we catch issues with CRS handling, geometry
    repair, clipping, and the .gpkg roundtrip — the failure modes most
    likely to surprise users supplying ad-hoc inputs."""

    @classmethod
    def setUpClass(cls):
        try:
            import geopandas  # noqa: F401
            import shapely  # noqa: F401
        except ImportError as e:  # pragma: no cover - env without GIS
            raise unittest.SkipTest(f"geopandas/shapely not available: {e}")

    def _aoi(self):
        import shapely.geometry

        return shapely.geometry.Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        )

    def _write_input(self, dst_dir, gdf, name="in.gpkg"):
        import os

        path = os.path.join(dst_dir, name)
        gdf.to_file(path, driver="GPKG")
        return path

    def test_happy_path_preserves_overlapping_polygons(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        inside = shapely.geometry.box(2, 2, 3, 3)
        partial = shapely.geometry.box(8, 8, 12, 12)
        outside = shapely.geometry.box(20, 20, 21, 21)
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b", "c"],
                "subtype": ["residential", "commercial", "residential"],
                "class": ["house", "office", "house"],
                "geometry": [inside, partial, outside],
            },
            crs="EPSG:4326",
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf)
            output_path = os.path.join(tmp, "out.gpkg")
            count = clip_and_normalize_user_footprints(
                input_path, self._aoi(), output_path
            )

            self.assertEqual(count, 2)
            out = gpd.read_file(output_path)
            self.assertEqual(len(out), 2)
            self.assertEqual(out.crs.to_epsg(), 4326)
            self.assertEqual(
                set(out.columns), {"id", "geometry", "subtype", "class"}
            )

    def test_reprojects_from_web_mercator(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        # Build a polygon in 4326 around (0.001, 0.001) so it falls inside
        # our AOI box (0,0)-(10,10), then project to Web Mercator before
        # writing so the helper has to call to_crs.
        small_4326 = shapely.geometry.box(0.001, 0.001, 0.002, 0.002)
        gdf_4326 = gpd.GeoDataFrame(
            {"id": ["a"], "geometry": [small_4326]}, crs="EPSG:4326"
        )
        gdf_3857 = gdf_4326.to_crs(epsg=3857)

        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf_3857)
            output_path = os.path.join(tmp, "out.gpkg")
            count = clip_and_normalize_user_footprints(
                input_path, self._aoi(), output_path
            )

            self.assertEqual(count, 1)
            out = gpd.read_file(output_path)
            self.assertEqual(out.crs.to_epsg(), 4326)
            # Geometry should be close to the original 4326 box, not the
            # huge mercator coords.
            self.assertLess(out.geometry.iloc[0].bounds[2], 1.0)

    def test_synthesizes_missing_id_subtype_class(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        polys = [
            shapely.geometry.box(1, 1, 2, 2),
            shapely.geometry.box(3, 3, 4, 4),
        ]
        gdf = gpd.GeoDataFrame(
            {"geometry": polys}, crs="EPSG:4326"
        )  # no id/subtype/class

        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf)
            output_path = os.path.join(tmp, "out.gpkg")
            count = clip_and_normalize_user_footprints(
                input_path, self._aoi(), output_path
            )

            self.assertEqual(count, 2)
            out = gpd.read_file(output_path)
            # id derived from geometry (stable across re-clips), others null
            ids = out["id"].astype(str).tolist()
            self.assertEqual(len(set(ids)), 2)
            self.assertTrue(all(len(i) == 40 for i in ids))
            # Same input clipped again -> the same ids.
            again = os.path.join(tmp, "again.gpkg")
            clip_and_normalize_user_footprints(input_path, self._aoi(), again)
            self.assertEqual(
                sorted(ids), sorted(gpd.read_file(again)["id"].astype(str))
            )
            self.assertTrue(out["subtype"].isna().all())
            self.assertTrue(out["class"].isna().all())

    def test_normalizes_int_ids_to_string(self):
        """User GPKGs with integer id columns (common in GIS data) must
        be cast to string on write. Without this the downstream join
        in GetValidationReport / GetAssessmentReport fails universally:
        labels_dict has string keys (JSON object keys are always
        strings on the wire), but a fiona-read int-id GPKG produces
        an int-keyed lookup table → matched == 0 even when labels are
        in the inference set."""
        import os
        import tempfile

        try:
            import fiona
        except ImportError:
            self.skipTest("fiona not installed in local environment")
        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        # User-supplied GPKG with integer building IDs
        gdf = gpd.GeoDataFrame(
            {
                "id": [101, 202, 303],
                "geometry": [
                    shapely.geometry.box(1, 1, 2, 2),
                    shapely.geometry.box(3, 3, 4, 4),
                    shapely.geometry.box(5, 5, 6, 6),
                ],
                "subtype": [None] * 3,
                "class": [None] * 3,
            },
            crs="EPSG:4326",
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf)
            output_path = os.path.join(tmp, "out.gpkg")
            count = clip_and_normalize_user_footprints(
                input_path, self._aoi(), output_path
            )
            self.assertEqual(count, 3)

            # Values preserved
            out = gpd.read_file(output_path)
            self.assertEqual(sorted(out["id"].tolist()), ["101", "202", "303"])

            # Critical: the on-disk type is string, so fiona reads it
            # back as string — matches what the JSON object-key path
            # used by the validation labels produces.
            with fiona.open(output_path) as src:
                ids = [feat["properties"]["id"] for feat in src]
            self.assertTrue(
                all(isinstance(v, str) for v in ids),
                f"Expected all str ids, got types: "
                f"{[type(v).__name__ for v in ids]}",
            )

    def test_drops_non_polygon_features(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        gdf = gpd.GeoDataFrame(
            {
                "id": ["p", "ln", "pt"],
                "geometry": [
                    shapely.geometry.box(1, 1, 2, 2),
                    shapely.geometry.LineString([(3, 3), (4, 4)]),
                    shapely.geometry.Point(5, 5),
                ],
            },
            crs="EPSG:4326",
        )

        with tempfile.TemporaryDirectory() as tmp:
            # GPKG doesn't accept heterogeneous geometries; write each
            # feature to its own GeoJSON and merge them back into a GPKG.
            paths = []
            for i, row in gdf.iterrows():
                p = os.path.join(tmp, f"row_{i}.geojson")
                gpd.GeoDataFrame(
                    [row], crs=gdf.crs, geometry="geometry"
                ).to_file(p, driver="GeoJSON")
                paths.append(p)
            combined = gpd.GeoDataFrame(
                gpd.pd.concat(
                    [gpd.read_file(p) for p in paths], ignore_index=True
                ),
                crs="EPSG:4326",
            )
            input_gpkg = os.path.join(tmp, "mixed.gpkg")
            combined.to_file(input_gpkg, driver="GPKG")
            output_path = os.path.join(tmp, "out.gpkg")

            count = clip_and_normalize_user_footprints(
                input_gpkg, self._aoi(), output_path
            )

            # Only the polygon survives the polygon-only filter.
            self.assertEqual(count, 1)
            out = gpd.read_file(output_path)
            self.assertTrue(
                out.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
            )

    def test_fully_outside_aoi_raises(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        far = shapely.geometry.box(100, 100, 101, 101)
        gdf = gpd.GeoDataFrame(
            {"id": ["x"], "geometry": [far]}, crs="EPSG:4326"
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf)
            output_path = os.path.join(tmp, "out.gpkg")
            with self.assertRaises(ValueError):
                clip_and_normalize_user_footprints(
                    input_path, self._aoi(), output_path
                )

    def test_no_polygons_raises(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b"],
                "geometry": [
                    shapely.geometry.LineString([(1, 1), (2, 2)]),
                    shapely.geometry.LineString([(3, 3), (4, 4)]),
                ],
            },
            crs="EPSG:4326",
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, "in.gpkg")
            gdf.to_file(input_path, driver="GPKG")
            output_path = os.path.join(tmp, "out.gpkg")
            with self.assertRaises(ValueError) as cm:
                clip_and_normalize_user_footprints(
                    input_path, self._aoi(), output_path
                )
            self.assertIn("Polygon", str(cm.exception))

    def test_missing_crs_raises(self):
        """The helper must reject inputs without a CRS rather than silently
        treating them as 4326. Stripping CRS from a read-back GeoJSON is
        unreliable across drivers, so we patch :func:`geopandas.read_file`
        to return a CRS-less GeoDataFrame directly."""
        import os
        import tempfile
        from unittest.mock import patch

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        nocrs = gpd.GeoDataFrame(
            {
                "id": ["a"],
                "geometry": [shapely.geometry.box(1, 1, 2, 2)],
            },
            crs=None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, "in.gpkg")
            # Just touch the input path so the helper's existence check
            # (if any) doesn't fire; the read is mocked.
            with open(input_path, "wb"):
                pass
            output_path = os.path.join(tmp, "out.gpkg")
            with patch(
                "hastegeo.core.utils.footprints.gpd.read_file",
                return_value=nocrs,
            ):
                with self.assertRaises(ValueError) as cm:
                    clip_and_normalize_user_footprints(
                        input_path, self._aoi(), output_path
                    )
            self.assertIn("CRS", str(cm.exception))

    def test_overwrite_false_rejects_existing_output(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        gdf = gpd.GeoDataFrame(
            {
                "id": ["a"],
                "geometry": [shapely.geometry.box(1, 1, 2, 2)],
            },
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf)
            output_path = os.path.join(tmp, "out.gpkg")
            with open(output_path, "w") as f:
                f.write("existing")
            with self.assertRaises(FileExistsError):
                clip_and_normalize_user_footprints(
                    input_path, self._aoi(), output_path
                )

    def test_overwrite_true_replaces_existing_output(self):
        import os
        import tempfile

        import geopandas as gpd
        import shapely.geometry
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        gdf = gpd.GeoDataFrame(
            {
                "id": ["a"],
                "geometry": [shapely.geometry.box(1, 1, 2, 2)],
            },
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._write_input(tmp, gdf)
            output_path = os.path.join(tmp, "out.gpkg")
            with open(output_path, "w") as f:
                f.write("existing")
            count = clip_and_normalize_user_footprints(
                input_path, self._aoi(), output_path, overwrite=True
            )
            self.assertEqual(count, 1)

    def test_rejects_non_gpkg_output_path(self):
        from hastegeo.core.utils.footprints import (
            clip_and_normalize_user_footprints,
        )

        with self.assertRaises(ValueError):
            clip_and_normalize_user_footprints(
                "/tmp/in.gpkg",
                self._aoi(),
                "/tmp/out.geojson",
            )


if __name__ == "__main__":
    unittest.main()
