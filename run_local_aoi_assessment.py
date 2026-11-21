"""Run DamageTriageFormer building-by-building over an AOI, entirely locally
(no Azure batch queue, no PostGIS, no Docker) - the local/standalone
counterpart of ``hastegeo.workflows.damage_triage_aoi_infer``'s server job.

For each building whose footprint intersects the given AOI polygon, this:
  1. crops the imagery around just that building,
  2. runs one DamageTriageFormer forward pass for it,
  3. records its damage class + confidence,
using ``run_damage_triage_aoi`` (loads the model once, then loops building by
building) - the same function the server-side per-building path calls.

Usage:
    python run_local_aoi_assessment.py --aoi path/to/aoi.geojson
    python run_local_aoi_assessment.py --building-ids 12,47,103

Requires the model weights to already be downloaded locally:
    python download_models.py

With no arguments it scores buildings in a 500 m box at the centre of the
default imagery tile and writes GPKG + GeoParquet + manifest to
data/outputs/local_aoi/.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HASTE_FORK = BASE_DIR / "Haste-DTF" / "haste-fork"
# DamageTriageFormer model code (the single vendored copy HASTE also uses).
DTF_REPO = HASTE_FORK / "external" / "damage-triage-former"

sys.path.insert(0, str(HASTE_FORK / "hastelib" / "src"))
sys.path.insert(0, str(DTF_REPO))

import os

os.environ["DTF_REPO_ROOT"] = str(DTF_REPO)

DEFAULT_IMAGERY = (
    BASE_DIR / "data" / "data" / "imagery" / "naip_2020" / "m_3510141_ne_14_060_20200915.tif"
)
DEFAULT_FOOTPRINTS = BASE_DIR / "data" / "data" / "footprints" / "Texas.geojson"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "local_aoi"
# Extra footprints read around the AOI (~200 m) so a small AOI still has
# enough buildings for the imagery alignment to measure on.
ALIGN_MARGIN_DEG = 0.002


def get_local_checkpoint():
    damage_triage_dir = BASE_DIR / "models" / "damage_triage"
    checkpoint_path = damage_triage_dir / "best_model.pth"
    config_path = damage_triage_dir / "config.json"
    if not checkpoint_path.exists() or not config_path.exists():
        print(
            "[ERROR] Model checkpoint not found in models/damage_triage/.\n"
            "        Run `python download_models.py` first."
        )
        sys.exit(1)
    return str(checkpoint_path), str(config_path)


def imagery_bounds_4326(imagery_path):
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(imagery_path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def default_aoi(imagery_path, out_path, size_m=500):
    """A ``size_m`` square at the imagery centre, saved as GeoJSON."""
    from pyproj import Transformer
    import rasterio
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    with rasterio.open(imagery_path) as src:
        cx = (src.bounds.left + src.bounds.right) / 2
        cy = (src.bounds.bottom + src.bounds.top) / 2
        to_4326 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform
    h = size_m / 2
    geom = transform(to_4326, box(cx - h, cy - h, cx + h, cy + h))
    with open(out_path, "w") as f:
        json.dump(mapping(geom), f)
    return str(out_path)


def load_buildings_in_scope(
    footprints_path, aoi_geojson_path, building_ids, imagery_path=None, align=True
):
    """The buildings to assess, aligned to the imagery.

    1. Read the footprints around the area (bbox read: the statewide file is
       ~3 GB and must never be loaded whole). A margin around the AOI is
       read too, so the alignment below has enough buildings to measure on.
    2. Shift the footprints onto the roofs of ``imagery_path``
       (``footprint_alignment``) -- before choosing buildings, because
       their true position decides which ones the AOI line crosses.
    3. Keep every building the AOI touches, even partly, as the WHOLE
       building (same ``ST_Intersects`` rule as the server's
       ``postgis_query.buildings_in_aoi``; geometry is never clipped).

    Returns ``(buildings, shift)``; ``shift`` is ``None`` when ``align`` is off.
    """
    import geopandas as gpd
    from shapely.geometry import mapping, shape

    aoi_geom = None
    if building_ids:
        bbox = imagery_bounds_4326(imagery_path)
    else:
        with open(aoi_geojson_path) as f:
            aoi_gj = json.load(f)
        aoi_geom = shape(aoi_gj["geometry"] if "geometry" in aoi_gj else aoi_gj)
        bbox = aoi_geom.bounds
    margin = ALIGN_MARGIN_DEG if align else 0.0
    read_bbox = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
    print(f"[+] Reading footprints in bbox {tuple(round(b, 5) for b in read_bbox)} ...")
    gdf = gpd.read_file(footprints_path, bbox=read_bbox, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if "id" in gdf.columns:
        gdf["id"] = gdf["id"].astype(str)
    else:
        # The footprint file carries no building id, and a row number would
        # change with every AOI. Derive a stable id from the footprint's own
        # (unshifted) location: the same building gets the same id every run.
        c = gdf.geometry.representative_point()
        gdf["id"] = [f"{x:.6f}_{y:.6f}" for x, y in zip(c.x, c.y)]
        gdf = gdf.drop_duplicates(subset="id")

    shift = None
    if align and len(gdf):
        from hastegeo.core.utils.footprint_alignment import align_footprints_to_raster

        try:
            gdf, shift = align_footprints_to_raster(str(imagery_path), gdf)
        except Exception as exc:  # never let alignment stop an assessment
            from hastegeo.core.utils.footprint_alignment import FootprintShift

            shift = FootprintShift(reason=f"alignment failed: {exc}")
        if shift.applied:
            print(
                f"[+] Footprints aligned to imagery: moved {shift.dx_m:+.2f} m east, "
                f"{shift.dy_m:+.2f} m north ({shift.windows_used} windows agree "
                f"within {shift.spread_m:.2f} m)"
            )
        else:
            print(f"[+] Footprints not shifted: {shift.reason}")

    if building_ids:
        wanted = {str(b) for b in building_ids}
        scoped = gdf[gdf["id"].isin(wanted)]
    else:
        scoped = gdf[gdf.geometry.intersects(aoi_geom)]
        n_partial = int((~scoped.geometry.within(aoi_geom)).sum())
        print(
            f"[+] {len(scoped)} building(s) touch the AOI; {n_partial} cross its "
            "boundary and are assessed whole"
        )

    buildings = [
        {"building_id": row["id"], "geometry": mapping(row.geometry)}
        for _, row in scoped.iterrows()
    ]
    return buildings, shift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery", default=str(DEFAULT_IMAGERY))
    parser.add_argument("--footprints", default=str(DEFAULT_FOOTPRINTS))
    parser.add_argument("--aoi", help="Path to a GeoJSON AOI polygon file")
    parser.add_argument(
        "--building-ids", help="Comma-separated building ids to score directly, skipping AOI filtering"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pad-px", type=int, default=32)
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Use the footprints exactly as given (skip aligning them to the imagery)",
    )
    args = parser.parse_args()

    imagery_path = Path(args.imagery)
    footprints_path = Path(args.footprints)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not imagery_path.exists():
        print(f"[ERROR] Imagery file not found: {imagery_path}")
        sys.exit(1)
    if not footprints_path.exists():
        print(f"[ERROR] Footprints file not found: {footprints_path}")
        sys.exit(1)

    if not args.aoi and not args.building_ids:
        args.aoi = default_aoi(imagery_path, output_dir / "default_aoi.geojson")
        print(f"[+] No --aoi/--building-ids given; using a 500 m box at the imagery centre: {args.aoi}")

    building_ids = args.building_ids.split(",") if args.building_ids else None
    buildings, shift = load_buildings_in_scope(
        footprints_path, args.aoi, building_ids, str(imagery_path), align=not args.no_align
    )
    if not buildings:
        print("[ERROR] No buildings found in the given AOI / id list.")
        sys.exit(1)
    print(f"[+] {len(buildings)} building(s) in scope")

    checkpoint_path, checkpoint_config_path = get_local_checkpoint()
    print(f"[+] Checkpoint: {checkpoint_path}")

    from hastegeo.workflows.damage_triage_aoi_infer import (
        run_damage_triage_aoi,
        write_predictions_geoparquet,
        write_predictions_gpkg,
    )

    results = run_damage_triage_aoi(
        image_path=str(imagery_path),
        buildings=buildings,
        checkpoint_path=checkpoint_path,
        checkpoint_config_path=checkpoint_config_path,
        # Per-building crop/mask PNGs go in their own subfolder so the three
        # result files below are not buried among thousands of images.
        output_dir=str(output_dir / "building_images"),
        pad_px=args.pad_px,
    )

    gpkg_path = output_dir / "local_aoi_predictions.gpkg"
    write_predictions_gpkg(buildings, results, str(gpkg_path))

    parquet_path = output_dir / "local_aoi_predictions.parquet"
    n_parquet = write_predictions_geoparquet(buildings, results, str(parquet_path))

    manifest_path = output_dir / "local_aoi_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "num_buildings": len(buildings),
                "num_classified": sum(1 for r in results if r["class"] != -1),
                "predictions_parquet_filename": parquet_path.name,
                "building_images_dir": "building_images",
                "footprint_alignment": shift.as_dict() if shift else None,
                "buildings": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    n_ok = sum(1 for r in results if r["class"] != -1)
    print(f" DONE. {n_ok}/{len(results)} building(s) classified.")
    print(f" GeoPackage: {gpkg_path}")
    print(f" GeoParquet: {parquet_path}  ({n_parquet} rows)")
    print(f" Manifest:   {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
