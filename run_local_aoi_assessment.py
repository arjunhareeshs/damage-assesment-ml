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
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HASTE_FORK = BASE_DIR / "Haste-DTF" / "haste-fork"
DTF_REPO = BASE_DIR / "Haste-DTF" / "Damage-TriageFormer"

sys.path.insert(0, str(HASTE_FORK / "hastelib" / "src"))
sys.path.insert(0, str(DTF_REPO))

import os

os.environ["DTF_REPO_ROOT"] = str(DTF_REPO)

DEFAULT_IMAGERY = BASE_DIR / "data" / "imagery" / "m_3510141_ne_14_060_20200915.tif"
DEFAULT_FOOTPRINTS = BASE_DIR / "data" / "footprint" / "Texas.geojson" / "Texas.geojson"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "local_aoi"


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


def load_buildings_in_scope(footprints_path, aoi_geojson_path, building_ids):
    """Filter the footprints file down to just the buildings in scope,
    mirroring the semantics of the server's
    ``postgis_query.buildings_in_aoi`` (ST_Intersects) - just done locally
    with geopandas instead of a PostGIS query."""
    import geopandas as gpd
    from shapely.geometry import mapping, shape

    gdf = gpd.read_file(footprints_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if "id" not in gdf.columns:
        gdf["id"] = gdf.index.astype(str)
    else:
        gdf["id"] = gdf["id"].astype(str)

    if building_ids:
        wanted = {str(b) for b in building_ids}
        scoped = gdf[gdf["id"].isin(wanted)]
    else:
        with open(aoi_geojson_path) as f:
            aoi_geojson = json.load(f)
        aoi_geom = shape(
            aoi_geojson["geometry"] if "geometry" in aoi_geojson else aoi_geojson
        )
        scoped = gdf[gdf.geometry.intersects(aoi_geom)]

    return [
        {"building_id": row["id"], "geometry": mapping(row.geometry)}
        for _, row in scoped.iterrows()
    ]


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
    args = parser.parse_args()

    if not args.aoi and not args.building_ids:
        print("[ERROR] Provide either --aoi <file.geojson> or --building-ids id1,id2,...")
        sys.exit(1)

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

    building_ids = args.building_ids.split(",") if args.building_ids else None
    buildings = load_buildings_in_scope(footprints_path, args.aoi, building_ids)
    if not buildings:
        print("[ERROR] No buildings found in the given AOI / id list.")
        sys.exit(1)
    print(f"[+] {len(buildings)} building(s) in scope")

    checkpoint_path, checkpoint_config_path = get_local_checkpoint()
    print(f"[+] Checkpoint: {checkpoint_path}")

    from hastegeo.workflows.damage_triage_aoi_infer import (
        run_damage_triage_aoi,
        write_predictions_gpkg,
    )

    results = run_damage_triage_aoi(
        image_path=str(imagery_path),
        buildings=buildings,
        checkpoint_path=checkpoint_path,
        checkpoint_config_path=checkpoint_config_path,
        output_dir=str(output_dir),
        pad_px=args.pad_px,
    )

    gpkg_path = output_dir / "local_aoi_predictions.gpkg"
    write_predictions_gpkg(buildings, results, str(gpkg_path))

    manifest_path = output_dir / "local_aoi_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "num_buildings": len(buildings),
                "num_classified": sum(1 for r in results if r["class"] != -1),
                "buildings": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print(f" DONE. {len(results)} building(s) classified.")
    print(f" GeoPackage: {gpkg_path}")
    print(f" Manifest:   {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
