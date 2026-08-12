import os
import sys
import json
from pathlib import Path

# Add project roots to sys.path so all imports resolve seamlessly
BASE_DIR = Path(__file__).resolve().parent
HASTE_FORK = BASE_DIR / "Haste-DTF" / "haste-fork"
DTF_REPO = BASE_DIR / "Haste-DTF" / "Damage-TriageFormer"

sys.path.insert(0, str(HASTE_FORK / "hastelib" / "src"))
sys.path.insert(0, str(DTF_REPO))

# Set DTF REPO ROOT environment variable
os.environ["DTF_REPO_ROOT"] = str(DTF_REPO)

def get_local_checkpoint():
    """Look up the DamageTriageFormer checkpoint in the local models/ folder
    (populated by ``python download_models.py``)."""
    damage_triage_dir = BASE_DIR / "models" / "damage_triage"
    checkpoint_path = damage_triage_dir / "best_model.pth"
    config_path = damage_triage_dir / "config.json"

    if not checkpoint_path.exists() or not config_path.exists():
        print(
            "[ERROR] Model checkpoint not found in models/damage_triage/.\n"
            "        Run `python download_models.py` first to download it."
        )
        sys.exit(1)

    return str(checkpoint_path), str(config_path)

def main():
    print("=" * 70)
    print(" TEXAS BUILDING DAMAGE ASSESSMENT - STANDALONE INFERENCE RUNNER")
    print("=" * 70)
    
    # 1. Locate dataset files
    imagery_path = BASE_DIR / "data" / "imagery" / "m_3510141_ne_14_060_20200915.tif"
    footprints_path = BASE_DIR / "data" / "footprint" / "Texas.geojson" / "Texas.geojson"
    output_dir = BASE_DIR / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not imagery_path.exists():
        print(f"[ERROR] Imagery file not found at: {imagery_path}")
        sys.exit(1)
        
    if not footprints_path.exists():
        print(f"[ERROR] Footprints file not found at: {footprints_path}")
        sys.exit(1)

    print(f"[+] Imagery File Found: {imagery_path}")
    print(f"[+] Footprints File Found: {footprints_path}")

    # 2. Get Model Weights
    ckpt_path, ckpt_cfg_path = get_local_checkpoint()
    print(f"[+] Model Checkpoint Ready: {ckpt_path}")
    print(f"[+] Model Architecture Config Ready: {ckpt_cfg_path}")

    # 3. Run Inference via hastegeo workflow
    from hastegeo.workflows.damage_triage_infer import run_damage_triage
    
    print("\n[+] Starting DamageTriageFormer Inference Pipeline...")
    visualizer_prefix = str(output_dir / "texas_mosaic")
    
    gdf, _ = run_damage_triage(
        image_path=str(imagery_path),
        footprints_path=str(footprints_path),
        checkpoint_path=ckpt_path,
        checkpoint_config_path=ckpt_cfg_path,
        tile_size=1024,
        visualizer_output_prefix=visualizer_prefix
    )

    # 4. Save Prediction Results
    output_gpkg = output_dir / "texas_building_damage_predictions.gpkg"
    output_csv = output_dir / "texas_building_damage_predictions.csv"
    
    print(f"\n[+] Saving predictions to GeoPackage: {output_gpkg}")
    gdf.to_file(str(output_gpkg), layer="predictions", driver="GPKG")
    
    print(f"[+] Saving predictions summary to CSV: {output_csv}")
    df_summary = gdf[["id", "dtf_class", "dtf_class_name", "dtf_confidence"]].copy()
    df_summary.to_csv(str(output_csv), index=False)

    print("\n" + "=" * 70)
    print(" ASSESSMENT COMPLETE!")
    print(f" Total Buildings Processed: {len(gdf)}")
    print(f" Results Saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
