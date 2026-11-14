"""Download both models used by the local damage-assessment pipeline into a
dedicated project-local folder (``models/``), instead of the opaque global
Hugging Face cache (``~/.cache/huggingface/hub``).

This replaces the three previously-duplicated download code paths
(``step2_download_model_weights.py``, ``run_texas_assessment.py``'s
``download_hf_checkpoint()``, and the server-side
``damage_triage.py::_download_hf_checkpoint()``) for local/standalone use.
The server-side Azure batch-job copy is left untouched - it downloads inside
a remote container filesystem, a different environment than this local
``models/`` folder.

Downloads:
    1. Backbone  - DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m)
       -> models/backbone/dinov3-vitl16/
    2. Damage triage model - DamageTriageFormer (Ymx1025/DamageTriageFormer-model)
       -> models/damage_triage/  (best_model.pth, config.json)

Idempotent: re-running skips any file that's already present.
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BACKBONE_DIR = MODELS_DIR / "backbone" / "dinov3-vitl16"
DAMAGE_TRIAGE_DIR = MODELS_DIR / "damage_triage"

BACKBONE_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DTF_REPO = "Ymx1025/DamageTriageFormer-model"
DTF_FILES = ["best_model.pth", "config.json"]


def download_damage_triage_model():
    from huggingface_hub import hf_hub_download

    DAMAGE_TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/2] Damage triage model ({DTF_REPO}) -> {DAMAGE_TRIAGE_DIR}")

    for filename in DTF_FILES:
        target = DAMAGE_TRIAGE_DIR / filename
        if target.exists():
            print(f"      [=] {filename} already present, skipping")
            continue
        print(f"      [+] downloading {filename}...")
        hf_hub_download(
            repo_id=DTF_REPO,
            filename=filename,
            local_dir=str(DAMAGE_TRIAGE_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"      [OK] {filename} -> {target}")


def download_backbone_model():
    from transformers import AutoConfig, AutoModel

    has_config = (BACKBONE_DIR / "config.json").exists()
    has_weights = any(BACKBONE_DIR.glob("*.safetensors")) or any(
        BACKBONE_DIR.glob("*.bin")
    )
    already_present = has_config and has_weights
    print(f"\n[2/2] Backbone ({BACKBONE_REPO}) -> {BACKBONE_DIR}")

    if already_present:
        print("      [=] backbone already present, skipping")
        return

    print("      [+] downloading config + weights (this pulls ~1.2 GB)...")
    try:
        # Fetched into the global HF cache first; only written into our
        # local BACKBONE_DIR afterward (below), once we know it fully
        # succeeded - so a failed/interrupted attempt never leaves a
        # half-populated local directory that a later run would mistake
        # for "already downloaded" (see config.py's _LOCAL_BACKBONE_READY).
        config = AutoConfig.from_pretrained(BACKBONE_REPO)
        model = AutoModel.from_pretrained(BACKBONE_REPO, config=config)
    except Exception as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            print(
                "\n      [ERROR] facebook/dinov3-vitl16-pretrain-lvd1689m is a "
                "GATED Hugging Face repo - Meta requires an approved HF "
                "account to download it. To fix:\n"
                "        1. Log in to huggingface.co and request access at:\n"
                "           https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m\n"
                "        2. Once approved, authenticate this machine:\n"
                "           huggingface-cli login\n"
                "           (or set the HF_TOKEN environment variable to your token)\n"
                "        3. Re-run: python download_models.py\n"
                "        (The damage triage model, if already downloaded, will be "
                "skipped automatically on re-run.)"
            )
            sys.exit(1)
        raise
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(BACKBONE_DIR))
    config.save_pretrained(str(BACKBONE_DIR))
    print(f"      [OK] backbone saved -> {BACKBONE_DIR}")


def main():
    print("=" * 70)
    print(" DOWNLOADING MODELS INTO LOCAL FOLDER")
    print("=" * 70)

    try:
        import huggingface_hub  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("[ERROR] Required packages not found. Activate your venv / "
              "install requirements.txt first.")
        sys.exit(1)

    start = time.time()
    download_damage_triage_model()
    download_backbone_model()
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(f" DONE in {elapsed:.1f}s. Local model paths:")
    print(f"   Backbone:       {BACKBONE_DIR}")
    print(f"   Damage triage:  {DAMAGE_TRIAGE_DIR}")
    print(" Next: python run_local_aoi_assessment.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
