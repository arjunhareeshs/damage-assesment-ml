import os
import sys
from pathlib import Path

# Repository root and path bootstrapping
REPO_ROOT = Path(__file__).resolve().parent.parent
DTF_REPO = REPO_ROOT / "Haste-DTF" / "Damage-TriageFormer"

# Inject DTF_REPO into sys.path to enable src.* imports
if str(DTF_REPO) not in sys.path:
    sys.path.insert(0, str(DTF_REPO))

# Model Checkpoint and Configuration default paths
CKPT_PATH = Path(os.getenv("IDTF_CHECKPOINT", REPO_ROOT / "models" / "damage_triage" / "best_model.pth"))
CKPT_CONFIG = Path(os.getenv("IDTF_CHECKPOINT_CONFIG", REPO_ROOT / "models" / "damage_triage" / "config.json"))

# Image constants
IM_SIZE = 1024

# Damage Categories (matching model checkpoint index order)
CATEGORIES = [
    "Undamaged",
    "Partial Roof Damage",
    "Total Roof Damage",
    "Partial Structural Collapse",
    "Total Structural Collapse",
]

# Color Palette for damage classes (RGB tuples, 0-255)
# Index 0: Undamaged (White)
# Index 1: Partial Roof Damage (Light Peach/Orange)
# Index 2: Total Roof Damage (Orange Red)
# Index 3: Partial Structural Collapse (Bright Red)
# Index 4: Total Structural Collapse (Dark Maroon)
CLASS_COLORS = [
    (255, 255, 255),
    (252, 190, 165),
    (251, 112, 80),
    (211, 32, 32),
    (103, 0, 13),
]
