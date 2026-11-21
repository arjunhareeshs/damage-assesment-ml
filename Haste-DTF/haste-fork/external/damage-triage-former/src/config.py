import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Dataset root: expects tiles_1024/{images,damage} and stratified_splits.json
# underneath. Override with the DINOV3_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("DINOV3_DATA_ROOT", "./data")
# Output root for runs/checkpoints. Override with DINOV3_RUNS_ROOT.
RUNS_ROOT = os.environ.get("DINOV3_RUNS_ROOT", "./runs")

# Data directories - use PNG images
IMAGE_DIR = os.path.join(DATA_ROOT, "combined_images_png")
LABEL_DIR = os.path.join(DATA_ROOT, "combined_labels")
MASK_DIR = os.path.join(DATA_ROOT, "combined_masks")
DAMAGE_DIR = os.path.join(DATA_ROOT, "combined_damage_png")

# Tile directories (1024x1024 cropped tiles)
TILE_IMAGE_DIR = os.path.join(DATA_ROOT, "tiles_1024", "images")
TILE_DAMAGE_DIR = os.path.join(DATA_ROOT, "tiles_1024", "damage")
TILE_SIZE = 1024

# Hyperparameters
IM_SIZE = 1024  # Use tiled images by default
BATCH_SIZE = 4  # Can increase batch size with smaller tiles
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Model Configuration
# BACKBONE is the DINOv3 ViT-L/16 backbone: a local directory when one is
# found (AutoModel/AutoConfig.from_pretrained accept a directory), otherwise
# a Hugging Face repo id (network fetch). Searched in this order:
#   1. DTF_BACKBONE_DIR                (HASTE's task runner sets this)
#   2. models/backbone/dinov3-vitl16/  then the flat models/backbone/
#      (config.json + model.safetensors directly inside) in every folder
#      above this file -- e.g. <repo>/models/backbone locally, /app/models
#      in the standalone AOI image
#   3. the same two layouts above the current working directory
#   4. fixed container locations
#   5. DTF_BACKBONE_REPO, else the public Hugging Face id
# A folder only counts when it holds config.json AND a weights file, so an
# interrupted download (empty or config-only folder) is never used.
from pathlib import Path

_HF_BACKBONE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _backbone_dir_ready(path):
    try:
        path = Path(path)
        return (
            path.is_dir()
            and (path / "config.json").is_file()
            and any(p.suffix in (".safetensors", ".bin") for p in path.iterdir())
        )
    except OSError:
        return False


def _backbone_candidates():
    env_dir = os.environ.get("DTF_BACKBONE_DIR", "")
    if env_dir:
        yield Path(env_dir)
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for folder in (start, *start.parents):
            yield folder / "models" / "backbone" / "dinov3-vitl16"
            yield folder / "models" / "backbone"
            # Legacy checkout layout kept from the earlier vendored copy.
            yield folder / "Damage assesment" / "damage-assesment-ml" / "models" / "backbone" / "dinov3-vitl16"
    yield Path("/opt/dtf-backbone")
    yield Path("/app/models/backbone/dinov3-vitl16")
    yield Path("/app/models/backbone")


def _find_local_backbone():
    for candidate in _backbone_candidates():
        if _backbone_dir_ready(candidate):
            return str(candidate)
    return None


_LOCAL_BACKBONE_DIR = _find_local_backbone()
_LOCAL_BACKBONE_READY = _LOCAL_BACKBONE_DIR is not None
BACKBONE = (
    _LOCAL_BACKBONE_DIR
    if _LOCAL_BACKBONE_READY
    else os.environ.get("DTF_BACKBONE_REPO", _HF_BACKBONE)
)
PATCH_SIZE = 16
USE_HIERARCHICAL = True  # Use hierarchical Type+Extent classification

# Categories for damage severity classification
# Note on class definitions:
#   - Partial Roof Damage: <50% of roof area damaged, structure intact
#   - Total Roof Damage: >50% of roof area damaged, structure intact
#   - Partial Structural Collapse: <50% structural damage
#   - Total Structural Collapse: >50% structural damage or complete collapse
#
# Classes 1 and 2 are visually harder to distinguish due to subtle damage thresholds
COLOR_TO_CAT = {
    (0, 0, 0): -1,         # none / background -> IGNORE
    (255, 255, 255): 0,    # white -> Undamaged (Class 0)
    (0, 255, 83): 1,       # Partial Roof Damage (green)
    (246, 255, 11): 2,     # Total Roof Damage (yellow) - RARE
    (255, 138, 18): 3,     # Partial Structural Collapse (orange) - RARE
    (255, 0, 0): 4         # Total Structural Collapse (red)
}

CATEGORIES = [
    "Undamaged",                    # Class 0
    "Partial Roof Damage",          # Class 1
    "Total Roof Damage",            # Class 2
    "Partial Structural Collapse",  # Class 3
    "Total Structural Collapse"     # Class 4
]

# Hierarchical classification labels
# Type: 0=No Damage, 1=Roof, 2=Structural
TYPE_CATEGORIES = ["No Damage", "Roof Damage", "Structural Collapse"]
# Extent: 0=Partial, 1=Total (Undefined for No Damage)
EXTENT_CATEGORIES = ["Partial", "Total"]

NUM_CLASSES = len(CATEGORIES)
IGNORE_INDEX = -1  # We map 0 (No Damage) to -1 (Ignore)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Processing
FILENAME_REGEX = r'(\d+\.\d+(?:[_A-Za-z0-9]*)?)$'

# Class weights (precomputed from dataset, inverse sqrt frequency)
# Higher weight = rarer class, will receive more gradient signal
# These can be recalculated with --new-data flag or by deleting class_weights.json
CLASS_WEIGHTS = None  # Loaded dynamically during training
