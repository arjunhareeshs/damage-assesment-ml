import json
import logging
from pathlib import Path
import torch
import torch.nn as nn

from .config import CATEGORIES, CKPT_CONFIG, CKPT_PATH, IM_SIZE

logger = logging.getLogger("i-dtf.model_loader")


def _patch_backbone_compatibility():
    """
    Patch DinoV3Backbone dynamically to support HuggingFace Transformers model attribute variations
    (e.g., vit.model.layer vs vit.encoder.layer vs vit.layer).
    """
    try:
        import src.model.backbone as backbone_mod
        from transformers import AutoModel, AutoConfig

        if getattr(backbone_mod, "_idtf_patched", False):
            return

        orig_init = backbone_mod.DinoV3Backbone.__init__

        def patched_init(
            self,
            name_or_model=backbone_mod.BACKBONE,
            frozen=True,
            unfreeze_blocks=4,
            unfreeze_norm=True,
            drop_path_rate=0.0,
        ):
            nn.Module.__init__(self)
            if isinstance(name_or_model, str):
                if drop_path_rate > 0.0:
                    config = AutoConfig.from_pretrained(name_or_model)
                    if hasattr(config, "drop_path_rate"):
                        config.drop_path_rate = drop_path_rate
                    self.vit = AutoModel.from_pretrained(name_or_model, config=config)
                else:
                    self.vit = AutoModel.from_pretrained(name_or_model)
            else:
                self.vit = name_or_model

            if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
                self.transformer_layers = self.vit.encoder.layer
            elif hasattr(self.vit, "layer"):
                self.transformer_layers = self.vit.layer
            elif hasattr(self.vit, "model") and hasattr(self.vit.model, "layer"):
                self.transformer_layers = self.vit.model.layer
            else:
                raise ValueError("Could not find transformer layers in model")

            self.num_blocks = len(self.transformer_layers)

            if frozen:
                for p in self.vit.parameters():
                    p.requires_grad_(False)
            else:
                self._setup_partial_finetuning(unfreeze_blocks, unfreeze_norm)

        backbone_mod.DinoV3Backbone.__init__ = patched_init
        backbone_mod._idtf_patched = True
    except Exception as e:
        logger.warning(f"Could not apply backbone patch: {e}")


def build_model(
    checkpoint_path: Path = CKPT_PATH,
    config_path: Path = CKPT_CONFIG,
    device: str | torch.device = "cpu",
):
    """
    Build Damage-TriageFormer Model architecture from training config and load weights.

    Args:
        checkpoint_path: Path to best_model.pth checkpoint
        config_path: Path to config.json with model training hyperparameters
        device: Torch device (cpu or cuda)

    Returns:
        tuple: (model: torch.nn.Module, use_gated_head: bool)
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at {checkpoint_path}.\n"
            "Please run 'python download_models.py' from the project root to download pre-trained weights."
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}.\n"
            "Please run 'python download_models.py' from the project root to download pre-trained config."
        )

    # Patch backbone for transformers compatibility before loading Model
    _patch_backbone_compatibility()

    # Import Model dynamically from DTF_REPO
    try:
        from src.model.net import Model as DTFModel
    except ImportError as e:
        raise ImportError(
            f"Could not import Damage-TriageFormer Model from src.model.net: {e}\n"
            "Ensure Damage-TriageFormer submodule is available under Haste-DTF/Damage-TriageFormer."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    logger.info(f"Loading Damage-TriageFormer config from {config_path}")

    model = DTFModel(
        im_size=IM_SIZE,
        num_classes=len(CATEGORIES),
        use_hierarchical=bool(train_cfg.get("use_hierarchical", False)),
        use_gated_head=bool(train_cfg.get("use_gated_head", False)),
        use_aux_severity=bool(train_cfg.get("use_aux_severity", False)),
        use_sfp=bool(train_cfg.get("use_sfp", False)),
        sfp_stride=int(train_cfg.get("sfp_stride", 4)),
        use_multiscale=not train_cfg.get("no_multiscale", False),
        use_attention=not train_cfg.get("no_attention_pool", False),
        drop_path_rate=float(train_cfg.get("drop_path", 0.0)),
    )

    device = torch.device(device)
    logger.info(f"Loading weights from {checkpoint_path} onto {device}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info(f"Checkpoint loaded: {len(missing)} missing keys (expected for auxiliary modules)")
    if unexpected:
        logger.info(f"Checkpoint loaded: {len(unexpected)} unexpected keys")

    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    return model, getattr(model, "use_gated_head", False)
