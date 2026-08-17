from dataclasses import dataclass, field
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import CATEGORIES, IM_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Instance:
    id: int
    bbox: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) in full image coords
    area: int
    class_id: int
    class_name: str
    confidence: float
    probs: list[float]
    mask: np.ndarray = field(repr=False)  # 2D boolean mask over full image


def decode_gated(damage_out) -> np.ndarray:
    """
    Decode GatedDamageHead output (gate_list, leaf_list[, aux_list]) into a (N, 5) probability matrix.
    P(0) = 1 - P(damaged)
    P(i+1) = P(damaged) * P(leaf=i) for i in 0..3
    """
    gate_list, leaf_list = damage_out[0], damage_out[1]
    gate_logits = gate_list[0]  # (N,)
    leaf_logits = leaf_list[0]  # (N, 4)

    n = gate_logits.shape[0]
    if n == 0:
        return np.empty((0, len(CATEGORIES)), dtype=np.float32)

    p_damaged = torch.sigmoid(gate_logits).unsqueeze(-1)  # (N, 1)
    leaf_probs = F.softmax(leaf_logits, dim=-1)  # (N, 4)

    probs = torch.cat([1.0 - p_damaged, p_damaged * leaf_probs], dim=-1)  # (N, 5)
    return probs.detach().cpu().numpy()


def preprocess_tile(tile_rgb: np.ndarray, norm: str = "imagenet") -> torch.Tensor:
    """
    Convert (H, W, 3) uint8 RGB array to (1, 3, H, W) normalized float tensor.
    """
    img = tile_rgb.astype(np.float32)
    if norm == "imagenet":
        img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    elif norm == "unit":
        img = img / 255.0
    else:
        raise ValueError(f"Unknown normalization mode: {norm}. Choose 'imagenet' or 'unit'.")

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def predict_image(
    model: torch.nn.Module,
    rgb_img: np.ndarray,
    *,
    seg_threshold: float = 0.5,
    min_area: int = 200,
    norm: str = "imagenet",
    device: str | torch.device = "cuda",
) -> tuple[list[Instance], np.ndarray]:
    """
    Run damage prediction on a full RGB image array (H, W, 3).

    Args:
        model: Loaded Damage-TriageFormer model
        rgb_img: numpy array of shape (H, W, 3), dtype uint8, RGB format
        seg_threshold: Sigmoid probability threshold for building segmentation
        min_area: Minimum area in pixels to keep a building instance
        norm: 'imagenet' or 'unit'
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (list of Instance objects, full_binary_seg_mask)
    """
    device = torch.device(device)
    img_h, img_w = rgb_img.shape[:2]

    full_seg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    instances: list[Instance] = []
    instance_counter = 0

    # Process image in non-overlapping 1024x1024 tiles
    step = IM_SIZE
    for y_min in range(0, img_h, step):
        y_max = min(y_min + step, img_h)
        tile_h = y_max - y_min

        for x_min in range(0, img_w, step):
            x_max = min(x_min + step, img_w)
            tile_w = x_max - x_min

            tile_rgb = rgb_img[y_min:y_max, x_min:x_max]

            # Pad tile to 1024x1024 if edge tile
            if tile_h < IM_SIZE or tile_w < IM_SIZE:
                padded_tile = np.zeros((IM_SIZE, IM_SIZE, 3), dtype=np.uint8)
                padded_tile[:tile_h, :tile_w] = tile_rgb
                input_tile = padded_tile
            else:
                input_tile = tile_rgb

            x_tensor = preprocess_tile(input_tile, norm=norm).to(device)

            with torch.inference_mode():
                # Single backbone pass for both segmentation and damage head
                f = model.backbone(x_tensor)  # (1, C, H_f, W_f)
                seg_logits = model.seg_head(f, out_hw=x_tensor.shape[-2:])  # (1, 1, 1024, 1024)

                seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
                building_tile = (seg_prob > seg_threshold).astype(np.uint8)

                # Store seg mask into full mask
                full_seg_mask[y_min:y_max, x_min:x_max] = building_tile[:tile_h, :tile_w]

                # Instance extraction via connected components
                num_labels, labels = cv2.connectedComponents(building_tile, connectivity=8)

                valid_indices = []
                local_masks = []
                for label_idx in range(1, num_labels):
                    mask_i = (labels == label_idx)
                    area_i = int(mask_i.sum())
                    if area_i >= min_area:
                        valid_indices.append((label_idx, area_i))
                        local_masks.append(mask_i.astype(np.float32))

                if not local_masks:
                    continue

                # Batch instance masks for damage head (N, 1024, 1024)
                masks_tensor = torch.from_numpy(np.stack(local_masks)).to(device)

                # SFP feature refinement (if present)
                f_dmg = model.sfp(f) if getattr(model, "sfp", None) is not None else f

                # Gated damage head prediction
                damage_out = model.damage_head(f_dmg, [masks_tensor])
                probs_matrix = decode_gated(damage_out)  # (N, 5)

                for k, (label_idx, area) in enumerate(valid_indices):
                    local_mask_bool = (labels == label_idx)
                    # Clip mask to actual tile boundaries (ignoring padding)
                    valid_local_mask = local_mask_bool[:tile_h, :tile_w]
                    if not valid_local_mask.any():
                        continue

                    # Construct full image boolean mask
                    full_mask = np.zeros((img_h, img_w), dtype=bool)
                    full_mask[y_min:y_max, x_min:x_max] = valid_local_mask

                    # Bounding box in full image coords
                    rows, cols = np.where(valid_local_mask)
                    xmin = int(x_min + cols.min())
                    ymin = int(y_min + rows.min())
                    xmax = int(x_min + cols.max() + 1)
                    ymax = int(y_min + rows.max() + 1)

                    probs_k = probs_matrix[k].tolist()
                    class_id = int(probs_matrix[k].argmax())
                    confidence = float(probs_matrix[k][class_id])

                    instance_counter += 1
                    instances.append(
                        Instance(
                            id=instance_counter,
                            bbox=(xmin, ymin, xmax, ymax),
                            area=area,
                            class_id=class_id,
                            class_name=CATEGORIES[class_id],
                            confidence=confidence,
                            probs=probs_k,
                            mask=full_mask,
                        )
                    )

    return instances, full_seg_mask
