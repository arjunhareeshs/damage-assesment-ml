import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CATEGORIES, CLASS_COLORS
from .inference import Instance


def render_overlay(
    rgb_img: np.ndarray,
    instances: list[Instance],
    alpha: float = 0.5,
    draw_legend_panel: bool = True,
) -> Image.Image:
    """
    Render damage classification overlay and legend onto an RGB image.

    Args:
        rgb_img: Base image array (H, W, 3) uint8 RGB
        instances: List of predicted Instance objects
        alpha: Transparency for class color fill (0.0 to 1.0)
        draw_legend_panel: Whether to render legend panel in bottom-left

    Returns:
        PIL Image in RGB format
    """
    img_h, img_w = rgb_img.shape[:2]

    # Create overlay canvas
    overlay = rgb_img.copy()

    # Kernel for boundary extraction (2px border outline)
    kernel = np.ones((3, 3), np.uint8)

    for inst in instances:
        color = CLASS_COLORS[inst.class_id]  # RGB tuple
        mask = inst.mask.astype(np.uint8)

        # Fill building interior with class color
        overlay[inst.mask] = color

        # Compute 2px boundary (mask minus eroded mask)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = (mask > 0) & (eroded == 0)

        # Draw opaque boundary outline
        overlay[boundary] = color

    # Blend filled interiors with original image
    blended = cv2.addWeighted(overlay, alpha, rgb_img, 1.0 - alpha, 0)

    # Redraw crisp opaque outlines over blended image
    for inst in instances:
        color = CLASS_COLORS[inst.class_id]
        mask = inst.mask.astype(np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = (mask > 0) & (eroded == 0)
        blended[boundary] = color

    # Convert to PIL Image for high-quality text and legend drawing
    pil_img = Image.fromarray(blended)

    if draw_legend_panel:
        draw_legend(pil_img, instances=instances)

    return pil_img


def draw_legend(pil_img: Image.Image, instances: list[Instance] | None = None) -> None:
    """
    Draw a translucent legend panel in the bottom-left corner of the PIL Image.
    """
    w, h = pil_img.size

    # Count detections per category if instances provided
    counts = {i: 0 for i in range(len(CATEGORIES))}
    if instances:
        for inst in instances:
            counts[inst.class_id] += 1

    # Panel styling parameters
    padding = 10
    swatch_size = 18
    row_height = 24
    header_height = 24

    panel_width = 300
    panel_height = header_height + len(CATEGORIES) * row_height + padding * 2

    # Draw position (bottom-left)
    x0 = padding
    y0 = max(0, h - panel_height - padding)

    # Create semi-transparent overlay layer for panel
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Draw panel background (dark translucent)
    draw_overlay.rectangle(
        [x0, y0, x0 + panel_width, y0 + panel_height],
        fill=(20, 24, 30, 200),
        outline=(255, 255, 255, 100),
        width=1,
    )

    # Convert main image to RGBA to composite panel
    if pil_img.mode != "RGBA":
        pil_img_rgba = pil_img.convert("RGBA")
    else:
        pil_img_rgba = pil_img

    composite = Image.alpha_composite(pil_img_rgba, overlay)
    draw = ImageDraw.Draw(composite)

    # Try loading default font or bitmap fallback
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Title
    draw.text((x0 + padding, y0 + padding), "i-DTF Damage Classification", fill=(255, 255, 255, 255), font=font)

    # Legend rows
    current_y = y0 + padding + header_height

    for cat_id, cat_name in enumerate(CATEGORIES):
        color_rgb = CLASS_COLORS[cat_id]
        count = counts[cat_id]

        # Swatch rectangle
        swatch_x0 = x0 + padding
        swatch_y0 = current_y + 2
        swatch_x1 = swatch_x0 + swatch_size
        swatch_y1 = swatch_y0 + swatch_size

        draw.rectangle([swatch_x0, swatch_y0, swatch_x1, swatch_y1], fill=color_rgb + (255,), outline=(0, 0, 0, 255))

        # Text label
        label_text = f"{cat_name} ({count})" if instances is not None else cat_name
        draw.text((swatch_x1 + 10, current_y + 2), label_text, fill=(240, 240, 240, 255), font=font)

        current_y += row_height

    # Paste composite back to pil_img
    pil_img.paste(composite.convert("RGB"))
