import argparse
import json
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from .config import CATEGORIES, CKPT_CONFIG, CKPT_PATH, REPO_ROOT
from .inference import predict_image
from .model_loader import build_model
from .render import render_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("i-dtf.predict")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="i-DTF: Standalone Image-in / Image-out Damage Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input image file or directory containing images."
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output image file path or directory (defaults to i-dtf/out/)."
    )
    parser.add_argument("--checkpoint", default=str(CKPT_PATH), help="Path to best_model.pth checkpoint.")
    parser.add_argument("--config", default=str(CKPT_CONFIG), help="Path to model config.json file.")
    parser.add_argument(
        "--seg-threshold", type=float, default=0.5, help="Building segmentation threshold (0.0 to 1.0)."
    )
    parser.add_argument("--min-area", type=int, default=200, help="Minimum pixel area for building instances.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of damage color fill.")
    parser.add_argument(
        "--norm", choices=["imagenet", "unit"], default="imagenet", help="Image normalization strategy."
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device (auto, cuda, cpu)."
    )
    parser.add_argument("--save-json", action="store_true", help="Save per-instance detection JSON file.")
    parser.add_argument("--save-mask", action="store_true", help="Save binary building segmentation mask PNG.")

    return parser.parse_args()


def get_image_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Supported: {SUPPORTED_EXTS}")
    elif input_path.is_dir():
        files = [p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        files.sort()
        if not files:
            raise ValueError(f"No supported image files found under directory {input_path}")
        return files
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Load Model once
    model, use_gated = build_model(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        device=device,
    )

    # Resolve input files
    input_path = Path(args.input).resolve()
    image_files = get_image_files(input_path)
    logger.info(f"Found {len(image_files)} image(s) to process.")

    # Determine output directory
    if args.output:
        out_target = Path(args.output).resolve()
    else:
        out_target = REPO_ROOT / "i-dtf" / "out"

    is_batch = len(image_files) > 1 or input_path.is_dir()
    if is_batch or out_target.suffix == "":
        out_dir = out_target if out_target.suffix == "" else out_target.parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_target.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    for idx, img_file in enumerate(image_files, 1):
        logger.info(f"[{idx}/{len(image_files)}] Processing: {img_file.name}")
        t0 = time.time()

        # Load image as RGB numpy array
        pil_in = Image.open(img_file).convert("RGB")
        rgb_img = np.array(pil_in)

        # Run inference
        instances, seg_mask = predict_image(
            model,
            rgb_img,
            seg_threshold=args.seg_threshold,
            min_area=args.min_area,
            norm=args.norm,
            device=device,
        )

        t_infer = time.time() - t0

        # Render output image
        rendered_img = render_overlay(rgb_img, instances, alpha=args.alpha, draw_legend_panel=True)

        # Output filenames
        stem = img_file.stem
        if is_batch or out_target.suffix == "":
            out_img_path = out_dir / f"{stem}_classified.png"
            out_json_path = out_dir / f"{stem}.json"
            out_mask_path = out_dir / f"{stem}_mask.png"
        else:
            out_img_path = out_target
            out_json_path = out_target.with_suffix(".json")
            out_mask_path = out_target.parent / f"{stem}_mask.png"

        rendered_img.save(out_img_path)
        logger.info(f"  Saved classified overlay -> {out_img_path}")

        if args.save_mask:
            mask_vis = (seg_mask * 255).astype(np.uint8)
            cv2.imwrite(str(out_mask_path), mask_vis)
            logger.info(f"  Saved segmentation mask -> {out_mask_path}")

        # Summary statistics
        cat_counts = {cat: 0 for cat in CATEGORIES}
        json_instances = []

        for inst in instances:
            cat_counts[inst.class_name] += 1
            if args.save_json:
                json_instances.append(
                    {
                        "id": inst.id,
                        "bbox": inst.bbox,
                        "area": inst.area,
                        "class_id": inst.class_id,
                        "class_name": inst.class_name,
                        "confidence": round(inst.confidence, 4),
                        "probs": [round(p, 4) for p in inst.probs],
                    }
                )

        if args.save_json:
            json_data = {
                "image": img_file.name,
                "dimensions": [rgb_img.shape[1], rgb_img.shape[0]],
                "total_instances": len(instances),
                "category_counts": cat_counts,
                "instances": json_instances,
            }
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"  Saved JSON detection data -> {out_json_path}")

        # Print console summary
        print(f"\n--- Results for {img_file.name} (took {t_infer:.2f}s) ---")
        print(f"Total Buildings Detected: {len(instances)}")
        for cat in CATEGORIES:
            print(f"  {cat:30s}: {cat_counts[cat]}")
        print("-" * 50)

    total_time = time.time() - total_start
    logger.info(f"Batch completed: processed {len(image_files)} image(s) in {total_time:.2f}s.")


if __name__ == "__main__":
    main()
