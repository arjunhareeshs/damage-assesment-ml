# i-dtf: Standalone Image-in / Image-out Damage Classifier

`i-dtf` is a completely self-contained damage classification pipeline built for direct image input and output.
It automatically detects building instances using the model's segmentation head (no external building vector footprints required) and evaluates damage severity using the pre-trained Damage-TriageFormer model.

---

## Key Features & Isolation Guarantees

- **100% Self-Contained:** Everything lives strictly within `i-dtf/`. If this folder is deleted, the surrounding repository is left completely unchanged.
- **Zero Weight Duplication:** Uses existing pre-trained weights in `models/damage_triage/` directly without copying or downloading duplicate files.
- **Single-Pass Efficiency:** Computes backbone feature representations once and shares them between the segmentation decoder and the gated damage head.
- **CLI & REST API:** Offers both command-line (`predict.py`) and Web API (`api.py`) entry points.

---

## Quick Start (CLI)

### Predict Single Image or Folder

```bash
# Single image prediction
python -m i-dtf.predict --input path/to/image.png --output i-dtf/out/result.png

# Directory batch processing with JSON metadata and binary masks
python -m i-dtf.predict --input path/to/images_dir/ --save-json --save-mask
```

### CLI Command Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input, -i` | Path | *Required* | Image file (`.png`, `.jpg`, `.tif`, etc.) or directory |
| `--output, -o` | Path | `i-dtf/out/` | Output classified PNG file or directory |
| `--seg-threshold` | Float | `0.5` | Threshold for building segmentation head |
| `--min-area` | Int | `200` | Minimum building area in pixels |
| `--alpha` | Float | `0.5` | Fill transparency for damage color overlay |
| `--norm` | Choice | `imagenet` | Normalization mode (`imagenet` or `unit`) |
| `--device` | Choice | `auto` | Device to run on (`auto`, `cuda`, `cpu`) |
| `--save-json` | Flag | `False` | Export detection bounding boxes, areas, and probabilities to JSON |
| `--save-mask` | Flag | `False` | Export raw binary segmentation mask PNG |

---

## Web API Service (REST Endpoints)

Launch the FastAPI REST service:

```bash
python -m i-dtf.api
```

The server will start at `http://0.0.0.0:8000`.

### Endpoints

- `GET /health` — Check server status, model readiness, device, and class list.
- `POST /predict` — Upload image file via multipart form. Returns JSON list of building detections and damage probabilities.
- `POST /predict/image` — Upload image file. Returns rendered overlay PNG image file directly.

---

## Damage Classification Palette

Output overlays use the official 5-color damage ramp:

1. **Undamaged** `(255, 255, 255)` — White
2. **Partial Roof Damage** `(252, 190, 165)` — Light Peach / Orange
3. **Total Roof Damage** `(251, 112, 80)` — Orange Red
4. **Partial Structural Collapse** `(211, 32, 32)` — Bright Red
5. **Total Structural Collapse** `(103, 0, 13)` — Dark Maroon
