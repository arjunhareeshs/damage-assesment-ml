import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import torch

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .config import CATEGORIES, CKPT_CONFIG, CKPT_PATH, CLASS_COLORS
from .inference import predict_image
from .model_loader import build_model
from .render import render_overlay

logger = logging.getLogger("i-dtf.api")

STATIC_DIR = Path(__file__).parent / "static"

model = None
use_gated = False
device = "cuda" if torch.cuda.is_available() else "cpu"


def _read_upload_as_rgb(contents: bytes) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    return np.array(pil_img)


def _instances_to_json(instances) -> tuple[dict, list[dict]]:
    cat_counts = {cat: 0 for cat in CATEGORIES}
    instances_data = []
    for inst in instances:
        cat_counts[inst.class_name] += 1
        instances_data.append(
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
    return cat_counts, instances_data


def _png_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


if FASTAPI_AVAILABLE:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global model, use_gated
        logger.info(f"Initializing i-DTF model on device {device}...")
        model, use_gated = build_model(
            checkpoint_path=CKPT_PATH,
            config_path=CKPT_CONFIG,
            device=device,
        )
        logger.info("i-DTF model ready for API requests.")
        yield

    app = FastAPI(
        title="i-DTF Damage Classifier API",
        description="Standalone Image-in / Image-out Damage Classification Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse("<h1>i-DTF</h1><p>UI not found; see /health and /docs.</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy" if model is not None else "loading",
            "model_loaded": model is not None,
            "device": str(device),
            "categories": CATEGORIES,
            "class_colors": CLASS_COLORS,
        }

    @app.post("/predict")
    async def predict_endpoint(
        file: UploadFile = File(...),
        seg_threshold: float = Form(0.5),
        min_area: int = Form(200),
        norm: str = Form("imagenet"),
    ):
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized.")

        rgb_img = _read_upload_as_rgb(await file.read())
        instances, _ = predict_image(
            model, rgb_img, seg_threshold=seg_threshold, min_area=min_area, norm=norm, device=device,
        )
        cat_counts, instances_data = _instances_to_json(instances)

        return JSONResponse(
            content={
                "filename": file.filename,
                "dimensions": [rgb_img.shape[1], rgb_img.shape[0]],
                "total_instances": len(instances),
                "category_counts": cat_counts,
                "instances": instances_data,
            }
        )

    @app.post("/predict/image")
    async def predict_image_overlay_endpoint(
        file: UploadFile = File(...),
        seg_threshold: float = Form(0.5),
        min_area: int = Form(200),
        alpha: float = Form(0.5),
        norm: str = Form("imagenet"),
    ):
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized.")

        rgb_img = _read_upload_as_rgb(await file.read())
        instances, _ = predict_image(
            model, rgb_img, seg_threshold=seg_threshold, min_area=min_area, norm=norm, device=device,
        )
        rendered_pil = render_overlay(rgb_img, instances, alpha=alpha, draw_legend_panel=True)

        buf = io.BytesIO()
        rendered_pil.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.post("/analyze")
    async def analyze_endpoint(
        file: UploadFile = File(...),
        seg_threshold: float = Form(0.5),
        min_area: int = Form(200),
        alpha: float = Form(0.5),
        norm: str = Form("imagenet"),
    ):
        """Single round trip for the UI: overlay PNG + segmentation mask PNG
        + full JSON detections, all base64-inlined in one response, so the
        browser doesn't pay for two separate inference passes on the same
        image (see /predict and /predict/image for the split versions used
        by scripts)."""
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized.")

        rgb_img = _read_upload_as_rgb(await file.read())

        t0 = time.time()
        instances, seg_mask = predict_image(
            model, rgb_img, seg_threshold=seg_threshold, min_area=min_area, norm=norm, device=device,
        )
        elapsed_s = time.time() - t0

        rendered_pil = render_overlay(rgb_img, instances, alpha=alpha, draw_legend_panel=True)
        mask_pil = Image.fromarray((seg_mask * 255).astype(np.uint8))

        cat_counts, instances_data = _instances_to_json(instances)

        return JSONResponse(
            content={
                "filename": file.filename,
                "dimensions": [rgb_img.shape[1], rgb_img.shape[0]],
                "elapsed_s": round(elapsed_s, 3),
                "total_instances": len(instances),
                "category_counts": cat_counts,
                "instances": instances_data,
                "image_b64": _png_b64(rendered_pil),
                "mask_b64": _png_b64(mask_pil),
            }
        )


def run_server(host: str = None, port: int = None):
    if not FASTAPI_AVAILABLE:
        print("FastAPI / Uvicorn is not installed. Run 'pip install fastapi uvicorn python-multipart' to enable the API.")
        return
    host = host or os.environ.get("HOST", "127.0.0.1")
    port = port or int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
