import io
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import torch

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, Response
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .config import CATEGORIES, CKPT_CONFIG, CKPT_PATH
from .inference import predict_image
from .model_loader import build_model
from .render import render_overlay

logger = logging.getLogger("i-dtf.api")

model = None
use_gated = False
device = "cuda" if torch.cuda.is_available() else "cpu"

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="i-DTF Damage Classifier API",
        description="Standalone Image-in / Image-out Damage Classification Service",
        version="1.0.0",
    )

    @app.on_event("startup")
    def load_model_on_startup():
        global model, use_gated, device
        logger.info(f"Initializing i-DTF model on device {device}...")
        model, use_gated = build_model(
            checkpoint_path=CKPT_PATH,
            config_path=CKPT_CONFIG,
            device=device,
        )
        logger.info("i-DTF model ready for API requests.")

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device),
            "categories": CATEGORIES,
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

        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            rgb_img = np.array(pil_img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        instances, _ = predict_image(
            model,
            rgb_img,
            seg_threshold=seg_threshold,
            min_area=min_area,
            norm=norm,
            device=device,
        )

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

        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            rgb_img = np.array(pil_img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        instances, _ = predict_image(
            model,
            rgb_img,
            seg_threshold=seg_threshold,
            min_area=min_area,
            norm=norm,
            device=device,
        )

        rendered_pil = render_overlay(rgb_img, instances, alpha=alpha, draw_legend_panel=True)

        buf = io.BytesIO()
        rendered_pil.save(buf, format="PNG")
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    if not FASTAPI_AVAILABLE:
        print("FastAPI / Uvicorn is not installed. Run 'pip install fastapi uvicorn python-multipart' to enable the API.")
        return
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
