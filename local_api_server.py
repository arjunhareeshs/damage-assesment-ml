"""Minimal local HTTP bridge between the browser UI (InteractiveLabeler.jsx's
"Classify selected" button) and the local DamageTriageFormer pipeline.

The real production backend (Haste-DTF/haste-fork/api/hastefuncapi) is Azure
Functions + PostGIS + Azurite, queued/async, and needs a full Docker Compose
stack to run locally. That's out of scope for the local-only pipeline this
project uses (download_models.py -> run_local_aoi_assessment.py) — a browser
still can't run PyTorch directly, though, so this server exists purely to
bridge "click Classify in the map UI" to that same local pipeline, synchronously,
without adopting the Azure stack.

Loads the DamageTriageFormer model ONCE at startup (not per-request) via
hastegeo.workflows.damage_triage_aoi_infer.run_damage_triage_aoi's
``preloaded_model`` param, so repeated "Classify" clicks are fast.

No new dependencies: stdlib http.server only (not Flask/FastAPI).

Endpoints:
    GET  /health              -> {"status": "ok", "model_loaded": bool}
    POST /classify            -> body: {"buildings": [{"building_id", "geometry"}, ...],
                                         "imagery": "<optional path override>"}
                                  returns: {"buildings": [{building_id, class,
                                            class_name, confidence, probs}, ...]}

Usage:
    python local_api_server.py [--port 8787] [--imagery path/to/imagery.tif]
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HASTE_FORK = BASE_DIR / "Haste-DTF" / "haste-fork"
DTF_REPO = BASE_DIR / "Haste-DTF" / "Damage-TriageFormer"

sys.path.insert(0, str(HASTE_FORK / "hastelib" / "src"))
sys.path.insert(0, str(DTF_REPO))
os.environ["DTF_REPO_ROOT"] = str(DTF_REPO)

DEFAULT_IMAGERY = BASE_DIR / "data" / "imagery" / "m_3510141_ne_14_060_20200915.tif"
OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "local_api"

# Populated once at startup by load_model(); a (model, use_gated) tuple or None.
_MODEL_STATE = {"preloaded_model": None}
_MODEL_LOCK = threading.Lock()


def get_local_checkpoint():
    damage_triage_dir = BASE_DIR / "models" / "damage_triage"
    checkpoint_path = damage_triage_dir / "best_model.pth"
    config_path = damage_triage_dir / "config.json"
    if not checkpoint_path.exists() or not config_path.exists():
        print(
            "[ERROR] Model checkpoint not found in models/damage_triage/.\n"
            "        Run `python download_models.py` first."
        )
        sys.exit(1)
    return str(checkpoint_path), str(config_path)


def load_model():
    """Load the DamageTriageFormer checkpoint once, at server startup."""
    import torch
    from hastegeo.workflows.damage_triage_infer import build_dtf_model

    checkpoint_path, checkpoint_config_path = get_local_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading DamageTriageFormer checkpoint on {device}...")
    try:
        model, use_gated = build_dtf_model(checkpoint_path, checkpoint_config_path, device)
    except Exception as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            print(
                "\n[ERROR] The DINOv3 backbone isn't available locally yet, and "
                "facebook/dinov3-vitl16-pretrain-lvd1689m is a GATED Hugging Face "
                "repo. Run `python download_models.py` after requesting/getting "
                "access at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m "
                "and running `huggingface-cli login`."
            )
            sys.exit(1)
        raise
    print(f"[+] Model ready ({'gated head' if use_gated else 'severity head'}).")
    _MODEL_STATE["preloaded_model"] = (model, use_gated)


class ClassifyHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Local-dev-only CORS: the Vite UI runs on a different port.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model_loaded": _MODEL_STATE["preloaded_model"] is not None,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send_json(404, {"error": "not found"})
            return

        if _MODEL_STATE["preloaded_model"] is None:
            self._send_json(503, {"error": "model not loaded yet"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": f"invalid JSON body: {e}"})
            return

        buildings = body.get("buildings") or []
        if not buildings:
            self._send_json(400, {"error": "no buildings in request"})
            return

        imagery_path = body.get("imagery") or str(DEFAULT_IMAGERY)
        if not os.path.exists(imagery_path):
            self._send_json(400, {"error": f"imagery not found: {imagery_path}"})
            return

        # One classify request at a time — the model isn't guaranteed
        # thread-safe for concurrent forward passes, and this is a
        # single-analyst local tool, not a multi-user server.
        with _MODEL_LOCK:
            try:
                from hastegeo.workflows.damage_triage_aoi_infer import run_damage_triage_aoi

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                results = run_damage_triage_aoi(
                    image_path=imagery_path,
                    buildings=buildings,
                    checkpoint_path=None,
                    checkpoint_config_path=None,
                    output_dir=str(OUTPUT_DIR),
                    preloaded_model=_MODEL_STATE["preloaded_model"],
                )
            except Exception as e:
                self._send_json(500, {"error": str(e)})
                return

        self._send_json(200, {"buildings": results})

    def log_message(self, fmt, *args):
        print(f"[local_api_server] {self.address_string()} - {fmt % args}")


def main():
    global DEFAULT_IMAGERY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--imagery",
        default=str(DEFAULT_IMAGERY),
        help="Default imagery COG used when a /classify request doesn't override it",
    )
    args = parser.parse_args()

    DEFAULT_IMAGERY = Path(args.imagery)

    load_model()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ClassifyHandler)
    print(f"[+] Listening on http://127.0.0.1:{args.port}  (POST /classify, GET /health)")
    print("    Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
