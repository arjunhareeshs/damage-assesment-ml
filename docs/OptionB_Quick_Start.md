# Option B — HASTE + DamageTriageFormer Quick Start (Docker already installed)

**Assumes:** Docker + Compose installed. Goal: HASTE's map UI, used for annotation
and display, powered by DamageTriageFormer's 5-class typology model for the actual
prediction.

**Reality:** This is a fork-and-integrate job. You're adding a new inference path to
HASTE — including the round trip of taking what the analyst draws/selects on the
map, sending it out to DamageTriageFormer, and bringing the result back to color
the same map. Start with Option A working first.

**The full loop this file implements:**
```
Analyst annotates on HASTE map
        │
        ▼
Frontend detects/extracts the annotation (polygons or selected buildings)
        │
        ▼
Annotation sent to backend API (new endpoint)
        │
        ▼
Backend converts annotation → tile + instance masks
        │
        ▼
DamageTriageFormer inference (5-class per building)
        │
        ▼
Backend writes result in HASTE's prediction-layer format
        │
        ▼
Frontend fetches the prediction layer
        │
        ▼
Map re-renders buildings colored by the 5 typology classes
```

---

## 0. Prerequisite: Option A must be working

```bash
cd haste
docker compose -f docker/docker-compose.yml up
# Verify UI loads at localhost:5173 with no errors
```

Once confirmed, proceed.

---

## 1. Get DamageTriageFormer code + weights

```bash
# Clone the model code (separate from HASTE)
git clone https://github.com/YimingXiao98/Damage-TriageFormer.git
cd Damage-TriageFormer

# Install dependencies
pip install -r requirements.txt

# Download weights from Hugging Face (code + weights are separate sources)
python << 'EOF'
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Ymx1025/DamageTriageFormer-model",
    filename="best_model.pth",
    local_dir="./checkpoints"
)
config_path = hf_hub_download(
    repo_id="Ymx1025/DamageTriageFormer-model",
    filename="config.json",
    local_dir="./checkpoints"
)

print(f"Model saved to: {model_path}")
print(f"Config saved to: {config_path}")
EOF
```

Verify:
```bash
ls -lh checkpoints/
# best_model.pth (~500 MB), config.json (~1 KB)
```

---

## 2. Test DamageTriageFormer standalone

Prove it works alone before integrating:
```bash
python -m inference \
  --model-path ./checkpoints/best_model.pth \
  --config ./checkpoints/config.json \
  --tile test_tile.tif \
  --masks test_masks.tif \
  --output predictions.json
```

Expected `predictions.json`:
```json
{
  "buildings": [
    { "building_id": 0, "class": 0, "class_name": "Undamaged", "confidence": 0.92 },
    { "building_id": 1, "class": 2, "class_name": "Total Roof Damage", "confidence": 0.74 }
  ]
}
```

**If this doesn't work, stop here.** Fix the model before touching HASTE.

---

## 3. Capture the annotation from the HASTE map (frontend)

This is the "detection of the annotated view" step. When the analyst draws polygons
or clicks buildings in HASTE's Labeling Tool, that annotation currently lives only in
the React component's local state until it's saved. You need to hook into that save
event and grab the annotation geometry.

**Where this lives:** HASTE's labeling components under `web/src/features/labeling/`
(exact path may differ by version — look for the component that handles
`onSave` / `onSubmitLabels` for the Segmentation or Embedding tool).

**File:** `web/src/features/labeling/hooks/useDamageTriageAnnotation.ts`

```typescript
import { useCallback } from 'react';
import type { FeatureCollection, Polygon } from 'geojson';

interface AnnotatedBuilding {
  building_id: string;
  geometry: Polygon;
  label?: string;       // optional — only needed if you still want a human label
}

/**
 * Extracts the current annotation state from the labeling tool
 * and formats it for the DamageTriageFormer pipeline.
 */
export function useDamageTriageAnnotation(layerId: string) {

  const captureAnnotation = useCallback((
    drawnFeatures: FeatureCollection,
    selectedBuildingIds: string[]
  ) => {
    // Case 1: analyst drew polygons (Method 1 style) — use the drawn geometry
    // directly as the area of interest to run inference on.
    const drawnPolygons = drawnFeatures.features
      .filter(f => f.geometry.type === 'Polygon')
      .map(f => f.geometry as Polygon);

    // Case 2: analyst selected/clicked specific buildings (Method 2 style) —
    // just pass the building IDs; the backend already has their footprints.
    const buildingIds = selectedBuildingIds;

    return {
      layer_id: layerId,
      drawn_polygons: drawnPolygons,   // may be empty
      building_ids: buildingIds,       // may be empty
      captured_at: new Date().toISOString(),
    };
  }, [layerId]);

  const sendAnnotation = useCallback(async (annotationPayload: object) => {
    const res = await fetch('/api/infer/damage-triage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(annotationPayload),
    });

    if (!res.ok) {
      throw new Error(`Annotation submit failed: ${res.status}`);
    }

    return res.json(); // { job_id, status: "queued" }
  }, []);

  return { captureAnnotation, sendAnnotation };
}
```

**Wire it into the existing "Save" / "Run" button** in the labeling tool — instead of
(or in addition to) HASTE's own train call, call `captureAnnotation()` then
`sendAnnotation()`.

```typescript
// Inside the labeling tool component, on the existing save/submit handler:
const { captureAnnotation, sendAnnotation } = useDamageTriageAnnotation(layerId);

async function handleRunDamageTriage() {
  const payload = captureAnnotation(drawnFeatures, selectedBuildingIds);
  const { job_id } = await sendAnnotation(payload);
  pollJobStatus(job_id); // see Section 8
}
```

This is the "detect the annotation and send it" half of your requirement.

---

## 4. Fork HASTE and add the integration point

```bash
# Fork microsoft/haste on GitHub, then:
git clone https://github.com/YOUR_USERNAME/haste.git haste-fork
cd haste-fork
```

You're adding one new **inference job processor** to `hastegeo`, one new **API
route**, and one new **frontend fetch-and-render** step (Section 9).

---

## 5. Add the DamageTriageFormer inference processor

**File:** `hastegeo/processors/damage_triage_infer.py`

```python
"""DamageTriageFormer inference processor for HASTE."""

import json
import numpy as np
import torch
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

import sys
sys.path.insert(0, '/path/to/Damage-TriageFormer')
from models import load_model  # adjust to whatever the model code exports


class DamageTriageInferenceProcessor:
    """Turns an annotation payload into a tile + instance masks, runs the model."""

    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path, config_path)
        self.model.to(self.device)
        self.model.eval()

    def build_instance_masks(self, footprints: list, tile_transform, tile_shape) -> np.ndarray:
        """
        Rasterize building footprints into an instance-ID mask matching the tile.
        `footprints` = list of {building_id, geometry} from storage (already
        clipped to this tile), keyed by whatever building_ids came in the
        annotation payload (or ALL footprints in the tile if none were specified).
        """
        shapes = [
            (shape(f["geometry"]), int(f["building_id"]))
            for f in footprints
        ]
        masks = rasterize(
            shapes,
            out_shape=tile_shape,
            transform=tile_transform,
            fill=0,
            dtype='int32'
        )
        return masks

    def process(self, tile_path: str, footprints: list) -> dict:
        with rasterio.open(tile_path) as src:
            tile = src.read([1, 2, 3])
            tile = np.transpose(tile, (1, 2, 0))
            tile_t = torch.from_numpy(tile).float().to(self.device) / 255.0
            transform = src.transform
            shape_hw = (src.height, src.width)

        masks = self.build_instance_masks(footprints, transform, shape_hw)

        with torch.no_grad():
            predictions = self.model(tile_t.unsqueeze(0), masks)

        class_names = [
            "Undamaged", "Partial Roof Damage", "Total Roof Damage",
            "Partial Structural Damage", "Total Structural Collapse"
        ]

        results = []
        for building_id in np.unique(masks):
            if building_id == 0:
                continue
            pred = predictions[building_id]
            class_id = int(torch.argmax(pred).item())
            results.append({
                "building_id": int(building_id),
                "class": class_id,
                "class_name": class_names[class_id],
                "confidence": float(torch.max(pred).item()),
                "probs": [float(p) for p in pred.cpu().numpy()]
            })

        return {"buildings": results}
```

**Key change from before:** the processor now takes the actual annotation
(`footprints` — either the drawn-polygon area or the specifically selected
building IDs from Section 3) rather than assuming the whole layer.

---

## 6. Register the processor as a job type

**File:** `hastegeo/jobs/damage_triage_job.py`

```python
"""Job definition for DamageTriageFormer inference, driven by a captured annotation."""

from hastegeo.jobs import Job
from hastegeo.processors.damage_triage_infer import DamageTriageInferenceProcessor
from hastegeo.storage import fetch_layer_tile, fetch_footprints, write_prediction_layer


class DamageTriageJob(Job):
    job_type = "damage_triage_infer"

    def __init__(self, layer_id: str, drawn_polygons: list, building_ids: list,
                 model_path: str, config_path: str):
        super().__init__()
        self.layer_id = layer_id
        self.drawn_polygons = drawn_polygons
        self.building_ids = building_ids
        self.processor = DamageTriageInferenceProcessor(model_path, config_path)

    def run(self):
        # 1. Fetch the post-event tile for this layer
        tile_path = fetch_layer_tile(self.layer_id)

        # 2. Resolve which footprints to run inference on:
        #    - if building_ids were sent, use exactly those
        #    - else if drawn_polygons were sent, use footprints intersecting them
        #    - else (no annotation given), use all footprints in the tile
        footprints = fetch_footprints(
            layer_id=self.layer_id,
            building_ids=self.building_ids or None,
            intersect_polygons=self.drawn_polygons or None,
        )

        # 3. Run inference
        results = self.processor.process(tile_path, footprints)

        # 4. Convert to HASTE's prediction-layer format (see Section 11 —
        #    this MUST match what HASTE's own model writes, or the map won't render it)
        predictions = self.results_to_prediction_layer(results)

        # 5. Write back to storage — this is what the frontend will fetch in Section 9
        write_prediction_layer(self.layer_id, predictions)

        self.mark_complete()

    def results_to_prediction_layer(self, results: dict) -> dict:
        """Match HASTE's native prediction-layer schema. Capture a real example
        from Option A (Section 11 of this doc) and mirror its structure exactly."""
        return {
            "layer_id": self.layer_id,
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "building_id": b["building_id"],
                        "predicted_class": b["class"],
                        "predicted_class_name": b["class_name"],
                        "confidence": b["confidence"],
                    },
                    # geometry gets filled in from the footprint store when written
                }
                for b in results["buildings"]
            ]
        }
```

---

## 7. Wire the API endpoint (receives the annotation from Section 3)

**File:** `api/functions/infer_damage_triage.py`

```python
"""HTTP trigger: receives the captured annotation from the map, queues inference."""

import json
import azure.functions as func
from hastegeo.jobs import DamageTriageJob
from hastegeo.queue import JobQueue


async def main(req: func.HttpRequest) -> func.HttpResponse:
    """POST /api/infer/damage-triage
    Body (from Section 3's frontend hook):
    {
        "layer_id": "...",
        "drawn_polygons": [ ... ],   # may be empty
        "building_ids": [ ... ],     # may be empty
        "captured_at": "..."
    }
    """
    try:
        body = req.get_json()
        layer_id = body.get("layer_id")
        drawn_polygons = body.get("drawn_polygons", [])
        building_ids = body.get("building_ids", [])

        if not layer_id:
            return func.HttpResponse(
                json.dumps({"error": "layer_id is required"}), status_code=400
            )

        job = DamageTriageJob(
            layer_id=layer_id,
            drawn_polygons=drawn_polygons,
            building_ids=building_ids,
            model_path="./checkpoints/best_model.pth",
            config_path="./checkpoints/config.json"
        )

        queue = JobQueue()
        queue.enqueue(job)

        return func.HttpResponse(
            json.dumps({"job_id": job.id, "status": "queued"}),
            status_code=202
        )

    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=400)
```

**Add a status/result endpoint** so the frontend can poll (Section 8) and then fetch
the finished prediction layer (Section 9):

**File:** `api/functions/get_job_status.py`

```python
"""GET /api/jobs/{job_id} — poll job status; when complete, includes the layer_id
whose prediction layer is now ready to fetch."""

import json
import azure.functions as func
from hastegeo.jobs import get_job

async def main(req: func.HttpRequest) -> func.HttpResponse:
    job_id = req.route_params.get('job_id')
    job = get_job(job_id)

    if not job:
        return func.HttpResponse(json.dumps({"error": "not found"}), status_code=404)

    return func.HttpResponse(json.dumps({
        "job_id": job.id,
        "status": job.status,          # "queued" | "running" | "complete" | "failed"
        "layer_id": job.layer_id,
    }))
```

---

## 8. Poll for completion (frontend)

**File:** `web/src/features/labeling/hooks/useDamageTriageAnnotation.ts` (append)

```typescript
export async function pollJobStatus(jobId: string, onComplete: (layerId: string) => void) {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const data = await res.json();

    if (data.status === 'complete') {
      clearInterval(interval);
      onComplete(data.layer_id);
    } else if (data.status === 'failed') {
      clearInterval(interval);
      throw new Error(`DamageTriageFormer job ${jobId} failed`);
    }
    // else keep polling — still "queued" or "running"
  }, 2000); // poll every 2s
}
```

---

## 9. Return the result to HASTE and render it on the map

This is the "same output returns to HASTE and shows on the map" half of your
requirement. Once the job is `complete`, the prediction layer is sitting in
storage in HASTE's native format (Section 6, step 4). The Visualizer component
just needs to fetch and render it — this reuses HASTE's **existing** prediction
rendering code, since the data now looks exactly like what HASTE's own model
would have produced.

**File:** `web/src/features/visualizer/hooks/useLayerPredictions.ts`

```typescript
import { useCallback } from 'react';
import { useMapStore } from '../store/mapStore'; // HASTE's existing map state store

export function useLayerPredictions() {
  const setPredictionLayer = useMapStore(s => s.setPredictionLayer);

  const fetchAndRenderPredictions = useCallback(async (layerId: string) => {
    const res = await fetch(`/api/layers/${layerId}/predictions`);
    const predictionGeoJSON = await res.json();

    // This is the SAME store action HASTE's own model results use —
    // that's what makes it "show on the map itself" with no new map code.
    setPredictionLayer(layerId, predictionGeoJSON);
  }, [setPredictionLayer]);

  return { fetchAndRenderPredictions };
}
```

**Tie it together in the labeling tool component:**

```typescript
async function handleRunDamageTriage() {
  const payload = captureAnnotation(drawnFeatures, selectedBuildingIds);
  const { job_id } = await sendAnnotation(payload);

  pollJobStatus(job_id, async (layerId) => {
    await fetchAndRenderPredictions(layerId);
    // Map now shows buildings colored by the 5 DamageTriageFormer classes
  });
}
```

At this point the full loop is closed: annotate → detect → send → infer → write in
HASTE's format → fetch → render, all on the same map view the analyst started on.

---

## 10. Update the UI legend for 5 classes

**File:** `web/src/constants/damage_classes.ts`

```typescript
export const DAMAGE_CLASSES = [
  { id: 0, name: "Undamaged", color: "#2ecc71", severity: 0 },
  { id: 1, name: "Partial Roof Damage", color: "#f39c12", severity: 1 },
  { id: 2, name: "Total Roof Damage", color: "#e67e22", severity: 2 },
  { id: 3, name: "Partial Structural Damage", color: "#e74c3c", severity: 3 },
  { id: 4, name: "Total Structural Collapse", color: "#c0392b", severity: 4 },
];
```

Point the Visualizer's color-ramp lookup at this list instead of HASTE's native
4-class severity scale.

---

## 11. Match HASTE's native prediction-layer format exactly

**Before writing any of Section 6/9 code for real**, capture what HASTE's own model
writes when you run Option A once:

```bash
# After running Option A's Method 1 or 2 and getting a colored map,
# inspect the stored prediction layer directly (path depends on your
# storage backend — Azurite locally):
docker compose exec blob ls /data/predictions/
docker compose exec blob cat /data/predictions/<layer_id>.geojson
```

Copy that structure. Your `results_to_prediction_layer()` in Section 6 must produce
byte-for-byte the same shape (same property names, same geometry nesting), or the
Visualizer's existing rendering code won't recognize it.

---

## 12. Decide: human review or fully automated?

- **12a — Fully automated:** skip Section 3's drawing step; on layer upload,
  auto-send all footprints (`building_ids` empty → job uses every footprint in the
  tile) straight to inference.
- **12b — With review:** keep Section 3 as-is; analyst draws/selects an area first,
  triggering inference only for that area, then reviews before export.

---

## 13. End-to-end test

1. Start your fork:
   ```bash
   docker compose -f docker/docker-compose.yml up
   ```
2. Create a project, upload post-event image + footprints (Option A, Sections 5–6)
3. In the Labeling Tool, draw a polygon or select a few buildings
4. Click your new "Run DamageTriageFormer" action
5. Watch the job go `queued` → `running` → `complete` (Section 8 polling)
6. Map should auto-refresh showing the 5-class colors (Section 9)
7. Export — GeoPackage should contain the 5 typology classes

---

## 14. Debugging checklist

- [ ] DamageTriageFormer runs standalone on a test tile + masks
- [ ] Frontend `captureAnnotation()` produces non-empty `drawn_polygons` or `building_ids`
- [ ] `POST /api/infer/damage-triage` returns `202` with a `job_id`
- [ ] Worker picks up the job, `fetch_footprints()` returns the right subset
- [ ] Instance-mask rasterization matches the footprint geometry (no offset)
- [ ] Model inference completes without OOM/device errors
- [ ] `results_to_prediction_layer()` output matches HASTE's native format exactly (Section 11)
- [ ] `GET /api/jobs/{job_id}` reaches `"complete"`
- [ ] Frontend fetches `/api/layers/{layer_id}/predictions` and calls `setPredictionLayer`
- [ ] Map re-renders with the 5-class legend, not HASTE's 4-class one

---

## 15. Biggest pitfall

**Prediction-layer format mismatch (Section 11).** This is where almost all the risk
in this integration lives — everything upstream (annotation capture, API, job queue,
model inference) is standard engineering; this one step is the one place you're
reverse-engineering HASTE's internals. Do Section 11 first, before writing Section 6
for real.

---

## 16. When you're done

You have the exact loop from your diagram:
- Analyst annotates on HASTE's map (Section 3)
- Annotation is detected and sent out (Section 3 → 7)
- DamageTriageFormer processes it (Section 5)
- Result is written back in HASTE's format and fetched by the frontend (Section 6 → 9)
- Map updates in place, colored by the 5 typology classes (Section 9 → 10)
