# HASTE ↔ DamageTriageFormer — AOI-Scoped Integration Plan

**Status:** Supersedes the "fully automated, whole-layer" build path for interactive
use. Companion to [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) and
[`OptionB_Quick_Start.md`](OptionB_Quick_Start.md). This document is the
implementation plan for the architecture described in the HASTE ↔
DamageTriageFormer Integration Architecture & Workflow Plan (the "plan.pdf"),
translated 1:1 onto the actual file paths, models, queues, and conventions of
this repository's `Haste-DTF/haste-fork` (real Microsoft HASTE source, not a
generic template) and `Haste-DTF/Damage-TriageFormer` (the model code).

---

## 0. Why this document exists — what's already built vs. what plan.pdf asks for

Something already runs end-to-end in this repo (`Haste-DTF/haste-fork`), and it
is **not** the pipeline described in plan.pdf. Read this section first so the
rest of the plan reads as a *change list*, not a green-field build.

### 0.1 What is already implemented (do not re-explain this as "the plan" — it's a different plan)

| Piece | File | Behavior |
|---|---|---|
| "Run DamageTriageFormer" button | `ui/src/Components/CreateEditDTFModal.jsx` | Fires on the **whole image layer**. No AOI, no building selection. |
| Queue endpoint | `api/hastefuncapi/function_app.py` → `PutRunDTFInferenceQueueMessage` | Creates a `Model(modelType="dtf")`, no AOI/annotation payload accepted. |
| Preprocessor/Postprocessor | `hastelib/src/hastegeo/core/processors/damage_triage.py` | `DTFInferencePreprocessor` queues; `DTFInferencePostprocessor` submits an Azure Batch task, polls it, and on completion reads back a manifest + writes `model.gpkgUrl` / `model.predictedDamageLayerUrl`. |
| Inference workflow | `hastelib/src/hastegeo/workflows/damage_triage_infer.py` | Tiles the **entire mosaic** into non-overlapping 1024×1024 windows, rasterizes **one binary instance mask per intersecting footprint per tile**, runs `Model.forward(tile, [masks])` **once per tile** (batched, not per-building crops), keeps the highest-confidence class per building across tiles, writes a predictions GeoPackage + two COG rasters (`*_predictions.tif`, `*_visualizer.tif`) for the map. |
| Model loading | same file, `build_dtf_model()` | Already reverse-engineered the real released checkpoint: `use_gated_head=True, use_sfp=True, sfp_stride=8, use_aux_severity=True`. **This part plan.pdf glosses over ("the model may not directly say `{"damage": "Destroyed", "confidence": 0.96}`") is already solved** — reuse `decode_gated_predictions()` verbatim. |
| Review | none | Buildings are colored on the map from the raster; there is no per-building Accept/Reject affordance. |

This is architecturally the **"12a — fully automated" branch** described at the
end of `OptionB_Quick_Start.md` §12, implemented with a real Azure Batch job
instead of a toy synchronous call. It is efficient (batched tile inference,
one job for potentially millions of buildings) but it cannot do what plan.pdf
asks for: **let an analyst draw an AOI and run inference on just that area**,
with **individual per-building crops**, **individual attribute updates**, and
an **Accept/Reject review loop** on the map.

### 0.2 What plan.pdf specifies that is genuinely new work

1. **AOI as a spatial filter drawn on the map** — not "whole layer," not a
   pre-baked import-time clip.
2. **PostGIS `ST_Intersects(geometry, AOI)`** to resolve which buildings are
   in scope — a live spatial query, not a static cached GeoPackage read once
   at image-layer creation.
3. **Per-building crop generation** — `building_B101.png`, `building_B102.png`,
   … — each building becomes its own inference unit ("Job 1: B101", "Job 2:
   B102", …), not a shared tile+mask-batch call.
4. **Per-building attribute update**, not a full-layer raster repaint —
   `Polygon B101` gains `damage = Destroyed`, `mask = mask_B101.png`; the
   polygon geometry itself never changes.
5. **Human review affordance** — clicking a polygon shows Building ID,
   original crop, segmentation mask, damage class, confidence, **Accept /
   Reject**.
6. **Scale discipline** — Part 00 of plan.pdf is explicit that AOI clipping
   exists so a 10.6M-footprint, 500GB state-wide layer is never touched in
   full; only the drawn AOI (e.g. ~25,000 buildings) is queried, cropped, and
   inferred.

None of items 1–5 exist in the current build. Item 6 is *partially* true today
(the whole-layer job never re-downloads the whole state's imagery — Overture
footprints were already AOI-clipped once, at import time, by
`hastegeo/core/utils/footprints.py::download_building_footprints`) but is not
true in the sense plan.pdf means (re-scoping per analyst session).

### 0.3 Decision this plan encodes

Keep the existing whole-layer `modelType="dtf"` path untouched (it is useful
for "auto-classify everything, review later" scenes). **Add a second,
parallel job type** — `modelType="dtf-aoi"` — that implements plan.pdf's
AOI → PostGIS → per-building-crop → per-building-update → Accept/Reject loop.
Both share the same model loader (`build_dtf_model`, `decode_gated_predictions`)
and the same checkpoint. They differ only in *scoping* (whole layer vs. AOI)
and *unit of work* (tile-batch vs. per-building crop) and *output* (raster
repaint vs. per-building attribute PATCH + review flag).

---

## PART 000 — End-to-end system overview (mapped onto this repo)

```
┌────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES                                                            │
│   Satellite Imagery (GeoTIFF/COG)      Building Footprints (GeoJSON)    │
└───────────────────────────┬──────────────────────────────────────────┬─┘
                             │                                          │
                             ▼                                          ▼
              hastegeo/workflows/prepare_imagery.py     hastegeo/core/utils/footprints.py
              - validate CRS / reproject                - download_building_footprints() /
              - build COG                                 clip_and_normalize_user_footprints()
                             │                                          │
                             └──────────────┬───────────────────────────┘
                                             ▼
                             NEW: hastegeo/core/utils/postgis_load.py
                             - load the cached footprints GPKG into the
                               PostGIS `hg_buildings` table (once per
                               ImageLayer, keyed by imageLayerId)
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│ HASTE PROJECT  (ui/src/Components/Project.jsx, CreateEditProjectModal)  │
│   Import Imagery + Import Building Footprints (existing PutLayer flow)  │
└───────────────────────────┬──────────────────────────────────────────┬─┘
                             ▼
              ui/src/Components/Visualizer/Visualizer.jsx
              (GIS workspace: zoom/pan/measure — existing, unchanged)
                             │
                             ▼
              NEW: ui/src/Components/LabelingTool/AoiDrawTool.jsx
              (draw AOI rectangle/polygon — reuses DrawingToolbar.jsx's
               Azure Maps drawing manager, new "AOI" draw mode)
                             │
                             ▼
              NEW: PUT /api/RunDTFAoiInference   (api/hastefuncapi/function_app.py)
                             │
                             ▼
              NEW: hastegeo/core/processors/damage_triage_aoi.py
              - DTFAoiInferencePreprocessor: validates AOI, enqueues
                             │
                             ▼
              NEW: hastegeo/core/utils/postgis_query.py
              - ST_Intersects(geometry, AOI) → building_ids in scope
                             │
                             ▼
              NEW: hastegeo/core/utils/crop_service.py
              - per-building RGB crop from the COG using each footprint's
                bbox (+ padding), NOT the polygon sent to the model
                             │
                             ▼
              NEW: hastegeo/workflows/damage_triage_aoi_infer.py
              - one DamageTriageFormer forward pass per building crop
              - reuses build_dtf_model() / decode_gated_predictions()
                from workflows/damage_triage_infer.py
                             │
                             ▼
              NEW: hastegeo/core/processors/damage_triage_aoi.py
              - DTFAoiInferencePostprocessor: writes per-building rows back
                to PostGIS (damage_class, confidence, mask_url, status="pending_review")
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│ HASTE Viewer (ui/src/Components/Visualizer + NEW BuildingReviewPanel)  │
│   Overlay Results + Accept/Reject                                       │
└───────────────────────────┬──────────────────────────────────────────┘
                             ▼
              Existing export path: GetModelArtifact?kind=gpkg /
              NEW: GetModelArtifact?kind=aoi-gpkg (per-AOI export)
```

---

## MODULE 1 — Data collection (unchanged, already implemented)

This happens before HASTE, exactly as plan.pdf describes, and this repo
already has the code for it:

- **Satellite imagery** → `hastegeo/workflows/prepare_imagery.py` ingests a
  GeoTIFF and produces a Cloud-Optimized GeoTIFF (`postEventMosaicCogImageryUrl`
  on `ImageLayer`).
- **Building footprints** → `hastegeo/core/utils/footprints.py` either pulls
  Overture Maps footprints for the imagery's AOI bbox
  (`download_building_footprints`) or clips a user-supplied file
  (`clip_and_normalize_user_footprints`). Both write a GeoPackage with columns
  `(id, geometry, subtype, class)` to `ImageLayer.buildingFootprintsUrl`.

**No changes needed in Module 1.** The plan.pdf's warning — "these polygons
are NOT predictions, the model only predicts damage" — already matches this
repo's data model: `Model.gpkgUrl` (predictions) is always a separate object
from `ImageLayer.buildingFootprintsUrl` (footprints).

---

## MODULE 2 — Data preparation (unchanged, already implemented)

CRS/projection/resolution validation happens in `prepare_imagery.py` and
`footprints.py` (both reproject to EPSG:4326 / the raster's native CRS as
needed — see `clip_and_normalize_user_footprints`'s explicit reprojection
block). **No changes needed.**

**New addition required by this plan:** after footprints are cached to the
GeoPackage, load them into PostGIS so Module 6 can run `ST_Intersects`
queries against them instead of re-reading the GeoPackage from blob storage
on every AOI draw.

### New file: `hastelib/src/hastegeo/core/utils/postgis_load.py`

```python
def load_footprints_into_postgis(image_layer: ImageLayer, gpkg_local_path: str) -> int:
    """Load ImageLayer.buildingFootprintsUrl into hg_buildings, keyed by
    imageLayerId. Idempotent: DELETE ... WHERE image_layer_id = %s before
    INSERT, so re-running imagery prep doesn't duplicate rows."""
```

Called once, at the end of `prepare_imagery.py`'s existing footprint-caching
step (after `download_building_footprints` / `clip_and_normalize_user_footprints`
returns), not on every AOI draw. This is the plan.pdf "Build Spatial Index
(PostGIS)" box in the master diagram.

### New PostGIS table

```sql
CREATE TABLE IF NOT EXISTS hg_buildings (
    building_id     TEXT        NOT NULL,   -- from footprints.id (string, see footprints.py note)
    image_layer_id  TEXT        NOT NULL,
    project_id      TEXT        NOT NULL,
    geometry        GEOMETRY(Geometry, 4326) NOT NULL,
    subtype         TEXT,
    class           TEXT,
    damage_class    INTEGER,               -- 0..4, NULL until inferred
    damage_class_name TEXT,
    confidence      DOUBLE PRECISION,
    mask_url        TEXT,                  -- blob path to mask_<building_id>.png
    crop_url        TEXT,                  -- blob path to building_<building_id>.png
    review_status   TEXT        DEFAULT 'unclassified',  -- unclassified|pending_review|accepted|rejected
    last_aoi_model_id TEXT,                 -- which dtf-aoi Model run last touched this row
    updated_at      TIMESTAMPTZ,
    PRIMARY KEY (image_layer_id, building_id)
);
CREATE INDEX IF NOT EXISTS hg_buildings_geom_gix ON hg_buildings USING GIST (geometry);
```

`AzurePostgreSQLDataLayer` (`hastegeo/core/data_layer/azure_postgresql_data_layer.py`)
already exists and already knows how to authenticate against Azure Database
for PostgreSQL with `DefaultAzureCredential` and a per-partition JSONB table —
but its schema (`identifier, data_type, partition_key, data JSONB, bounds
GEOMETRY`) is a generic KV store, not a per-building row table with real
columns. `hg_buildings` is a **new, dedicated table** (not routed through
`AzurePostgreSQLDataLayer`), accessed through a thin new client described in
Module 6.

---

## MODULE 3 — Create HASTE project (unchanged, already implemented)

`ui/src/Components/CreateEditProjectModal.jsx` → `PutProject` →
`ui/src/Components/CreateEditImageLayerForm.jsx` → `PutLayer`. Imagery +
footprints import. **No changes.**

---

## MODULE 4 — GIS workspace (unchanged, already implemented)

`ui/src/Components/Visualizer/Visualizer.jsx` (Azure Maps: zoom/pan/measure)
and `ui/src/Components/LabelingTool/LabelingTool.jsx` (draw/edit/inspect).
**No changes** — Module 5/6 add a new drawing mode and a new right-panel
action on top of this, they don't replace it.

---

## MODULE 5 — Draw AOI

Plan.pdf: *"AOI is only a Spatial Filter — not a building, not a crop, not
an image."* This repo already has AOI-polygon extraction from a raster
(`hastegeo/core/utils/aoi.py::extract_aoi_polygon`) but that is a
*server-side, from-the-COG's-valid-pixels* AOI (used once at import time to
clip Overture downloads) — **not** an analyst-drawn, per-session AOI on the
already-loaded map. Plan.pdf's AOI is the latter. New frontend work:

### New file: `ui/src/Components/LabelingTool/AoiDrawTool.jsx`

- Reuses the same Azure Maps `drawing` module already wired up in
  `DrawingToolbar.jsx` (`atlas.drawing.DrawingManager`), adding one more
  `drawingMode: "draw-rectangle"` / `"draw-polygon"` entry alongside the
  existing label-polygon modes, tagged with a distinct `properties.role =
  "aoi"` so `LabelingToolHelper.js` can tell an AOI shape apart from a damage
  label shape on `onDrawingComplete`.
- On completion, stores the AOI polygon (EPSG:4326 GeoJSON) in local React
  state (`selectedAoi`) — mirrors exactly how `drawnFeatures` already flows
  through `LabelingTool.jsx` for Method 1 labeling, just a new named slot
  instead of a new mechanism.
- Does **not** POST anything by itself. It only produces a GeoJSON polygon
  that the new "Run DamageTriageFormer on AOI" button (Module 6) reads.

---

## MODULE 6 — Annotation, AOI selection & sending buildings to DamageTriageFormer

This is the load-bearing module. Plan.pdf's shape:

```
AOI  →  Query buildings inside AOI  →  Send to DamageTriageFormer
```

replacing (for this new path only) the existing "Predict all buildings"
whole-layer flow. Concretely:

### 6.1 Frontend: capture + send

**New file:** `ui/src/Components/LabelingTool/RunDtfAoiButton.jsx` (a sibling
of the existing `CreateEditDTFModal.jsx`, reusing its `SectionModal` /
`apiPut` pattern):

```jsx
async function submit() {
  const body = {
    projectId,
    imageLayerId: imageLayer.imageLayerId,
    modelType: "dtf-aoi",
    name: state.name,
    aoi: selectedAoi,          // GeoJSON Polygon, EPSG:4326, from AoiDrawTool
    buildingIds: selectedBuildingIds,  // optional: explicit click-selection instead of a drawn box
    userId: appParams.userId,
  };
  await apiPut("PutRunDTFAoiInferenceQueueMessage", body);
}
```

`buildingIds` covers plan.pdf's "Method 2 style" (analyst clicks specific
buildings instead of drawing a box) using the same `CustomSelection.jsx`
click-select mechanism the existing Labeling Tool already has for choosing
buildings — no new selection UI needed, just a new consumer of it.

### 6.2 Backend: new endpoint

**New route in `api/hastefuncapi/function_app.py`:**
`PutRunDTFAoiInferenceQueueMessage` (PUT). Mirrors `PutRunDTFInferenceQueueMessage`
(§0.1 table) almost exactly — same `Model` creation, same
`StatsPreProcessor` bump — with two differences:

```python
output = Model(**req_body)
output.modelType = "dtf-aoi"
output.dtfAoi = req_body.get("aoi")            # NEW Model field, GeoJSON Polygon
output.dtfBuildingIds = req_body.get("buildingIds", [])  # NEW Model field
if not output.dtfAoi and not output.dtfBuildingIds:
    return func.HttpResponse(
        "Either aoi or buildingIds is required for dtf-aoi.", status_code=400
    )
output = await asyncio.to_thread(DTFAoiInferencePreprocessor(output).send_to_queue)
```

**New `Model` fields** (`hastelib/src/hastegeo/core/models/projects.py`, next
to the existing `dtfJob: Optional[TrainingJob]`):

```python
dtfAoi: Optional[dict] = Field(default=None)          # GeoJSON Polygon, EPSG:4326
dtfBuildingIds: Optional[list[str]] = Field(default=None)
dtfAoiJob: Optional[TrainingJob] = Field(default=None) # separate job handle from dtfJob
```

### 6.3 Backend: resolve which buildings are in scope (the PostGIS step)

**New file:** `hastelib/src/hastegeo/core/utils/postgis_query.py`

```python
def buildings_in_aoi(image_layer_id: str, aoi_geojson: dict = None,
                      building_ids: list[str] = None) -> list[dict]:
    """
    SELECT building_id, ST_AsGeoJSON(geometry) AS geometry
    FROM hg_buildings
    WHERE image_layer_id = %s
      AND ( %(building_ids)s::text[] IS NOT NULL AND building_id = ANY(%(building_ids)s)
            OR ST_Intersects(geometry, ST_SetSRID(ST_GeomFromGeoJSON(%(aoi)s), 4326)) )
    """
```

This is plan.pdf's Part 000 "Query Buildings Inside AOI (PostGIS)" box and
Part 00's "Backend clips ONLY the AOI" box, made literal: `building_ids`
(explicit selection) takes precedence; otherwise `ST_Intersects` against the
drawn polygon. Returns exactly the `B101, B102, B103, …` shortlist from
plan.pdf's worked example — never the whole layer.

### 6.4 Backend: crop generation (per building, not per tile)

**New file:** `hastelib/src/hastegeo/core/utils/crop_service.py`

```python
def crop_building(raster: rasterio.DatasetReader, building_geom_4326,
                   pad_px: int = 32) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Reproject building_geom into raster.crs, take its bounding box, pad by
    pad_px on each side (context margin — plan.pdf's crops are single
    buildings, but a bare bbox with zero margin starves the model of
    surrounding context DamageTriageFormer was trained on 1024x1024 tiles
    with), read that window, and return (crop_array, crop_transform).

    Explicitly mirrors plan.pdf's warning: "It does NOT send Polygon to the
    model. It uses the polygon to extract pixels." The polygon is consumed
    here only to pick the window; from this point on the model sees pixels
    plus ONE binary instance mask, not a vector geometry.
    """
```

For each building in the Module 6.3 shortlist: `building_B101.png`,
`building_B102.png`, `building_B103.png` — written to a per-run scratch
prefix in blob storage (`aoi-runs/<modelId>/crops/`), exactly the file
naming plan.pdf uses. This is the one deliberate architectural fork from the
existing whole-layer workflow (`damage_triage_infer.py`), which crops
**tiles**, not **buildings** — see §0.2 item 3 for why plan.pdf requires the
per-building unit.

### 6.5 Backend: one inference job per building (logical), one model load (practical)

Plan.pdf frames this as "Job 1: B101 / Job 2: B102 / Job 3: B103" — logically
independent units of work, each carrying `{building_id, image}`. Implemented
as a **single Batch task** that loops over the AOI's building list (loading
the DINOv3 checkpoint once, not once per building — reloading a ~500MB ViT-L
checkpoint per building would make even a 25,000-building AOI (§0.2 item 6)
impractical). This keeps faith with plan.pdf's *logical* per-building
contract ("the backend keeps the Building ID. The model never needs to know
it") while being implementable:

**New file:** `hastelib/src/hastegeo/workflows/damage_triage_aoi_infer.py`

```python
from hastegeo.workflows.damage_triage_infer import (
    build_dtf_model, decode_gated_predictions, decode_severity_predictions,
    DTF_CATEGORIES, DTF_IM_SIZE,
)

def run_damage_triage_aoi(image_path, buildings: list[dict],
                           checkpoint_path, checkpoint_config_path) -> list[dict]:
    """
    buildings: [{building_id, geometry (EPSG:4326)}, ...] — the PostGIS
    shortlist from postgis_query.buildings_in_aoi().

    For each building:
      1. crop_service.crop_building() -> (crop_array, crop_transform)
      2. rasterio.features.geometry_mask() of THIS building's own geometry,
         reprojected into crop_transform -> single binary instance mask
         at crop resolution (pad to DTF_IM_SIZE if the crop is smaller,
         matching how the whole-layer workflow zero-pads edge tiles)
      3. model(crop_tensor.unsqueeze(0), [mask_tensor.unsqueeze(0)])
         -> decode_gated_predictions() (reused verbatim from
            damage_triage_infer.py — same checkpoint, same decode contract)
      4. write crop PNG + a per-building mask PNG (visualization only —
         see write_visualizer_rasters's per-class color ramp, reused,
         but rasterized against just this one building's polygon)
      5. append {building_id, class, class_name, confidence, probs,
                 crop_url, mask_url}

    Model is loaded ONCE via build_dtf_model() before the loop — reuse, do
    not reload per building.
    """
```

This function is called from a new, thinner
`DTFAoiInferencePostprocessor._execute_inference()` in
`hastelib/src/hastegeo/core/processors/damage_triage_aoi.py`, which follows
the *exact* Azure Batch submission pattern already proven in
`damage_triage.py::DTFInferencePostprocessor._execute_inference` (§0.1) —
same `UnifiedRunner.add_task`, same config-JSON-in/manifest-JSON-out
contract, same friendly-log polling — just pointed at a new entrypoint
console script `damage-triage-aoi-infer` (registered in the training Docker
image's `pyproject.toml`/`setup.py` next to the existing `damage-triage-infer`
entrypoint) instead of `damage_triage_infer.py`'s `main()`.

**Do not duplicate `build_dtf_model` / `decode_gated_predictions` /
`decode_severity_predictions` / `DTF_CATEGORIES`.** Import them from
`damage_triage_infer.py` — that file is the single source of truth for "how
do you load this exact checkpoint and turn its output into 5 class
probabilities," and it was hard-won (see its own docstring on the gated-head
architecture mismatch trap). The AOI workflow only needs a different *outer
loop* (per-building crop vs. per-tile batch), not a different model contract.

### 6.6 Backend: prediction mapping and per-building attribute update

Plan.pdf's worked example:

```
Building ID = B101
  ↓
Mask = mask_B101.png
  ↓
Damage Class = Destroyed
  ↓
Confidence = 0.96
```
```
Before: ID=B101, Damage=NULL
After:  ID=B101, Damage=Destroyed, Mask=mask_B101.png
```

**New file:** `hastelib/src/hastegeo/core/processors/damage_triage_aoi.py`
(`DTFAoiInferencePostprocessor._update_results_from_job`) — instead of the
whole-layer path's single GeoPackage overwrite + raster repaint, this does a
**row-level UPDATE per building**:

```python
def _update_results_from_job(self):
    results = json.loads(self._manifest_from_task())  # [{building_id, class, ...}, ...]
    postgis_query.update_building_predictions(
        image_layer_id=self.model_data.imageLayerId,
        model_id=self.model_data.modelId,
        rows=results,
        review_status="pending_review",   # never auto-"accepted" — see Module 9/UI Rendering
    )
```

```sql
UPDATE hg_buildings SET
  damage_class = %(class)s, damage_class_name = %(class_name)s,
  confidence = %(confidence)s, mask_url = %(mask_url)s, crop_url = %(crop_url)s,
  review_status = 'pending_review', last_aoi_model_id = %(model_id)s,
  updated_at = now()
WHERE image_layer_id = %(image_layer_id)s AND building_id = %(building_id)s;
```

**The polygon geometry column is never touched by this statement** — matching
plan.pdf's explicit callout ("the polygon never changed, only its attribute
changed"). This is the key behavioral difference from the whole-layer path,
which repaints a raster (§0.1); the AOI path mutates a small number of
attribute rows and leaves everything else (imagery, footprints, the raster
of any earlier whole-layer run) untouched.

Also set `Model.gpkgUrl` to a lightweight **AOI-scoped** export (only the
buildings just touched — not a full layer) so the existing
`GetModelArtifact?kind=gpkg` download path works unchanged for a `dtf-aoi`
model row too.

---

## Module 6.7 — UI rendering + Accept/Reject (plan.pdf's "click the polygon" step)

Plan.pdf: *"If the user clicks the polygon, show Building ID, Original Crop,
Segmentation Mask, Damage Class, Confidence, Accept, Reject."*

### New file: `ui/src/Components/BuildingValidation/DtfAoiReviewPanel.jsx`

Sibling of the existing `ui/src/Components/BuildingValidation/BuildingValidation.jsx`
(which already renders a building-detail side panel on map click — reuse its
click-to-select wiring in `Visualizer.jsx`, don't rebuild it). New content
for a building whose `review_status != 'unclassified'`:

- Building ID (`hg_buildings.building_id`)
- Original crop (`crop_url` — served via the existing blob-SAS pattern
  `GetModelArtifact` already uses for other artifact kinds)
- Segmentation/damage mask (`mask_url`)
- Damage class + confidence (`damage_class_name`, `confidence`)
- **Accept** / **Reject** buttons →

**New endpoint:** `PutBuildingDTFReview` (PUT), body
`{ imageLayerId, buildingId, reviewStatus: "accepted" | "rejected" }`.
Handler in `function_app.py` calls a small `postgis_query.set_review_status()`
UPDATE (same table, `review_status` column only). This directly extends the
existing `PutBuildingValidation` handler's pattern
(`api/hastefuncapi/function_app.py:3667`) rather than inventing a new one —
`PutBuildingValidation` already does "analyst marks a building's AI result
right/wrong," this is the same shape against `hg_buildings` instead of the
embedding sub-model's validation labels.

**Coloring rule for the map layer** (extends `Visualizer.jsx`'s existing
color-ramp lookup, `ui/src/Components/Visualizer/VisualizerImageryControls.jsx`):
`review_status = 'rejected'` buildings render at reduced opacity / hatch
pattern rather than the solid damage-class color, so an analyst can see at a
glance which predictions still need a second look — this is the concrete
form plan.pdf's "reviewer gets both the AI result and the underlying
evidence" takes in this codebase.

---

## PART 00 — AOI clipping & scale (mapped)

Plan.pdf's numeric example (10.6M statewide footprints, AOI narrows to
~25,000) maps onto this repo as:

- **10.6M footprints, 500GB imagery** → already true today for the
  *whole-layer* `dtf` path's inputs (`ImageLayer.buildingFootprintsUrl`,
  `postEventMosaicCogImageryUrl`).
- **AOI clip → ~25,000 buildings** → Module 6.3's `ST_Intersects` query
  against `hg_buildings`, **not** a re-read of the cached GeoPackage (a
  25,000-row GIST-indexed spatial query is fast; loading and re-clipping a
  10.6M-row GeoPackage per AOI draw would not be).
- **AOI imagery → ~2GB** → `crop_service.crop_building()` never reads more
  of the COG than each building's padded bbox window — rasterio's windowed
  read against a COG only pulls the needed blocks over the network/from
  cache, so this is automatic once `crop_building` uses `rasterio.windows`
  (as specified in §6.4), not a manual "clip the COG to the AOI first" step.
- **25,000 buildings → 25,000 crops** → §6.4/§6.5, one crop + one forward
  pass per building, checkpoint loaded once.

No new code beyond what §6.3–§6.5 already specify; this section exists in
plan.pdf to justify *why* Module 6 is scoped the way it is, and the mapping
above documents that the scoping is preserved.

---

## PART 01 — Inference pipeline & prediction mapping (mapped)

Plan.pdf's `HASTE → AOI+Polygon+ID → Crop Extraction Service → Building RGB
Image → DamageTriageFormer → {Segmentation Mask, Scores} → Post-processing
Service → Prediction Mapper → {Building ID + Mask + Damage Class +
Confidence} → PostgreSQL/Object Storage → HASTE` is, end to end:

| Plan.pdf stage | This repo |
|---|---|
| AOI + Building Polygon + Building ID | `postgis_query.buildings_in_aoi()` (§6.3) |
| Crop Extraction Service | `crop_service.crop_building()` (§6.4) |
| DamageTriageFormer | `damage_triage_aoi_infer.py::run_damage_triage_aoi` calling the shared `build_dtf_model()` (§6.5) |
| Post-processing Service (count pixels / dominant class / confidence) | `decode_gated_predictions()` (reused, already does this — softmax over gate×leaf, `argmax`/`max` for class+confidence) |
| Prediction Mapper | `DTFAoiInferencePostprocessor._update_results_from_job` (§6.6) |
| PostgreSQL / Object Storage | `hg_buildings` table (predictions) + blob storage (`crop_url`, `mask_url`) |
| HASTE (polygon, damage attribute, confidence, overlay mask) | `DtfAoiReviewPanel.jsx` (§6.7) |

What DamageTriageFormer "returns" (plan.pdf's `White/Green/Orange/Red`
segmentation-color framing) is a simplification of the real checkpoint's
output — this repo already carries the accurate version
(`GatedDamageHead`: a damaged/undamaged **gate** plus a 4-way **leaf**
classifier, decoded to a 5-way probability vector by
`decode_gated_predictions`, see `damage_triage_infer.py` lines ~209–235).
The AOI workflow must reuse that decode function rather than re-deriving a
color-mask-counting heuristic from plan.pdf's simplified description — the
gated-head reality is already correctly solved in this codebase.

---

## Full folder structure — new and modified files

```
Haste-DTF/haste-fork/
├── hastelib/src/hastegeo/
│   ├── core/
│   │   ├── models/
│   │   │   └── projects.py                       [MODIFY] +dtfAoi, +dtfBuildingIds, +dtfAoiJob on Model
│   │   ├── processors/
│   │   │   ├── damage_triage.py                   [UNCHANGED] whole-layer path, reused for reference
│   │   │   └── damage_triage_aoi.py                [NEW] DTFAoiInferencePreprocessor / Postprocessor
│   │   ├── utils/
│   │   │   ├── aoi.py                              [UNCHANGED] raster-derived AOI (import-time), unrelated
│   │   │   ├── footprints.py                       [UNCHANGED]
│   │   │   ├── postgis_load.py                     [NEW] load footprints GPKG -> hg_buildings
│   │   │   ├── postgis_query.py                    [NEW] ST_Intersects query + per-building UPDATE
│   │   │   └── crop_service.py                     [NEW] per-building windowed COG crop
│   │   └── data_layer/
│   │       └── azure_postgresql_data_layer.py       [UNCHANGED] generic KV+bounds table, not reused for hg_buildings
│   └── workflows/
│       ├── damage_triage_infer.py                   [UNCHANGED, imported from] build_dtf_model, decode_*, DTF_CATEGORIES
│       └── damage_triage_aoi_infer.py                [NEW] per-building crop inference loop
├── api/hastefuncapi/
│   └── function_app.py                               [MODIFY] +PutRunDTFAoiInferenceQueueMessage, +PutBuildingDTFReview
├── db/
│   └── migrations/
│       └── 0001_hg_buildings.sql                     [NEW] CREATE TABLE hg_buildings + GIST index (§Module 2)
├── docker/training/
│   └── (pyproject.toml / entrypoint registration)     [MODIFY] register `damage-triage-aoi-infer` console script
└── ui/src/Components/
    ├── CreateEditDTFModal.jsx                         [UNCHANGED] whole-layer "Run DamageTriageFormer"
    ├── LabelingTool/
    │   ├── AoiDrawTool.jsx                             [NEW] AOI rectangle/polygon draw mode (§Module 5)
    │   ├── RunDtfAoiButton.jsx                         [NEW] captures AOI/selection, calls PutRunDTFAoiInferenceQueueMessage
    │   └── DrawingToolbar.jsx                          [MODIFY] register the new "aoi" drawing mode
    ├── Visualizer/
    │   ├── Visualizer.jsx                              [MODIFY] wire click -> DtfAoiReviewPanel for hg_buildings rows
    │   └── VisualizerImageryControls.jsx                [MODIFY] review_status opacity/hatch rule (§Module 6.7)
    └── BuildingValidation/
        └── DtfAoiReviewPanel.jsx                        [NEW] Building ID / crop / mask / class / confidence / Accept / Reject
```

---

## API contract summary (new endpoints only)

| Route | Method | Body | Response |
|---|---|---|---|
| `PutRunDTFAoiInferenceQueueMessage` | PUT | `{projectId, imageLayerId, name, aoi?, buildingIds?, userId}` | `Model` (dict), `modelType="dtf-aoi"`, `status="Pending"` |
| `GetDTFAoiJobStatus` *(reuses `GetLayerModelsDetails` — no new route needed)* | GET | — | existing `Model` polling already covers `status`/`progressPct`/`statusMessage` |
| `PutBuildingDTFReview` | PUT | `{imageLayerId, buildingId, reviewStatus}` | `{success: true}` |
| `GetBuildingsInAoi` *(optional, for a live building-count preview before submit)* | POST | `{imageLayerId, aoi}` | `{count: N}` — thin wrapper around `postgis_query.buildings_in_aoi`, `SELECT count(*)` variant |

`PutRunDTFAoiInferenceQueueMessage`'s response is polled the same way the
existing `dtf` model already is (`Project.jsx`'s existing model-status
polling loop against `GetLayerModelsDetails`) — no new polling mechanism,
just a new `modelType` value flowing through the same status machine
(`Pending → InProgress → Completed/Failed`, `Config.get_status_types()`).

---

## Decide: human review or fully automated? (plan.pdf §12, mapped)

- **12a — fully automated** is what already exists (`modelType="dtf"`,
  §0.1). Keep it. Do not remove `CreateEditDTFModal.jsx` or
  `damage_triage.py`.
- **12b — with review** is this entire document — `modelType="dtf-aoi"`
  always lands new predictions at `review_status="pending_review"` (§6.6);
  nothing auto-flips to `"accepted"`. The map may still color
  pending-review buildings by their predicted class immediately (so the
  analyst sees the AI result without a blocking step), but export
  (`GetModelArtifact?kind=aoi-gpkg`) should offer a filter to exclude
  `rejected` rows and flag `pending_review` rows in a `reviewed` column, so
  downstream GIS consumers can tell reviewed from unreviewed predictions
  apart.

---

## End-to-end test (plan.pdf §13, mapped)

1. `docker compose -f docker/docker-compose.yml up` (existing dev stack).
2. Create a project, upload post-event imagery + footprints (Module 1–3,
   unchanged).
3. Confirm `hg_buildings` was populated for the new `imageLayerId`
   (`SELECT count(*) FROM hg_buildings WHERE image_layer_id = ...`) —
   proves Module 2's new load step ran.
4. In the Visualizer, draw an AOI box over a small cluster of buildings
   (Module 5).
5. Click "Run DamageTriageFormer on AOI" (`RunDtfAoiButton.jsx`).
6. Confirm `PutRunDTFAoiInferenceQueueMessage` returns 200 with a `modelId`.
7. Watch status go `Pending → InProgress → Completed` via the existing
   polling UI.
8. Confirm exactly the buildings inside the drawn AOI (not the whole layer)
   changed `review_status` from `unclassified` to `pending_review`.
9. Click one of the newly-classified polygons → `DtfAoiReviewPanel` shows
   crop image, mask, class, confidence.
10. Click **Accept** → `review_status` flips to `accepted`, panel/coloring
    updates without a page reload.
11. Export (`GetModelArtifact?kind=aoi-gpkg`) → confirm only the AOI's
    buildings are present, each with the correct 5-class typology and
    `review_status` column.

---

## Debugging checklist (plan.pdf §14, mapped)

- [ ] `hg_buildings` row count for the layer matches the footprints GeoPackage row count (Module 2 load didn't silently drop rows)
- [ ] `ST_Intersects` query returns a strict subset of the layer for a small drawn AOI, not the whole table (Module 6.3)
- [ ] Each `building_<id>.png` crop is non-black / non-empty (Module 6.4 windowed read didn't miss the raster)
- [ ] `damage_triage_aoi_infer.py` loads the checkpoint exactly once per job, not once per building (performance — §6.5)
- [ ] `decode_gated_predictions` output sums to 1.0 per building row (reused correctly, not re-derived)
- [ ] Row-level `UPDATE` in §6.6 leaves `hg_buildings.geometry` byte-identical before/after (plan.pdf's "polygon never changed" invariant)
- [ ] `review_status` starts at `pending_review`, never `accepted`, for a fresh AOI run (§Module 6.7 / §12b)
- [ ] Map coloring distinguishes `pending_review` vs `accepted` vs `rejected` (§Module 6.7)
- [ ] `GetModelArtifact?kind=aoi-gpkg` only contains the AOI's buildings, not the full layer

---

## Biggest pitfalls (mapped from plan.pdf §15 + this repo's own scars)

1. **Don't re-derive the checkpoint-loading / head-decoding logic.**
   `damage_triage_infer.py`'s `build_dtf_model` docstring documents a real,
   already-hit trap: the released checkpoint uses a gated head
   (`use_gated_head=True, use_sfp=True, sfp_stride=8, use_aux_severity=True`),
   and loading it with the plain 5-way `SeverityHead` silently drops most of
   the damage-head weights under `strict=False`. The AOI workflow **must**
   import `build_dtf_model` / `decode_gated_predictions` from that file, not
   copy-paste-and-drift.
2. **Per-building crops need context padding.** A bare bounding-box crop
   with zero margin (plan.pdf's diagrams show tight crops) starves the model
   of the surrounding context it was trained on (1024×1024 tiles). Use the
   `pad_px` margin in `crop_service.crop_building` (§6.4) — tune this
   empirically against a held-out AOI before trusting AOI-path confidence
   scores as comparable to the whole-layer path's tile-batch scores.
3. **PostGIS row-order is not the whole-layer GeoPackage's row-order.**
   The whole-layer path (`damage_triage_infer.py`) has a documented
   row-order invariant (`id = 0..N-1` matching the footprints file's native
   order) that `GetValidationReport`/`GetAssessmentReport` depend on for the
   `dtf` model type. The AOI path does **not** produce a full-layer
   GeoPackage in that same row-order contract — it PATCHes a subset of
   `hg_buildings` rows keyed by `building_id` (a stable identifier), not by
   positional index. Do not wire the existing row-order-dependent report
   code to `dtf-aoi` models without adapting it to key on `building_id`.
4. **Load the checkpoint once per job, not once per building** (§6.5) — the
   single largest performance risk plan.pdf's per-building framing invites
   if taken too literally.
5. **AOI count has no server-side cap in this plan.** If an analyst draws a
   box over a dense downtown core, `ST_Intersects` can still return tens of
   thousands of buildings. Consider a soft warning in `RunDtfAoiButton.jsx`
   (via the optional `GetBuildingsInAoi` preview endpoint) above some
   threshold (e.g. 5,000) before submit, so an analyst doesn't accidentally
   queue a job with whole-layer-scale cost through the per-building-crop
   path, which is slower per-building than the whole-layer path's batched
   tile inference.

---

## Rollout order

1. `db/migrations/0001_hg_buildings.sql` + `postgis_load.py` + wire into
   `prepare_imagery.py` — land and verify row counts before anything else
   depends on the table.
2. `postgis_query.py` (`buildings_in_aoi`, `update_building_predictions`,
   `set_review_status`) — unit-testable against a local PostGIS/Postgres
   with PostGIS extension, no Azure Batch/queue involved yet.
3. `crop_service.py` + `damage_triage_aoi_infer.py` — testable standalone
   against a test COG + a handful of footprints, exactly like
   `OptionB_Quick_Start.md` §2's "prove it works alone before integrating"
   step, before touching the queue/API layer.
4. `Model` field additions + `damage_triage_aoi.py` processor +
   `PutRunDTFAoiInferenceQueueMessage` — backend loop closed, testable via
   curl/Postman without the frontend.
5. `AoiDrawTool.jsx` + `RunDtfAoiButton.jsx` — frontend capture + send.
6. `DtfAoiReviewPanel.jsx` + `PutBuildingDTFReview` + Visualizer coloring —
   review loop closed.
7. End-to-end test (above), then AOI-scoped export.

This order mirrors `OptionB_Quick_Start.md`'s own advice ("Section 11 is the
load-bearing wall... do it first") applied to this plan's equivalent risk
concentration: the PostGIS table and query layer, then the per-building
inference contract, then the API/queue plumbing, then the UI — each stage
independently testable before the next depends on it.
