# Building Damage Assessment Pipeline: Project Overview & Implementation Options

## Executive Summary

**Project Goal:** Build an end-to-end, production-grade pipeline for automated post-disaster building damage assessment from satellite imagery. The pipeline takes post-event satellite images and building footprints as input, produces per-building damage predictions, and returns an interactive map with damage classifications and summary reports.

**Current Decision Point:** Two distinct implementation paths with different trade-offs in speed-to-deploy, automation level, accuracy, and code ownership.

**Outcome:** Interactive map showing building-by-building damage typology, exportable GeoPackage with predictions, and summary statistics with confidence intervals.

---

## 1. What the problem is

Post-disaster response teams need rapid (hours, not days) building damage assessment to prioritize aid, route emergency resources, and understand the scope of disaster. Current workflows rely on manual inspection (slow) or one-off ML models that don't generalize across disasters.

**Key constraints:**
- Imagery may not have a matched pre-disaster baseline (pre/post alignment is hard)
- Sensor/resolution/angle changes every disaster
- Need both **per-building typology** (what kind of damage) and **confidence scores** (how sure is the model)
- Human analyst review is sometimes needed, sometimes not (varies by use case)

---

## 2. Research: Three candidate systems

We evaluated three systems: **DAHiTra**, **HASTE**, and **DamageTriageFormer**.

### 2.1 DAHiTra (nka77/DAHiTra on GitHub)

| Aspect | Detail |
| --- | --- |
| **What it is** | A research model (2023) using hierarchical transformers for change detection |
| **Input** | Pre-disaster + post-disaster RGB tiles (matched pair required) |
| **Output** | 4-level damage severity (no damage / minor / major / destroyed) OR binary change detection |
| **Code** | Full training code (`train.py`, `train_loc.py`, `train_adapt.py`) + architecture |
| **Trained weights** | One checkpoint on LEVIR-CD (generic change detection, not damage) |
| **Strengths** | Accurate on benchmarks (xBD); hierarchical design handles multi-scale features |
| **Weaknesses** | Requires matched pre/post imagery (rare in real deployments); no pretrained weights for damage classification; train-your-own workflow required |
| **Usable as-is?** | Only the architecture. The checkpoint is for change detection, not damage typology. |

### 2.2 HASTE (microsoft/haste on GitHub)

| Aspect | Detail |
| --- | --- |
| **What it is** | A complete platform: React UI + REST API + queue workers + tile server + model training/inference |
| **Input** | Post-event imagery only (no pre-image needed). Building footprints. Analyst labels. |
| **Output** | Per-building damage predictions (4 severity levels), interactive map, report with uncertainty quantification |
| **Code** | Full stack (Azure Functions for API, Vite React UI, hastegeo core library, Docker local dev) |
| **Trained weights** | No pretrained model. Trains fresh from analyst labels per scene (Method 1: U-Net; Method 2: embedding + logistic regression) |
| **Strengths** | End-to-end platform; no pre/post requirement; per-event adaptation avoids generalization failure; UI/map/report already polished; used operationally 31+ times |
| **Weaknesses** | Human-in-loop (analyst must label every scene); no fixed versioned model; heavy infrastructure (Azure); Method 1 training is slow |
| **Usable as-is?** | Completely. Docker Compose starts it, you add imagery + footprints + labels + get colored map. |

### 2.3 DamageTriageFormer (Ymx1025 on Hugging Face)

| Aspect | Detail |
| --- | --- |
| **What it is** | A single pretrained checkpoint (June 2026) trained on Hurricane Michael, Helene, and LA wildfires |
| **Input** | Post-event RGB tile (1024×1024) + building instance masks (footprints rasterized to pixel IDs) |
| **Output** | 5-class typology per building (Undamaged / Partial Roof / Total Roof / Partial Structural / Total Structural Collapse) |
| **Code** | Inference + training code on GitHub (YimingXiao98/Damage-TriageFormer, MIT license) |
| **Trained weights** | `best_model.pth` (~500 MB) + `config.json` on Hugging Face (CC-BY-NC-4.0 license) |
| **Architecture** | DINOv3 ViT-L/16 backbone → Feature Pyramid → mask-pooled instance features → two-stage gated head |
| **Strengths** | Pretrained + ready-to-use; mono-temporal (no pre-image needed); 5-class typology is decision-relevant; foundation model (designed for fine-tuning); already packaged |
| **Weaknesses** | Single checkpoint (no uncertainty quantification in predictions); weak on roof-damage classification (F1 ≈ 0.33); CC-BY-NC-4.0 (non-commercial only); no interactive UI |
| **Usable as-is?** | The model inference, yes. But you need to build the ingestion pipeline, footprint sourcing, postprocessing, and UI yourself. |

---

## 3. The general pipeline shape (regardless of option)

```
┌─────────────────────────────────────────────────────────────────┐
│ INGESTION                                                        │
│ - Post-event satellite imagery (GeoTIFF, Cloud-Optimized)       │
│ - Building footprints (GeoJSON, from Overture/MS Global ML)     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING                                                    │
│ - Tile imagery (1024×1024 chunks)                               │
│ - Clip footprints to each tile's bounds                         │
│ - Rasterize footprints → instance masks (building IDs per pixel)│
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ PREDICTION                                                       │
│ Model input: tile RGB + instance masks                          │
│ Model output: class prediction per building                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ POSTPROCESSING                                                   │
│ - Buffer footprints by 5–20m to handle misalignment             │
│ - Aggregate per-pixel predictions to per-building               │
│ - Join back to original footprint geometries                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                           │
│ - Per-building GeoPackage (footprint + class + confidence)      │
│ - Interactive map (buildings colored by class)                  │
│ - Summary report (counts, % per class, confidence intervals)    │
└─────────────────────────────────────────────────────────────────┘
```

All three systems handle this shape somehow; they differ in how much is automated vs. manual, what's shipped vs. what you build.

---

## 4. The two implementation options

### Option A: Use HASTE as-is, drop DamageTriageFormer

**What you do:**
1. Clone HASTE, `docker compose up`
2. Create a project in the UI
3. Upload post-event imagery
4. Upload building footprints
5. Annotate a handful of buildings (draw damage polygons OR click-label)
6. Start a training job
7. Review colored map + export

**How the pipeline fills in:**

```
HASTE UI (ingestion + annotation)
        │
        ▼
HASTE preprocessing (tile + footprints → instance masks)
        │
        ▼
HASTE Method 1 or 2 training (analyst labels → U-Net or embedding classifier)
        │
        ▼
HASTE inference on whole layer
        │
        ▼
HASTE postprocessing (pixel predictions → building predictions)
        │
        ▼
HASTE output (map + report)
```

**Pros:**
- Zero integration work — works completely out of the box
- Polished, production-ready UI/map/report
- Human-in-the-loop can catch errors (sometimes good, sometimes friction)
- No external model dependencies
- Operational track record (31+ real disasters)

**Cons:**
- Every scene requires a human to label (slower)
- Output is 4-class severity (No/Minor/Major/Destroyed), not 5-class typology
- No fixed versioned model (hard to monitor/audit)
- Model is trained only on *that scene's* labels (may overfit if labels are sparse)
- Heavy infrastructure footprint (Azure, though local dev works with Docker)

**Cost:** ~2–3 hours end-to-end per disaster (depends on annotation speed)

**License:** HASTE itself is open-source; no commercial restrictions

---

### Option B: HASTE map/UI shell + DamageTriageFormer brain transplant

**What you do:**
1. Clone HASTE, `docker compose up`, verify Option A works
2. Clone DamageTriageFormer code, download weights from Hugging Face
3. Fork HASTE; add new `hastegeo` processor for DamageTriageFormer inference
4. Add new REST API endpoint to receive annotations from the map
5. Wire the frontend to capture annotations (drawn polygons or selected buildings)
6. Add job status polling and prediction-layer fetching
7. Update the map legend to show 5 typology classes
8. Test end-to-end: annotate → detect → send → infer → render

**How the pipeline fills in:**

```
HASTE UI (annotation capture)
        │
        ▼
Frontend hook (extract drawn polygons or building IDs)
        │
        ▼
POST /api/infer/damage-triage (send annotation to backend)
        │
        ▼
Backend: resolve footprints (selected or intersecting drawn area)
        │
        ▼
Rasterize footprints → instance masks
        │
        ▼
DamageTriageFormer inference (5-class per building)
        │
        ▼
Convert to HASTE's native prediction-layer format
        │
        ▼
Write to storage
        │
        ▼
Frontend polling + fetch prediction layer
        │
        ▼
Map re-renders (5 typology classes, same HASTE rendering code)
        │
        ▼
HASTE output (map + report)
```

**Pros:**
- Keeps HASTE's polished UI/map/report/footprint handling
- Fully automated inference (no per-scene labeling after initial setup)
- 5-class typology output (more decision-relevant than severity)
- DamageTriageFormer is a vetted, pretrained foundation model
- Versioned, monitorable model (same checkpoint across disasters)
- Analyst can optionally review before export (keep human-in-loop if needed)

**Cons:**
- Requires forking HASTE and writing integration code
- Single annotation → DamageTriageFormer pipeline not tested (you're building it)
- Prediction-layer format mismatch is the biggest risk (HASTE internals you must reverse-engineer)
- DamageTriageFormer weights are CC-BY-NC-4.0 (non-commercial only)
- More moving parts (more ways to break)
- Annotation-capture hook could be fragile if HASTE UI updates

**Effort:** 1–2 weeks (forking, integration, testing, debugging)

**License:** DamageTriageFormer weights CC-BY-NC-4.0; code MIT; your integration code is yours

---

## 5. Side-by-side comparison

| Dimension | Option A (HASTE only) | Option B (HASTE + DamageTriageFormer) |
| --- | --- | --- |
| **Time to first result** | 2–3 hours | 1–2 weeks + training time |
| **Per-scene human work** | Annotate several buildings | None (fully automated) |
| **Output taxonomy** | 4 severity classes | 5 typology classes |
| **Model stability** | Per-scene (no cross-disaster generalization) | Single pretrained model (stable, monitored) |
| **UI/Map/Report** | Built-in, polished | Built-in (reused from HASTE) |
| **Ingestion/footprint handling** | Built-in | Built-in (reused from HASTE) |
| **Code ownership** | None (use as-is) | Fork HASTE + write integration |
| **Integration risk** | None | Medium (prediction-layer format match) |
| **Commercial use** | Yes | No (weights CC-BY-NC-4.0) |
| **Scalability** | Per-scene retraining (expensive) | Single model (efficient) |
| **Audit/compliance** | Hard (model changes per scene) | Easy (fixed versioned model) |

---

## 6. When to pick each option

**Pick Option A if:**
- You need results in hours (not days/weeks of integration)
- You have analysts available to label scenes
- You prefer the safety of a battle-tested platform (HASTE has 31+ real-world uses)
- You can tolerate a different output format (4-class severity vs. 5-class typology)
- Your use case is research / non-commercial

**Pick Option B if:**
- You want a fully automated pipeline (no per-scene labeling)
- You need 5-class typology output (more nuanced damage assessment)
- You're willing to invest 1–2 weeks in integration & testing
- Your organization can commit a developer for fork maintenance
- You want a single, versioned, monitored model (better for compliance/audit)
- Your use case is research or internal-only (not commercial, due to CC-BY-NC-4.0)

---

## 7. Implementation checklist for each option

### Option A checklist

- [ ] Docker + Docker Compose installed
- [ ] HASTE cloned and `docker compose up` succeeds
- [ ] Post-event GeoTIFF sourced (Cloud-Optimized format preferred)
- [ ] Building footprints downloaded + clipped to tile bounds
- [ ] Project created in HASTE UI
- [ ] Imagery layer uploaded and tiled on map
- [ ] Footprints layer uploaded and visible on map
- [ ] Annotation complete (Method 1 polygons or Method 2 click-labels, 5–30 buildings)
- [ ] Training job queued and completed (check worker logs)
- [ ] Predictions visible on map (buildings colored)
- [ ] Report generated and reviewed
- [ ] Predictions exported to GeoPackage

**Estimated time:** 2–3 hours

---

### Option B checklist

- [ ] **Option A is 100% working** (prerequisite; don't start B until A works)
- [ ] DamageTriageFormer code cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Weights downloaded from Hugging Face
- [ ] DamageTriageFormer runs standalone on a test tile + masks
- [ ] HASTE forked on GitHub
- [ ] `hastegeo/processors/damage_triage_infer.py` written and imports work
- [ ] `hastegeo/jobs/damage_triage_job.py` written and registers as a job type
- [ ] `api/functions/infer_damage_triage.py` (new endpoint) written
- [ ] `api/functions/get_job_status.py` (status polling) written
- [ ] Frontend hook `useDamageTriageAnnotation.ts` captures annotations
- [ ] Annotation payload successfully POSTs to `/api/infer/damage-triage`
- [ ] Worker picks up `damage_triage_infer` job and processes
- [ ] Prediction-layer format matches HASTE's native format (Section 11 of Option B guide is critical)
- [ ] Job reaches `"complete"` status
- [ ] Frontend fetches `/api/layers/{layer_id}/predictions` and renders
- [ ] Map colors buildings with 5 typology classes
- [ ] UI legend updated to 5 classes
- [ ] Export produces GeoPackage with 5 classes
- [ ] Annotation round-trip tested end-to-end (draw/select → infer → render)

**Estimated time:** 1–2 weeks (including debugging)

---

## 8. Risk assessment

### Option A risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| HASTE stack won't start locally | Low | Medium | Docker + Compose troubleshooting guides in docs |
| Imagery doesn't tile efficiently | Medium | Low | Convert to Cloud-Optimized GeoTIFF (rio cogeo) |
| Training job hangs on CPU | Medium | Medium | Use GPU or switch to Method 2 (no training needed) |
| Footprints CRS mismatch with imagery | Medium | Medium | Verify CRS match, re-clip if needed |
| Analyst labels are insufficient | Low | High | Label more buildings; HASTE UI makes this easy |

**Overall risk:** Low. HASTE is proven; failures are usually environmental (Docker, CRS, GPU).

---

### Option B risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Prediction-layer format mismatch | High | High | **Do Section 11 first.** Capture real HASTE output, reverse-engineer format exactly |
| Instance-mask rasterization fails | Medium | High | Test standalone footprint→mask conversion; verify pixel alignment |
| Frontend annotation capture is fragile | Medium | Medium | Inspect HASTE labeling UI code; test with real click/draw actions |
| DamageTriageFormer OOM on GPU | Low | Medium | Reduce tile size or batch size; test with smaller imagery first |
| Model inference produces malformed output | Low | High | Parse standalone model output first; verify it matches expected schema |
| Job status polling times out | Low | Low | Increase poll interval; add timeout + error handling in frontend |

**Overall risk:** Medium–High. Integration is complex; prediction-layer format is the critical path.

---

## 9. Decision tree

```
Start here: Do you want results in hours or weeks?
    │
    ├─► HOURS → Option A
    │        (HASTE as-is; accept 4-class severity + human annotation per scene)
    │
    └─► WEEKS → Need fully automated + 5-class typology?
            │
            ├─► YES → Option B
            │      (Fork HASTE, integrate DamageTriageFormer, 1–2 week build)
            │
            └─► NO → Go back to Option A
```

---

## 10. Next steps

**If you choose Option A:**
1. Follow `OptionA_Quick_Start.md`
2. Load a post-event scene from xBD or NOAA
3. Annotate 10–20 buildings
4. Run inference and review results
5. Iterate (label more buildings if accuracy is low)

**If you choose Option B:**
1. Verify Option A works completely (don't skip this)
2. Follow `OptionB_Quick_Start.md` step by step
3. **Section 11 is the load-bearing wall:** capture HASTE's native prediction format before writing code
4. Write processor, API endpoint, frontend hook
5. Test with a small tile + a few buildings first
6. Scale up once end-to-end loop works

---

## 11. References

- **HASTE paper:** [arXiv:2607.11838](https://arxiv.org/abs/2607.11838)
- **DamageTriageFormer paper:** [arXiv:2606.12248](https://arxiv.org/abs/2606.12248)
- **DAHiTra paper:** Kaur et al., *Computer-Aided Civil and Infrastructure Engineering*, 2023
- **xBD dataset:** [huggingface.co/datasets/satlaspretrain/xBD](https://huggingface.co/datasets/satlaspretrain/xBD)
- **Overture Maps:** [beta.overturemaps.org](https://beta.overturemaps.org/)
- **Microsoft Global ML Building Footprints:** [github.com/microsoft/GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)

---

## 12. Project artifacts

Two companion implementation guides:
- **OptionA_Quick_Start.md** — Step-by-step, ready-to-run HASTE setup (no code changes needed)
- **OptionB_Quick_Start.md** — Fork + integrate guide with full code samples for processor, API, frontend hooks, and the full annotation→infer→render loop

Both assume Docker is already installed. Both include troubleshooting checklists.
