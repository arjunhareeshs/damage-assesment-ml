# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic aggregation of multi-window evidence into final building results.

Picks the winning prediction per building:
1. Highest coverage_fraction
2. Tie-break: highest confidence
3. Tie-break: lowest unit_id (alphabetical)

Never averages probabilities. Never depends on evidence arrival order.
Outputs class probabilities and categories only; damage_percentage is always None.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from hastegeo.core.models.projects import BuildingResultStatus
from hastegeo.core.utils.dtf_constants import DTF_CATEGORIES

logger = logging.getLogger(__name__)


def aggregate(
    evidence_rows: List[Dict[str, Any]],
    eligible_buildings: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    assessment_id: Optional[str] = None,
    image_layer_id: Optional[str] = None,
    model_digest: Optional[str] = None,
    preprocessing_version: Optional[str] = None,
    footprint_version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate per-unit evidence into final per-building assessment rows.

    Args:
        evidence_rows: List of dicts with {building_id, unit_id, probs, class,
            confidence, coverage_fraction, valid_pixel_fraction}.
        eligible_buildings: List of dicts representing all buildings in the AOI:
            {building_id, boundary_intersecting?, aoi_overlap_fraction?, status?}.
        config: Resolved assessment parameters (uncertain_confidence_threshold, etc.).
        assessment_id: Identifier of the parent assessment.
        image_layer_id: Identifier of the image layer.
        model_digest: Sha256 hash of the checkpoint used.
        preprocessing_version: Normalization/preprocessing pipeline version.
        footprint_version: Version of the footprints dataset.

    Returns:
        List[Dict[str, Any]]: Final assessment results for hg_damage_assessments.
    """
    config = config or {}
    uncertain_threshold = config.get("uncertain_confidence_threshold")
    if uncertain_threshold is not None:
        try:
            uncertain_threshold = float(uncertain_threshold)
        except (ValueError, TypeError):
            uncertain_threshold = None

    min_valid_fraction = float(config.get("min_valid_pixel_fraction", 0.5))

    # Group evidence by building_id, filtering out evidence below min_valid_pixel_fraction
    evidence_by_bldg: Dict[str, List[Dict[str, Any]]] = {}
    for ev in evidence_rows:
        if float(ev.get("valid_pixel_fraction", 1.0)) < min_valid_fraction:
            continue
        b_id = str(ev["building_id"])
        evidence_by_bldg.setdefault(b_id, []).append(ev)

    results: List[Dict[str, Any]] = []

    for b in eligible_buildings:
        b_id = str(b["building_id"])
        b_layer_id = b.get("image_layer_id") or image_layer_id or ""
        boundary_int = bool(b.get("boundary_intersecting", False))
        overlap_frac = float(b.get("aoi_overlap_fraction", 1.0))

        b_evidence = evidence_by_bldg.get(b_id, [])

        if b_evidence:
            # Sort deterministically:
            # - Primary: highest coverage_fraction (descending)
            # - Secondary: highest confidence (descending)
            # - Tertiary: lowest unit_id (alphabetical ascending)
            sorted_ev = sorted(
                b_evidence,
                key=lambda x: (
                    -float(x.get("coverage_fraction", 0.0)),
                    -float(x.get("confidence", 0.0)),
                    str(x.get("unit_id", "")),
                ),
            )
            winner = sorted_ev[0]
            conf = float(winner["confidence"])
            cls_idx = int(winner["class"])
            cls_name = (
                DTF_CATEGORIES[cls_idx]
                if 0 <= cls_idx < len(DTF_CATEGORIES)
                else None
            )

            # Determine status
            if uncertain_threshold is not None and conf < uncertain_threshold:
                res_status = BuildingResultStatus.UNCERTAIN.value
            else:
                res_status = BuildingResultStatus.COMPLETED.value

            results.append({
                "assessment_id": assessment_id or b.get("assessment_id", ""),
                "building_id": b_id,
                "image_layer_id": b_layer_id,
                "damage_class": cls_idx,
                "damage_class_name": cls_name,
                "confidence": conf,
                "damage_percentage": None,  # Model outputs class probabilities only
                "result_status": res_status,
                "boundary_intersecting": boundary_int,
                "aoi_overlap_fraction": overlap_frac,
                "winning_unit_id": winner.get("unit_id"),
                "model_digest": model_digest or b.get("model_digest"),
                "preprocessing_version": preprocessing_version or b.get("preprocessing_version"),
                "footprint_version": footprint_version or b.get("footprint_version"),
            })
        else:
            # Building has no evidence: determine whether nodata or failed
            # If the building was flagged with failed unit attempts vs insufficient data
            hint_status = b.get("status")
            if hint_status == BuildingResultStatus.FAILED.value:
                res_status = BuildingResultStatus.FAILED.value
            else:
                res_status = BuildingResultStatus.INSUFFICIENT_DATA.value

            results.append({
                "assessment_id": assessment_id or b.get("assessment_id", ""),
                "building_id": b_id,
                "image_layer_id": b_layer_id,
                "damage_class": None,
                "damage_class_name": None,
                "confidence": None,
                "damage_percentage": None,
                "result_status": res_status,
                "boundary_intersecting": boundary_int,
                "aoi_overlap_fraction": overlap_frac,
                "winning_unit_id": None,
                "model_digest": model_digest or b.get("model_digest"),
                "preprocessing_version": preprocessing_version or b.get("preprocessing_version"),
                "footprint_version": footprint_version or b.get("footprint_version"),
            })

    return results
