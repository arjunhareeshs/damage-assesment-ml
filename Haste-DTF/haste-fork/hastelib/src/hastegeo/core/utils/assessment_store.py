# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PostGIS persistence store for Large-AOI batched damage assessments.

Provides parameterized CRUD operations and spatial queries for:
- hg_assessments
- hg_processing_units
- hg_unit_buildings
- hg_batches
- hg_prediction_evidence
- hg_damage_assessments

Follows the strict ``conn=None`` injection pattern so callers can share
transactions or test with mocked connection objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shapely.geometry import mapping

from .postgis_connect import get_connection

logger = logging.getLogger(__name__)


def create_assessment(assessment: Dict[str, Any], conn=None) -> Dict[str, Any]:
    """Insert a new AOI assessment record.

    Args:
        assessment: Assessment attributes dictionary.
        conn: Optional external database connection.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO hg_assessments (
                    assessment_id, project_id, image_layer_id, event_id, name,
                    aoi_geometry, aoi_area_km2, inclusion_policy, status,
                    config, model_repo, model_digest, preprocessing_version,
                    eligible_buildings, total_units, completed_units, failed_units,
                    submitted_by, error, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, NOW(), NOW()
                ) RETURNING assessment_id
                """,
                (
                    assessment["assessment_id"],
                    assessment["project_id"],
                    assessment["image_layer_id"],
                    assessment.get("event_id"),
                    assessment.get("name"),
                    json.dumps(assessment["aoi_geometry"]),
                    assessment["aoi_area_km2"],
                    assessment.get("inclusion_policy", "intersects_aoi"),
                    assessment.get("status", "submitted"),
                    json.dumps(assessment.get("config", {})),
                    assessment.get("model_repo"),
                    assessment.get("model_digest"),
                    assessment.get("preprocessing_version", "1.0.0"),
                    assessment.get("eligible_buildings"),
                    assessment.get("total_units"),
                    assessment.get("completed_units", 0),
                    assessment.get("failed_units", 0),
                    assessment.get("submitted_by"),
                    assessment.get("error"),
                ),
            )
        connection.commit()
        return assessment
    finally:
        if owns_conn:
            connection.close()


def get_assessment(assessment_id: str, conn=None) -> Optional[Dict[str, Any]]:
    """Retrieve an assessment by ID with GeoJSON AOI geometry.

    Args:
        assessment_id: Canonical assessment identifier.
        conn: Optional external database connection.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    assessment_id, project_id, image_layer_id, event_id, name,
                    ST_AsGeoJSON(aoi_geometry), aoi_area_km2, inclusion_policy,
                    status, config, model_repo, model_digest, preprocessing_version,
                    eligible_buildings, total_units, completed_units, failed_units,
                    submitted_by, error, created_at, updated_at, completed_at
                FROM hg_assessments
                WHERE assessment_id = %s
                """,
                (assessment_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "assessment_id": row[0],
                "project_id": row[1],
                "image_layer_id": row[2],
                "event_id": row[3],
                "name": row[4],
                "aoi_geometry": json.loads(row[5]),
                "aoi_area_km2": float(row[6]),
                "inclusion_policy": row[7],
                "status": row[8],
                "config": row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
                "model_repo": row[10],
                "model_digest": row[11],
                "preprocessing_version": row[12],
                "eligible_buildings": row[13],
                "total_units": row[14],
                "completed_units": row[15],
                "failed_units": row[16],
                "submitted_by": row[17],
                "error": row[18],
                "created_at": row[19].isoformat() if row[19] else None,
                "updated_at": row[20].isoformat() if row[20] else None,
                "completed_at": row[21].isoformat() if row[21] else None,
            }
    finally:
        if owns_conn:
            connection.close()


TERMINAL_STATUSES = ("completed", "completed_with_errors", "failed", "cancelled", "rejected")


def update_assessment_status(
    assessment_id: str,
    status: str,
    error: Optional[str] = None,
    conn=None,
    *,
    eligible_buildings: Optional[int] = None,
    total_units: Optional[int] = None,
    completed_units: Optional[int] = None,
    failed_units: Optional[int] = None,
    model_digest: Optional[str] = None,
) -> bool:
    """Update assessment status, optional error, and optional counts.

    Sets ``completed_at`` when ``status`` is terminal. ``None`` keyword
    values leave the stored column unchanged.

    Args:
        assessment_id: Canonical assessment identifier.
        status: New AoiAssessmentStatus string.
        error: Optional error description string.
        conn: Optional external database connection.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            completed_clause = (
                ", completed_at = NOW()" if status in TERMINAL_STATUSES else ""
            )
            cursor.execute(
                f"""
                UPDATE hg_assessments
                SET status = %s,
                    error = COALESCE(%s, error),
                    eligible_buildings = COALESCE(%s, eligible_buildings),
                    total_units = COALESCE(%s, total_units),
                    completed_units = COALESCE(%s, completed_units),
                    failed_units = COALESCE(%s, failed_units),
                    model_digest = COALESCE(%s, model_digest),
                    updated_at = NOW()
                    {completed_clause}
                WHERE assessment_id = %s
                """,
                (
                    status,
                    error,
                    eligible_buildings,
                    total_units,
                    completed_units,
                    failed_units,
                    model_digest,
                    assessment_id,
                ),
            )
            updated = cursor.rowcount > 0
        connection.commit()
        return updated
    finally:
        if owns_conn:
            connection.close()


def update_assessment_counts(
    assessment_id: str,
    eligible_buildings: Optional[int] = None,
    total_units: Optional[int] = None,
    completed_units: Optional[int] = None,
    failed_units: Optional[int] = None,
    conn=None,
) -> bool:
    """Update progress and building/unit counts for an assessment."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_assessments
                SET eligible_buildings = COALESCE(%s, eligible_buildings),
                    total_units = COALESCE(%s, total_units),
                    completed_units = COALESCE(%s, completed_units),
                    failed_units = COALESCE(%s, failed_units),
                    updated_at = NOW()
                WHERE assessment_id = %s
                """,
                (eligible_buildings, total_units, completed_units, failed_units, assessment_id),
            )
            updated = cursor.rowcount > 0
        connection.commit()
        return updated
    finally:
        if owns_conn:
            connection.close()


def _geojson(geom: Any) -> str:
    """GeoJSON text for a shapely geometry or an already-GeoJSON dict."""
    if hasattr(geom, "__geo_interface__"):
        geom = mapping(geom)
    return json.dumps(geom)


def insert_processing_units(
    units: List[Dict[str, Any]],
    assessment_id: Optional[str] = None,
    conn=None,
) -> int:
    """Insert processing unit windows and their building associations.

    Each unit's ``buildings`` list (``{building_id, coverage_fraction}``,
    as produced by ``region_windows.select_units``) is written to
    ``hg_unit_buildings`` in the same transaction. Re-inserting the same
    unit ids is a no-op, so a retried tiling step cannot duplicate rows.
    """
    if not units:
        return 0

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO hg_processing_units (
                    unit_id, assessment_id, imagery_id, batch_id,
                    col_off, row_off, width, height, bounds,
                    kind, status, attempts, valid_pixel_fraction, error
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326),
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (unit_id) DO NOTHING
                """,
                [
                    (
                        u["unit_id"],
                        u.get("assessment_id") or assessment_id,
                        u["imagery_id"],
                        u.get("batch_id"),
                        u["col_off"],
                        u["row_off"],
                        u["width"],
                        u["height"],
                        _geojson(u["bounds"]),
                        u.get("kind", "grid"),
                        u.get("status", "pending"),
                        u.get("attempts", 0),
                        u.get("valid_pixel_fraction"),
                        u.get("error"),
                    )
                    for u in units
                ],
            )
            associations = [
                (u["unit_id"], b["building_id"], b["coverage_fraction"])
                for u in units
                for b in u.get("buildings", [])
            ]
            if associations:
                cursor.executemany(
                    """
                    INSERT INTO hg_unit_buildings (unit_id, building_id, coverage_fraction)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (unit_id, building_id) DO UPDATE
                    SET coverage_fraction = EXCLUDED.coverage_fraction
                    """,
                    associations,
                )
            count = len(units)
        connection.commit()
        return count
    finally:
        if owns_conn:
            connection.close()


def insert_unit_buildings(associations: List[Dict[str, Any]], conn=None) -> int:
    """Batch insert unit-to-building coverage associations."""
    if not associations:
        return 0

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO hg_unit_buildings (unit_id, building_id, coverage_fraction)
                VALUES (%s, %s, %s)
                ON CONFLICT (unit_id, building_id) DO UPDATE
                SET coverage_fraction = EXCLUDED.coverage_fraction
                """,
                [
                    (a["unit_id"], a["building_id"], a["coverage_fraction"])
                    for a in associations
                ],
            )
            count = len(associations)
        connection.commit()
        return count
    finally:
        if owns_conn:
            connection.close()


def insert_batches(
    batches: List[Dict[str, Any]],
    assessment_id: Optional[str] = None,
    conn=None,
) -> int:
    """Batch insert execution batch definitions (no-op for existing ids).

    Insert batches before their units: ``hg_processing_units.batch_id``
    references ``hg_batches``.
    """
    if not batches:
        return 0

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO hg_batches (
                    batch_id, assessment_id, status, attempts,
                    runner_job_id, runner_task_id, lease_expires_at, error,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    NOW(), NOW()
                )
                ON CONFLICT (batch_id) DO NOTHING
                """,
                [
                    (
                        b["batch_id"],
                        b.get("assessment_id") or assessment_id,
                        b.get("status", "pending"),
                        b.get("attempts", 0),
                        b.get("runner_job_id"),
                        b.get("runner_task_id"),
                        b.get("lease_expires_at"),
                        b.get("error"),
                    )
                    for b in batches
                ],
            )
            count = len(batches)
        connection.commit()
        return count
    finally:
        if owns_conn:
            connection.close()


def claim_next_batch(assessment_id: str, lease_seconds: int, conn=None) -> Optional[Dict[str, Any]]:
    """Atomically claim the next pending batch for execution.

    Sets status='running', lease_expires_at=NOW() + lease_seconds, and increments attempts.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = 'running',
                    attempts = attempts + 1,
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE batch_id = (
                    SELECT batch_id
                    FROM hg_batches
                    WHERE assessment_id = %s AND status = 'pending'
                    ORDER BY batch_id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING batch_id, assessment_id, attempts
                """,
                (lease_seconds, assessment_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            batch_id, a_id, attempts = row[0], row[1], row[2]

            # Also fetch units belonging to this batch
            cursor.execute(
                """
                SELECT
                    u.unit_id, u.imagery_id, u.col_off, u.row_off, u.width, u.height,
                    i.file_uri
                FROM hg_processing_units u
                JOIN hg_imagery i ON u.imagery_id = i.imagery_id
                WHERE u.batch_id = %s
                """,
                (batch_id,),
            )
            unit_rows = cursor.fetchall()
            units = [
                {
                    "unit_id": u[0],
                    "imagery_id": u[1],
                    "col_off": u[2],
                    "row_off": u[3],
                    "width": u[4],
                    "height": u[5],
                    "file_uri": u[6],
                }
                for u in unit_rows
            ]

        connection.commit()
        return {
            "batch_id": batch_id,
            "assessment_id": a_id,
            "attempts": attempts,
            "units": units,
        }
    finally:
        if owns_conn:
            connection.close()


def renew_batch_lease(batch_id: str, lease_seconds: int, conn=None) -> bool:
    """Renew the lease on an actively running batch."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE batch_id = %s AND status = 'running'
                """,
                (lease_seconds, batch_id),
            )
            updated = cursor.rowcount > 0
        connection.commit()
        return updated
    finally:
        if owns_conn:
            connection.close()


def expire_leases(assessment_id: str, max_attempts: int = 3, conn=None) -> Tuple[int, int]:
    """Check for expired leases. Requeues expired batches below max_attempts; fails others.

    Returns:
        Tuple[int, int]: (requeued_count, failed_count)
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        requeued = 0
        failed = 0
        with connection.cursor() as cursor:
            # Requeue batches below max_attempts
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = 'pending',
                    lease_expires_at = NULL,
                    updated_at = NOW()
                WHERE assessment_id = %s
                  AND status = 'running'
                  AND lease_expires_at < NOW()
                  AND attempts < %s
                """,
                (assessment_id, max_attempts),
            )
            requeued = cursor.rowcount

            # Mark batches exceeding max_attempts as failed
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = 'failed',
                    error = 'Exceeded maximum batch execution attempts after lease expiration',
                    updated_at = NOW()
                WHERE assessment_id = %s
                  AND status = 'running'
                  AND lease_expires_at < NOW()
                  AND attempts >= %s
                """,
                (assessment_id, max_attempts),
            )
            failed = cursor.rowcount

        connection.commit()
        return requeued, failed
    finally:
        if owns_conn:
            connection.close()


def complete_batch(batch_id: str, runner_job_id: Optional[str] = None, runner_task_id: Optional[str] = None, conn=None) -> bool:
    """Mark a batch and its child processing units as completed."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = 'completed',
                    runner_job_id = COALESCE(%s, runner_job_id),
                    runner_task_id = COALESCE(%s, runner_task_id),
                    lease_expires_at = NULL,
                    updated_at = NOW()
                WHERE batch_id = %s
                """,
                (runner_job_id, runner_task_id, batch_id),
            )
            cursor.execute(
                """
                UPDATE hg_processing_units
                SET status = 'completed'
                WHERE batch_id = %s AND status = 'pending'
                """,
                (batch_id,),
            )
        connection.commit()
        return True
    finally:
        if owns_conn:
            connection.close()


def fail_batch(batch_id: str, error: Optional[str] = None, conn=None) -> bool:
    """Mark a batch and its child processing units as failed."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = 'failed',
                    error = %s,
                    lease_expires_at = NULL,
                    updated_at = NOW()
                WHERE batch_id = %s
                """,
                (error, batch_id),
            )
            cursor.execute(
                """
                UPDATE hg_processing_units
                SET status = 'failed',
                    error = %s
                WHERE batch_id = %s
                """,
                (error, batch_id),
            )
        connection.commit()
        return True
    finally:
        if owns_conn:
            connection.close()


def get_batches(assessment_id: str, conn=None) -> List[Dict[str, Any]]:
    """All batches of an assessment, ordered by batch_id."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT batch_id, assessment_id, status, attempts,
                       runner_job_id, runner_task_id, lease_expires_at, error
                FROM hg_batches
                WHERE assessment_id = %s
                ORDER BY batch_id
                """,
                (assessment_id,),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()
    return [
        {
            "batch_id": r[0],
            "assessment_id": r[1],
            "status": r[2],
            "attempts": r[3],
            "runner_job_id": r[4],
            "runner_task_id": r[5],
            "lease_expires_at": r[6].isoformat() if r[6] else None,
            "error": r[7],
        }
        for r in rows
    ]


def update_batch_status(
    batch_id: str,
    status: str,
    error: Optional[str] = None,
    runner_job_id: Optional[str] = None,
    runner_task_id: Optional[str] = None,
    conn=None,
) -> bool:
    """Set a batch's status (and optionally its runner ids / error).

    Leaving ``running`` clears the lease. ``failed`` also marks the batch's
    not-yet-completed units failed; ``completed`` is applied per unit by
    ``apply_unit_results`` instead, because a completed batch can still
    contain individually failed units.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = %s,
                    error = COALESCE(%s, error),
                    runner_job_id = COALESCE(%s, runner_job_id),
                    runner_task_id = COALESCE(%s, runner_task_id),
                    lease_expires_at = CASE WHEN %s = 'running'
                                            THEN lease_expires_at ELSE NULL END,
                    updated_at = NOW()
                WHERE batch_id = %s
                """,
                (status, error, runner_job_id, runner_task_id, status, batch_id),
            )
            updated = cursor.rowcount > 0
            if status == "failed":
                cursor.execute(
                    """
                    UPDATE hg_processing_units
                    SET status = 'failed', error = COALESCE(%s, error)
                    WHERE batch_id = %s AND status <> 'completed'
                    """,
                    (error, batch_id),
                )
        connection.commit()
        return updated
    finally:
        if owns_conn:
            connection.close()


def claim_batch_lease(batch_id: str, lease_seconds: int, conn=None) -> bool:
    """Atomically move one ``pending`` batch to ``running`` with a lease.

    Increments ``attempts``. Returns False if another worker claimed it
    first (the row is no longer pending), so two queue messages processed
    concurrently cannot both launch the same batch.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = 'running',
                    attempts = attempts + 1,
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE batch_id = %s AND status = 'pending'
                """,
                (lease_seconds, batch_id),
            )
            claimed = cursor.rowcount > 0
            if claimed:
                cursor.execute(
                    """
                    UPDATE hg_processing_units
                    SET status = 'running', attempts = attempts + 1
                    WHERE batch_id = %s AND status <> 'completed'
                    """,
                    (batch_id,),
                )
        connection.commit()
        return claimed
    finally:
        if owns_conn:
            connection.close()


def requeue_or_fail_batch(
    batch_id: str, max_attempts: int, error: str, conn=None
) -> str:
    """After a failed runner task: back to ``pending`` while attempts remain,
    else ``failed``. Returns the new status."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE hg_batches
                SET status = CASE WHEN attempts < %s THEN 'pending' ELSE 'failed' END,
                    error = %s,
                    lease_expires_at = NULL,
                    runner_task_id = NULL,
                    updated_at = NOW()
                WHERE batch_id = %s
                RETURNING status
                """,
                (max_attempts, error, batch_id),
            )
            row = cursor.fetchone()
            new_status = row[0] if row else "failed"
            cursor.execute(
                """
                UPDATE hg_processing_units
                SET status = CASE WHEN %s = 'failed' THEN 'failed' ELSE 'pending' END,
                    error = %s
                WHERE batch_id = %s AND status <> 'completed'
                """,
                (new_status, error, batch_id),
            )
        connection.commit()
        return new_status
    finally:
        if owns_conn:
            connection.close()


def get_batch_units(batch_id: str, conn=None) -> List[Dict[str, Any]]:
    """Units of one batch with their imagery and building associations.

    Returns ``[{unit_id, imagery_id, file_uri, col_off, row_off, width,
    height, kind, buildings: [{building_id, coverage_fraction}]}]``.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT u.unit_id, u.imagery_id, i.file_uri, u.col_off, u.row_off,
                       u.width, u.height, u.kind
                FROM hg_processing_units u
                JOIN hg_imagery i ON u.imagery_id = i.imagery_id
                WHERE u.batch_id = %s
                ORDER BY u.unit_id
                """,
                (batch_id,),
            )
            unit_rows = cursor.fetchall()
            cursor.execute(
                """
                SELECT ub.unit_id, ub.building_id, ub.coverage_fraction
                FROM hg_unit_buildings ub
                JOIN hg_processing_units u ON u.unit_id = ub.unit_id
                WHERE u.batch_id = %s
                ORDER BY ub.unit_id, ub.building_id
                """,
                (batch_id,),
            )
            link_rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()

    links: Dict[str, List[Dict[str, Any]]] = {}
    for unit_id, building_id, coverage in link_rows:
        links.setdefault(unit_id, []).append(
            {"building_id": building_id, "coverage_fraction": float(coverage)}
        )
    return [
        {
            "unit_id": r[0],
            "imagery_id": r[1],
            "file_uri": r[2],
            "col_off": r[3],
            "row_off": r[4],
            "width": r[5],
            "height": r[6],
            "kind": r[7],
            "buildings": links.get(r[0], []),
        }
        for r in unit_rows
    ]


def apply_unit_results(
    batch_id: str, unit_summaries: List[Dict[str, Any]], conn=None
) -> None:
    """Record per-unit outcomes from a batch's ``region_manifest.json``.

    Units listed as ``completed`` or ``failed`` take that status; any unit
    of the batch missing from the manifest is marked failed.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            if unit_summaries:
                cursor.executemany(
                    """
                    UPDATE hg_processing_units
                    SET status = %s, error = %s
                    WHERE unit_id = %s AND batch_id = %s
                    """,
                    [
                        (
                            "completed" if u.get("status") == "completed" else "failed",
                            u.get("error"),
                            u["unit_id"],
                            batch_id,
                        )
                        for u in unit_summaries
                    ],
                )
            cursor.execute(
                """
                UPDATE hg_processing_units
                SET status = 'failed',
                    error = COALESCE(error, 'missing from batch manifest')
                WHERE batch_id = %s AND status NOT IN ('completed', 'failed')
                """,
                (batch_id,),
            )
        connection.commit()
    finally:
        if owns_conn:
            connection.close()


def unit_status_counts(assessment_id: str, conn=None) -> Dict[str, int]:
    """``{status: count}`` over an assessment's processing units."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, COUNT(*) FROM hg_processing_units
                WHERE assessment_id = %s GROUP BY status
                """,
                (assessment_id,),
            )
            return {r[0]: int(r[1]) for r in cursor.fetchall()}
    finally:
        if owns_conn:
            connection.close()


def buildings_without_completed_units(assessment_id: str, conn=None) -> List[str]:
    """Buildings whose every associated unit failed (nothing was scored).

    These end as ``result_status='failed'``; buildings that had a completed
    unit but no evidence were skipped for nodata and end as
    ``insufficient_data``.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT ub.building_id
                FROM hg_unit_buildings ub
                JOIN hg_processing_units u ON u.unit_id = ub.unit_id
                WHERE u.assessment_id = %s
                GROUP BY ub.building_id
                HAVING bool_and(u.status <> 'completed')
                """,
                (assessment_id,),
            )
            return [r[0] for r in cursor.fetchall()]
    finally:
        if owns_conn:
            connection.close()


def eligible_buildings_with_overlap(
    image_layer_id: str, aoi_geojson: Dict[str, Any], conn=None
) -> List[Dict[str, Any]]:
    """Buildings intersecting the AOI, with boundary flags.

    Same eligibility predicate as ``postgis_query.buildings_in_aoi``
    (``ST_Intersects``), plus ``boundary_intersecting`` (footprint not
    entirely within the AOI) and ``aoi_overlap_fraction`` (share of the
    footprint area inside the AOI). Geometry is returned whole, never
    clipped.
    """
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH aoi AS (
                    SELECT ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326) AS g
                )
                SELECT b.building_id,
                       ST_AsGeoJSON(b.geometry),
                       NOT ST_Within(b.geometry, aoi.g),
                       CASE WHEN ST_Area(b.geometry) > 0
                            THEN ST_Area(ST_Intersection(b.geometry, aoi.g))
                                 / ST_Area(b.geometry)
                            ELSE 1.0 END,
                       b.footprint_version
                FROM hg_buildings b, aoi
                WHERE b.image_layer_id = %s
                  AND ST_Intersects(b.geometry, aoi.g)
                ORDER BY b.building_id
                """,
                (json.dumps(aoi_geojson), image_layer_id),
            )
            rows = cursor.fetchall()
    finally:
        if owns_conn:
            connection.close()
    return [
        {
            "building_id": r[0],
            "geometry": json.loads(r[1]),
            "boundary_intersecting": bool(r[2]),
            "aoi_overlap_fraction": float(min(1.0, max(0.0, r[3]))),
            "footprint_version": r[4],
        }
        for r in rows
    ]


def get_building_geometries(
    image_layer_id: str, building_ids: Iterable[str], conn=None
) -> Dict[str, Dict[str, Any]]:
    """``{building_id: GeoJSON geometry}`` for the given buildings."""
    ids = list(building_ids)
    if not ids:
        return {}
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT building_id, ST_AsGeoJSON(geometry)
                FROM hg_buildings
                WHERE image_layer_id = %s AND building_id = ANY(%s)
                """,
                (image_layer_id, ids),
            )
            return {r[0]: json.loads(r[1]) for r in cursor.fetchall()}
    finally:
        if owns_conn:
            connection.close()


release_expired_leases = expire_leases


def upsert_prediction_evidence(assessment_id_or_rows, evidence_rows=None, conn=None) -> int:
    """Upsert raw prediction evidence rows (idempotent).
    Supports both signatures:
      upsert_prediction_evidence(evidence_rows, conn=None)
      upsert_prediction_evidence(assessment_id, evidence_rows, conn=None)
    """
    if isinstance(assessment_id_or_rows, str):
        assessment_id = assessment_id_or_rows
        rows = evidence_rows or []
    else:
        rows = assessment_id_or_rows or []
        assessment_id = None
        if evidence_rows is not None and conn is None:
            conn = evidence_rows

    if not rows:
        return 0

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO hg_prediction_evidence (
                    assessment_id, building_id, unit_id,
                    probs, class, confidence, coverage_fraction,
                    valid_pixel_fraction, created_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, NOW()
                )
                ON CONFLICT (assessment_id, building_id, unit_id) DO UPDATE
                SET probs = EXCLUDED.probs,
                    class = EXCLUDED.class,
                    confidence = EXCLUDED.confidence,
                    coverage_fraction = EXCLUDED.coverage_fraction,
                    valid_pixel_fraction = EXCLUDED.valid_pixel_fraction
                """,
                [
                    (
                        r.get("assessment_id") or assessment_id,
                        r["building_id"],
                        r["unit_id"],
                        r["probs"],
                        r["class"],
                        r["confidence"],
                        r["coverage_fraction"],
                        r.get("valid_pixel_fraction"),
                    )
                    for r in rows
                ],
            )
            count = len(rows)
        connection.commit()
        return count
    finally:
        if owns_conn:
            connection.close()


def get_evidence(assessment_id: str, conn=None) -> List[Dict[str, Any]]:
    """Retrieve all raw prediction evidence rows for an assessment."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT building_id, unit_id, probs, class, confidence,
                       coverage_fraction, valid_pixel_fraction
                FROM hg_prediction_evidence
                WHERE assessment_id = %s
                """,
                (assessment_id,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "assessment_id": assessment_id,
                    "building_id": r[0],
                    "unit_id": r[1],
                    "probs": r[2],
                    "class": r[3],
                    "confidence": r[4],
                    "coverage_fraction": r[5],
                    "valid_pixel_fraction": r[6] if r[6] is not None else 1.0,
                }
                for r in rows
            ]
    finally:
        if owns_conn:
            connection.close()


get_prediction_evidence = get_evidence


def upsert_damage_assessments(assessment_id_or_results, results_or_conn=None, conn=None) -> int:
    """Upsert final aggregated damage results per building.
    Supports both signatures:
      upsert_damage_assessments(results, conn=None)
      upsert_damage_assessments(assessment_id, results, conn=None)
    """
    if isinstance(assessment_id_or_results, str):
        assessment_id = assessment_id_or_results
        results = results_or_conn or []
    else:
        results = assessment_id_or_results or []
        assessment_id = results[0].get("assessment_id") if results else ""
        if results_or_conn is not None and conn is None:
            conn = results_or_conn

    if not results:
        return 0

    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO hg_damage_assessments (
                    assessment_id, building_id, image_layer_id,
                    damage_class, damage_class_name, confidence,
                    damage_percentage, result_status,
                    boundary_intersecting, aoi_overlap_fraction,
                    winning_unit_id, model_digest, preprocessing_version, footprint_version,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    NULL, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    NOW(), NOW()
                )
                ON CONFLICT (assessment_id, building_id) DO UPDATE
                SET damage_class = EXCLUDED.damage_class,
                    damage_class_name = EXCLUDED.damage_class_name,
                    confidence = EXCLUDED.confidence,
                    damage_percentage = NULL,
                    result_status = EXCLUDED.result_status,
                    boundary_intersecting = EXCLUDED.boundary_intersecting,
                    aoi_overlap_fraction = EXCLUDED.aoi_overlap_fraction,
                    winning_unit_id = EXCLUDED.winning_unit_id,
                    model_digest = EXCLUDED.model_digest,
                    preprocessing_version = EXCLUDED.preprocessing_version,
                    footprint_version = EXCLUDED.footprint_version,
                    updated_at = NOW()
                """,
                [
                    (
                        r.get("assessment_id") or assessment_id,
                        r["building_id"],
                        r.get("image_layer_id"),
                        r.get("damage_class"),
                        r.get("damage_class_name"),
                        r.get("confidence"),
                        r.get("result_status", "completed"),
                        r.get("boundary_intersecting", False),
                        r.get("aoi_overlap_fraction", 1.0),
                        r.get("winning_unit_id"),
                        r.get("model_digest"),
                        r.get("preprocessing_version"),
                        r.get("footprint_version"),
                    )
                    for r in results
                ],
            )
            count = len(results)
        connection.commit()
        return count
    finally:
        if owns_conn:
            connection.close()


def get_assessment_statistics(assessment_id: str, conn=None) -> Dict[str, Any]:
    """Compute aggregate damage classification counts and status counts."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COALESCE(damage_class, -1) AS d_class,
                    COALESCE(damage_class_name, 'Unknown') AS d_name,
                    result_status,
                    COUNT(*)
                FROM hg_damage_assessments
                WHERE assessment_id = %s
                GROUP BY damage_class, damage_class_name, result_status
                """,
                (assessment_id,),
            )
            rows = cursor.fetchall()

            class_counts: Dict[str, int] = {}
            status_counts: Dict[str, int] = {}
            total = 0

            for d_class, d_name, status, count in rows:
                total += count
                status_counts[status] = status_counts.get(status, 0) + count
                if d_class != -1:
                    class_counts[d_name] = class_counts.get(d_name, 0) + count

            return {
                "assessment_id": assessment_id,
                "total_assessed_buildings": total,
                "class_counts": class_counts,
                "status_counts": status_counts,
            }
    finally:
        if owns_conn:
            connection.close()


def get_assessment_buildings_paged(
    assessment_id: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    page: int = 1,
    page_size: int = 50,
    conn=None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Retrieve paged building results with footprint geometry GeoJSON."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        offset = (page - 1) * page_size
        where_clauses = ["d.assessment_id = %s"]
        params: List[Any] = [assessment_id]

        if bbox:
            minx, miny, maxx, maxy = bbox
            where_clauses.append(
                "ST_Intersects(b.geometry, ST_MakeEnvelope(%s, %s, %s, %s, 4326))"
            )
            params.extend([minx, miny, maxx, maxy])

        where_sql = " AND ".join(where_clauses)

        with connection.cursor() as cursor:
            # Count total
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM hg_damage_assessments d
                JOIN hg_buildings b ON d.building_id = b.building_id AND d.image_layer_id = b.image_layer_id
                WHERE {where_sql}
                """,
                params,
            )
            total = cursor.fetchone()[0]

            # Fetch paged rows
            query_params = list(params) + [page_size, offset]
            cursor.execute(
                f"""
                SELECT
                    d.building_id, d.damage_class, d.damage_class_name, d.confidence,
                    d.result_status, d.boundary_intersecting, d.aoi_overlap_fraction,
                    ST_AsGeoJSON(b.geometry)
                FROM hg_damage_assessments d
                JOIN hg_buildings b ON d.building_id = b.building_id AND d.image_layer_id = b.image_layer_id
                WHERE {where_sql}
                ORDER BY d.building_id
                LIMIT %s OFFSET %s
                """,
                query_params,
            )
            rows = cursor.fetchall()
            features = [
                {
                    "building_id": r[0],
                    "damage_class": r[1],
                    "damage_class_name": r[2],
                    "confidence": r[3],
                    "result_status": r[4],
                    "boundary_intersecting": r[5],
                    "aoi_overlap_fraction": r[6],
                    "geometry": json.loads(r[7]),
                }
                for r in rows
            ]

        return total, features
    finally:
        if owns_conn:
            connection.close()


def get_assessment_map_data(
    assessment_id: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    zoom: int = 14,
    min_zoom: int = 14,
    max_features: int = 5000,
    conn=None,
) -> Dict[str, Any]:
    """Retrieve GeoJSON map data for viewport: vector buildings or grid aggregations."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            where_clauses = ["d.assessment_id = %s"]
            params: List[Any] = [assessment_id]

            if bbox:
                minx, miny, maxx, maxy = bbox
                where_clauses.append(
                    "ST_Intersects(b.geometry, ST_MakeEnvelope(%s, %s, %s, %s, 4326))"
                )
                params.extend([minx, miny, maxx, maxy])

            where_sql = " AND ".join(where_clauses)

            if zoom < min_zoom:
                # Low zoom: spatial grid clustering
                cursor.execute(
                    f"""
                    SELECT
                        ST_AsGeoJSON(ST_Centroid(ST_Collect(b.geometry))),
                        COUNT(*),
                        AVG(d.damage_class)
                    FROM hg_damage_assessments d
                    JOIN hg_buildings b ON d.building_id = b.building_id AND d.image_layer_id = b.image_layer_id
                    WHERE {where_sql}
                    GROUP BY ST_SnapToGrid(b.geometry, 0.01)
                    LIMIT %s
                    """,
                    params + [max_features],
                )
                clusters = cursor.fetchall()
                features = [
                    {
                        "type": "Feature",
                        "geometry": json.loads(c[0]),
                        "properties": {
                            "point_count": c[1],
                            "avg_damage_class": round(float(c[2]), 2) if c[2] is not None else None,
                        },
                    }
                    for c in clusters
                    if c[0]
                ]
                return {
                    "type": "FeatureCollection",
                    "mode": "aggregated",
                    "zoom": zoom,
                    "truncated": False,
                    "features": features,
                }
            else:
                # High zoom: detailed vector polygons
                cursor.execute(
                    f"""
                    SELECT
                        d.building_id, d.damage_class, d.damage_class_name, d.confidence,
                        d.result_status, d.boundary_intersecting, ST_AsGeoJSON(b.geometry)
                    FROM hg_damage_assessments d
                    JOIN hg_buildings b ON d.building_id = b.building_id AND d.image_layer_id = b.image_layer_id
                    WHERE {where_sql}
                    LIMIT %s
                    """,
                    params + [max_features + 1],
                )
                rows = cursor.fetchall()
                truncated = len(rows) > max_features
                if truncated:
                    rows = rows[:max_features]

                features = [
                    {
                        "type": "Feature",
                        "id": r[0],
                        "geometry": json.loads(r[6]),
                        "properties": {
                            "building_id": r[0],
                            "damage_class": r[1],
                            "damage_class_name": r[2],
                            "confidence": r[3],
                            "result_status": r[4],
                            "boundary_intersecting": r[5],
                        },
                    }
                    for r in rows
                ]
                return {
                    "type": "FeatureCollection",
                    "mode": "features",
                    "zoom": zoom,
                    "truncated": truncated,
                    "features": features,
                }
    finally:
        if owns_conn:
            connection.close()


def get_building_history(image_layer_id: str, building_id: str, conn=None) -> List[Dict[str, Any]]:
    """Retrieve historical assessment results for a specific building footprint."""
    owns_conn = conn is None
    connection = conn or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.assessment_id, a.name, a.created_at,
                    d.damage_class, d.damage_class_name, d.confidence,
                    d.result_status, d.winning_unit_id
                FROM hg_damage_assessments d
                JOIN hg_assessments a ON d.assessment_id = a.assessment_id
                WHERE d.image_layer_id = %s AND d.building_id = %s
                ORDER BY a.created_at DESC
                """,
                (image_layer_id, building_id),
            )
            rows = cursor.fetchall()
            return [
                {
                    "assessment_id": r[0],
                    "assessment_name": r[1],
                    "created_at": r[2].isoformat() if r[2] else None,
                    "damage_class": r[3],
                    "damage_class_name": r[4],
                    "confidence": r[5],
                    "result_status": r[6],
                    "winning_unit_id": r[7],
                }
                for r in rows
            ]
    finally:
        if owns_conn:
            connection.close()
