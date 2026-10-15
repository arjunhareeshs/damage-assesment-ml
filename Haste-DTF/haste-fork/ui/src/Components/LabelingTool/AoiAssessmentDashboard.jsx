// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
import {
  ActionButton,
  DefaultButton,
  ProgressIndicator,
  Spinner,
  SpinnerSize,
  Text,
} from "@fluentui/react";
import { apiGet, apiPut } from "../../util/api";
import { DTF_CATEGORIES, DTF_CLASS_COLORS } from "./LabelingToolHelper";

// Poll intervals (ms) while an assessment is running / after a failed poll.
const POLL_INTERVAL_MS = 3000;
const POLL_RETRY_MS = 5000;

const TERMINAL_STATUSES = ["completed", "completed_with_errors", "failed", "cancelled", "rejected"];

const AoiAssessmentDashboard = ({ assessmentId, onClose, onRefreshMap }) => {
  AoiAssessmentDashboard.propTypes = {
    assessmentId: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    onRefreshMap: PropTypes.func,
  };

  const [assessment, setAssessment] = useState(null);
  const [stats, setStats] = useState(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [error, setError] = useState("");
  // The parent passes a new callback each render; keep it out of the
  // polling effect's dependencies, or every refresh restarts the poll
  // immediately (a request loop instead of a timed poll).
  const onRefreshMapRef = useRef(onRefreshMap);
  useEffect(() => {
    onRefreshMapRef.current = onRefreshMap;
  }, [onRefreshMap]);

  useEffect(() => {
    let timer = null;
    let cancelled = false;
    let lastProgressKey = null;

    async function fetchAssessmentData() {
      try {
        const data = await apiGet(`GetAoiAssessment?assessmentId=${assessmentId}`);
        if (cancelled) return;
        setAssessment(data);
        setError("");

        // If completed or has progress, fetch statistics
        if (data.status === "completed" || data.status === "completed_with_errors" || data.completedUnits > 0) {
          try {
            const statsData = await apiGet(`GetAoiAssessmentStatistics?assessmentId=${assessmentId}`);
            if (!cancelled) setStats(statsData);
          } catch {
            // Stats optional during progress
          }
        }

        // Refresh the map only when stored results may have changed.
        const progressKey = `${data.status}:${data.completedUnits}:${data.failedUnits}`;
        if (progressKey !== lastProgressKey) {
          lastProgressKey = progressKey;
          if (onRefreshMapRef.current) onRefreshMapRef.current();
        }

        // Keep polling if not in a terminal state
        if (!TERMINAL_STATUSES.includes(data.status)) {
          timer = setTimeout(fetchAssessmentData, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err?.message || "Failed to load assessment.");
          console.error("Error polling assessment:", err);
          timer = setTimeout(fetchAssessmentData, POLL_RETRY_MS);
        }
      }
    }

    fetchAssessmentData();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [assessmentId]);

  async function handleCancel() {
    setIsCancelling(true);
    try {
      await apiPut("PutCancelAoiAssessment", { assessmentId });
      setAssessment((prev) => (prev ? { ...prev, status: "cancelling" } : prev));
    } catch (err) {
      console.error("Error cancelling assessment:", err);
    } finally {
      setIsCancelling(false);
    }
  }

  if (error && !assessment) {
    return (
      <div
        id="aoiAssessmentDashboard"
        style={{
          position: "absolute",
          top: 10,
          left: 60,
          width: 340,
          backgroundColor: "#ffffff",
          padding: "12px",
          borderRadius: "6px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
          zIndex: 1000,
        }}
      >
        <div className="d-flex justify-content-between align-items-center">
          <Text variant="medium" className="fw-bold" style={{ color: "#a4262c" }}>
            Assessment Error
          </Text>
          <ActionButton iconProps={{ iconName: "Cancel" }} onClick={onClose} />
        </div>
        <p style={{ fontSize: 12, color: "#666", margin: "6px 0" }}>{error}</p>
      </div>
    );
  }

  if (!assessment) {
    return (
      <div
        id="aoiAssessmentDashboard"
        style={{
          position: "absolute",
          top: 10,
          left: 60,
          width: 320,
          backgroundColor: "#ffffff",
          padding: "12px",
          borderRadius: "6px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
          zIndex: 1000,
        }}
      >
        <div className="d-flex align-items-center gap-2">
          <Spinner size={SpinnerSize.small} />
          <Text variant="small">Loading assessment {assessmentId}...</Text>
        </div>
      </div>
    );
  }

  const isTerminal = TERMINAL_STATUSES.includes(assessment.status);
  const totalUnits = assessment.totalUnits || 0;
  const completedUnits = assessment.completedUnits || 0;
  const failedUnits = assessment.failedUnits || 0;
  const progressPercent = totalUnits > 0 ? (completedUnits / totalUnits) : 0;

  const getStatusBadgeStyle = (status) => {
    switch (status) {
      case "completed":
        return { bg: "#e8f5e9", color: "#2e7d32", label: "Completed" };
      case "completed_with_errors":
        return { bg: "#fff3e0", color: "#e65100", label: "Completed with Errors" };
      case "running":
        return { bg: "#e3f2fd", color: "#1565c0", label: "Running" };
      case "aggregating":
        return { bg: "#e0f7fa", color: "#006064", label: "Aggregating" };
      case "tiling":
        return { bg: "#ede7f6", color: "#4a148c", label: "Tiling Windows" };
      case "discovering":
        return { bg: "#e8eaf6", color: "#283593", label: "Discovering" };
      case "queued":
      case "submitted":
        return { bg: "#f3e5f5", color: "#6a1b9a", label: "Queued" };
      case "cancelling":
      case "cancelled":
        return { bg: "#f5f5f5", color: "#616161", label: "Cancelled" };
      case "failed":
        return { bg: "#ffebee", color: "#c62828", label: "Failed" };
      default:
        return { bg: "#f5f5f5", color: "#333", label: status };
    }
  };

  const badge = getStatusBadgeStyle(assessment.status);

  return (
    <div
      id="aoiAssessmentDashboard"
      style={{
        position: "absolute",
        top: 10,
        left: 60,
        width: 360,
        maxHeight: "85vh",
        overflowY: "auto",
        backgroundColor: "#ffffff",
        padding: "14px",
        borderRadius: "8px",
        boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
        zIndex: 1000,
      }}
    >
      <div className="d-flex justify-content-between align-items-center mb-2">
        <div>
          <Text variant="mediumPlus" className="fw-bold">
            {assessment.name || `Assessment ${assessmentId}`}
          </Text>
          <div style={{ fontSize: 11, color: "#888" }}>ID: {assessmentId}</div>
        </div>
        <ActionButton
          iconProps={{ iconName: "Cancel" }}
          onClick={onClose}
          styles={{ root: { minWidth: 0, padding: 4 } }}
          title="Close panel"
        />
      </div>

      <div className="d-flex align-items-center gap-2 mb-3">
        <span
          style={{
            backgroundColor: badge.bg,
            color: badge.color,
            fontSize: 12,
            fontWeight: 600,
            padding: "3px 8px",
            borderRadius: "12px",
          }}
        >
          {badge.label}
        </span>
        {!isTerminal && <Spinner size={SpinnerSize.xSmall} />}
      </div>

      {/* Progress */}
      {totalUnits > 0 && (
        <div className="mb-3">
          <div className="d-flex justify-content-between" style={{ fontSize: 12, marginBottom: 4 }}>
            <span>Tile Batches:</span>
            <span>
              {completedUnits} / {totalUnits} ({Math.round(progressPercent * 100)}%)
              {failedUnits > 0 ? ` [${failedUnits} failed]` : ""}
            </span>
          </div>
          <ProgressIndicator percentComplete={progressPercent} />
        </div>
      )}

      {/* Summary info */}
      <div
        style={{
          backgroundColor: "#f8f9fa",
          padding: "8px 10px",
          borderRadius: "6px",
          fontSize: 12,
          marginBottom: 12,
        }}
      >
        <div className="d-flex justify-content-between py-1 border-bottom">
          <span style={{ color: "#666" }}>Eligible Buildings:</span>
          <strong>{assessment.eligibleBuildings != null ? assessment.eligibleBuildings.toLocaleString() : "Counting..."}</strong>
        </div>
        <div className="d-flex justify-content-between py-1 border-bottom">
          <span style={{ color: "#666" }}>Model Digest:</span>
          <span style={{ fontFamily: "monospace", fontSize: 11 }}>
            {assessment.versions?.modelDigest ? assessment.versions.modelDigest.substring(0, 10) + "..." : "Pending"}
          </span>
        </div>
        <div className="d-flex justify-content-between py-1">
          <span style={{ color: "#666" }}>Preprocessing:</span>
          <span>{assessment.versions?.preprocessingVersion || "1.0.0"}</span>
        </div>
      </div>

      {/* Statistics breakdown if available */}
      {stats && stats.class_counts && (
        <div className="mb-3">
          <Text variant="small" className="fw-bold mb-2 d-block">
            Damage Classification Summary
          </Text>
          <div className="d-flex flex-column gap-1">
            {DTF_CATEGORIES.map((catName, idx) => {
              const count = stats.class_counts[catName] || 0;
              const color = DTF_CLASS_COLORS[idx];
              return (
                <div
                  key={catName}
                  className="d-flex align-items-center justify-content-between"
                  style={{ fontSize: 12 }}
                >
                  <div className="d-flex align-items-center gap-2">
                    <span
                      style={{
                        display: "inline-block",
                        width: 12,
                        height: 12,
                        borderRadius: 2,
                        backgroundColor: color,
                        border: "1px solid #999",
                      }}
                    />
                    <span>{catName}</span>
                  </div>
                  <strong>{count.toLocaleString()}</strong>
                </div>
              );
            })}
          </div>

          {stats.status_counts && (stats.status_counts.insufficient_data || stats.status_counts.failed) ? (
            <div className="mt-2 pt-2 border-top" style={{ fontSize: 11, color: "#666" }}>
              {stats.status_counts.insufficient_data ? (
                <div>Insufficient data (nodata): {stats.status_counts.insufficient_data}</div>
              ) : null}
              {stats.status_counts.failed ? (
                <div>Inference failed: {stats.status_counts.failed}</div>
              ) : null}
            </div>
          ) : null}
        </div>
      )}

      {/* Action buttons */}
      <div className="d-flex justify-content-between align-items-center pt-2 border-top">
        {!isTerminal ? (
          <DefaultButton
            text="Cancel Assessment"
            iconProps={{ iconName: "Stop" }}
            onClick={handleCancel}
            disabled={isCancelling || assessment.status === "cancelling"}
            styles={{ root: { color: "#d32f2f" } }}
          />
        ) : (
          <span style={{ fontSize: 11, color: "#888" }}>
            {assessment.completedAt ? `Finished at ${new Date(assessment.completedAt).toLocaleTimeString()}` : "Completed"}
          </span>
        )}
        <DefaultButton text="Close" onClick={onClose} />
      </div>
    </div>
  );
};

export default AoiAssessmentDashboard;
