// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { ActionButton, Spinner, SpinnerSize, Text } from "@fluentui/react";
import { FontIcon } from "@fluentui/react/lib/Icon";
import { apiGet } from "../../util/api";
import {
  DTF_CATEGORIES,
  DTF_CLASS_COLORS,
  RESULT_STATUS_COLORS,
} from "./LabelingToolHelper";

const BuildingAssessmentResultPanel = ({
  imageLayerId,
  buildingProperties,
  onClose,
}) => {
  BuildingAssessmentResultPanel.propTypes = {
    imageLayerId: PropTypes.string.isRequired,
    buildingProperties: PropTypes.object.isRequired,
    onClose: PropTypes.func.isRequired,
  };

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const buildingId =
    buildingProperties.building_id ||
    buildingProperties.id ||
    buildingProperties.buildingId;

  const damageClass = buildingProperties.damage_class;
  const damageClassName =
    buildingProperties.damage_class_name ||
    (damageClass != null && DTF_CATEGORIES[damageClass]) ||
    "Unclassified";
  const confidence = buildingProperties.confidence;
  const resultStatus = buildingProperties.result_status || "completed";
  const boundaryIntersecting = Boolean(buildingProperties.boundary_intersecting);

  useEffect(() => {
    let cancelled = false;
    async function fetchHistory() {
      if (!buildingId || !imageLayerId) return;
      setHistoryLoading(true);
      try {
        const res = await apiGet(
          `GetBuildingAssessmentHistory?imageLayerId=${imageLayerId}&buildingId=${encodeURIComponent(buildingId)}`
        );
        if (!cancelled && res && res.history) {
          setHistory(res.history);
        }
      } catch (err) {
        console.error("Error fetching building assessment history:", err);
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    }

    fetchHistory();
    return () => {
      cancelled = true;
    };
  }, [buildingId, imageLayerId]);

  const classColor =
    damageClass != null && damageClass >= 0 && damageClass < DTF_CLASS_COLORS.length
      ? DTF_CLASS_COLORS[damageClass]
      : "#9e9e9e";

  const statusColor = RESULT_STATUS_COLORS[resultStatus] || "#666";

  return (
    <div
      id="buildingAssessmentResultPanel"
      style={{
        position: "absolute",
        bottom: 20,
        left: 20,
        width: 320,
        backgroundColor: "#ffffff",
        padding: "14px",
        borderRadius: "8px",
        boxShadow: "0 4px 16px rgba(0,0,0,0.25)",
        zIndex: 1001,
      }}
    >
      <div className="d-flex justify-content-between align-items-center mb-2">
        <div>
          <Text variant="medium" className="fw-bold d-block">
            Building Assessment
          </Text>
          <span style={{ fontSize: 11, color: "#666", wordBreak: "break-all" }}>
            ID: {buildingId}
          </span>
        </div>
        <ActionButton
          iconProps={{ iconName: "Cancel" }}
          onClick={onClose}
          styles={{ root: { minWidth: 0, padding: 4 } }}
          title="Close panel"
        />
      </div>

      <div
        className="p-2 mb-2 rounded"
        style={{
          backgroundColor: "#f8f9fa",
          border: `2px solid ${classColor}`,
        }}
      >
        <div className="d-flex align-items-center justify-content-between">
          <span className="fw-bold" style={{ fontSize: 13, color: "#222" }}>
            {damageClassName}
          </span>
          <span
            style={{
              display: "inline-block",
              width: 14,
              height: 14,
              borderRadius: "50%",
              backgroundColor: classColor,
              border: "1px solid #777",
            }}
          />
        </div>

        <div className="mt-1 d-flex justify-content-between" style={{ fontSize: 12 }}>
          <span style={{ color: "#666" }}>Confidence:</span>
          <strong>
            {confidence != null ? `${(confidence * 100).toFixed(1)}%` : "N/A"}
          </strong>
        </div>

        <div className="d-flex justify-content-between" style={{ fontSize: 12 }}>
          <span style={{ color: "#666" }}>Result Status:</span>
          <span style={{ color: statusColor, fontWeight: 600 }}>{resultStatus}</span>
        </div>

        {boundaryIntersecting && (
          <div className="mt-1" style={{ fontSize: 11, color: "#e65100" }}>
            <FontIcon iconName="Warning" style={{ marginRight: 4 }} />
            Footprint intersects AOI boundary
          </div>
        )}
      </div>

      {/* History section */}
      <div className="border-top pt-2 mt-2">
        <div className="d-flex justify-content-between align-items-center mb-1">
          <span style={{ fontSize: 11, fontWeight: 600, color: "#666" }}>
            Historical Runs ({history.length})
          </span>
          {historyLoading && <Spinner size={SpinnerSize.xSmall} />}
        </div>

        {history.length > 0 ? (
          <div style={{ maxHeight: 120, overflowY: "auto" }}>
            {history.map((h, i) => (
              <div
                key={h.assessment_id || i}
                className="d-flex justify-content-between py-1 border-bottom"
                style={{ fontSize: 11 }}
              >
                <span style={{ color: "#555" }}>
                  {h.assessment_name || h.assessment_id.substring(0, 10)}
                </span>
                <span className="fw-semibold">
                  {h.damage_class_name || (h.damage_class != null ? DTF_CATEGORIES[h.damage_class] : h.result_status)}
                </span>
              </div>
            ))}
          </div>
        ) : !historyLoading ? (
          <span style={{ fontSize: 11, color: "#999" }}>No prior assessments recorded</span>
        ) : null}
      </div>
    </div>
  );
};

export default BuildingAssessmentResultPanel;
