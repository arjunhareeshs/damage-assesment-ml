// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AOI-scoped DamageTriageFormer (Option B, AOI-scoped —
// docs/OptionB_AOI_Integration_Plan.md Module 6.7). Shown when an analyst
// clicks a building on the Visualizer's AOI-predictions vector layer
// (Visualizer.jsx's dtfAoiPredictionsLayer). Displays the evidence behind
// one prediction — Building ID, original crop, damage mask, class,
// confidence — and lets the analyst Accept or Reject it, exactly as
// plan.pdf's Module 6.7 specifies: "This gives the reviewer both the AI
// result and the underlying evidence."
import { useContext, useState } from "react";
import { ActionButton, DefaultButton, PrimaryButton, Text } from "@fluentui/react";
import PropTypes from "prop-types";

import { apiGet, apiPut } from "../../util/api";
import { AppContext } from "../../AppContext";
import { fileDownload } from "../../util/file";

const DtfAoiReviewPanel = ({ imageLayerId, building, onClose, onReviewed }) => {
  DtfAoiReviewPanel.propTypes = {
    imageLayerId: PropTypes.string.isRequired,
    // Clicked shape's properties, as written by
    // postgis_query.get_classified_buildings_geojson: { id, dtf_class,
    // dtf_class_name, dtf_confidence, mask_url, crop_url, review_status }
    building: PropTypes.object.isRequired,
    onClose: PropTypes.func.isRequired,
    // (buildingId, newReviewStatus) => void — lets the caller update the
    // layer's paint expression without a full re-fetch.
    onReviewed: PropTypes.func.isRequired,
  };

  const { setDialog } = useContext(AppContext);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [reviewStatus, setReviewStatus] = useState(building.review_status);

  async function handleExport() {
    if (!building.model_id) return;
    setIsExporting(true);
    try {
      const result = await apiGet(
        `GetAoiPredictionsGpkg?imageLayerId=${imageLayerId}&modelId=${building.model_id}`
      );
      fileDownload(result.url, setDialog);
    } catch (error) {
      console.error("Error exporting AOI predictions:", error);
      setDialog("Error", "Failed to export AOI predictions.");
    }
    setIsExporting(false);
  }

  async function handleReview(newStatus) {
    setIsSubmitting(true);
    try {
      await apiPut("PutBuildingDTFReview", {
        imageLayerId,
        buildingId: building.id,
        reviewStatus: newStatus,
      });
      setReviewStatus(newStatus);
      onReviewed(building.id, newStatus);
    } catch (error) {
      console.error("Error updating building review status:", error);
      setDialog("Error", "Failed to update the review status.");
    }
    setIsSubmitting(false);
  }

  return (
    <div
      style={{
        position: "absolute",
        left: 10,
        top: 10,
        width: 280,
        backgroundColor: "rgba(255, 255, 255, 1)",
        padding: "12px",
        borderRadius: "5px",
        zIndex: 1001,
        boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
      }}
      id="dtfAoiReviewPanel"
    >
      <div className="d-flex justify-content-between align-items-center">
        <Text variant="mediumPlus" className="fw-semibold">
          Building {building.id}
        </Text>
        <ActionButton
          iconProps={{ iconName: "Cancel" }}
          onClick={onClose}
          styles={{ root: { minWidth: 0, padding: 0 } }}
        />
      </div>

      <div className="d-flex gap-2 mt-2">
        {building.crop_url ? (
          <div style={{ flex: 1 }}>
            <Text variant="small">Original crop</Text>
            <img
              src={building.crop_url}
              alt={`Building ${building.id} crop`}
              style={{ width: "100%", borderRadius: 4 }}
            />
          </div>
        ) : null}
        {building.mask_url ? (
          <div style={{ flex: 1 }}>
            <Text variant="small">Damage mask</Text>
            <img
              src={building.mask_url}
              alt={`Building ${building.id} damage mask`}
              style={{ width: "100%", borderRadius: 4 }}
            />
          </div>
        ) : null}
      </div>

      <div className="mt-2">
        <Text variant="small">
          <b>Damage class:</b> {building.dtf_class_name}
        </Text>
        <br />
        <Text variant="small">
          <b>Confidence:</b>{" "}
          {typeof building.dtf_confidence === "number"
            ? `${Math.round(building.dtf_confidence * 100)}%`
            : "-"}
        </Text>
        <br />
        <Text variant="small">
          <b>Review status:</b> {reviewStatus}
        </Text>
      </div>

      <div className="d-flex gap-2 mt-3">
        <PrimaryButton
          className="w-100"
          disabled={isSubmitting || reviewStatus === "accepted"}
          onClick={() => handleReview("accepted")}
          iconProps={{ iconName: "Accept" }}
        >
          Accept
        </PrimaryButton>
        <DefaultButton
          className="w-100"
          disabled={isSubmitting || reviewStatus === "rejected"}
          onClick={() => handleReview("rejected")}
          iconProps={{ iconName: "Cancel" }}
        >
          Reject
        </DefaultButton>
      </div>

      <DefaultButton
        className="w-100 mt-2"
        disabled={isExporting || !building.model_id}
        onClick={handleExport}
        iconProps={{ iconName: "Download" }}
        title="Export this AOI run's current predictions (excludes rejected buildings)"
      >
        {isExporting ? "Exporting..." : "Export predictions (.gpkg)"}
      </DefaultButton>
    </div>
  );
};

export default DtfAoiReviewPanel;
