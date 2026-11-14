// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { useContext, useEffect, useState } from "react";
import PropTypes from "prop-types";
import { TextField, Spinner, SpinnerSize } from "@fluentui/react";
import { DefaultButton, PrimaryButton } from "@fluentui/react/lib/Button";
import { FontIcon } from "@fluentui/react/lib/Icon";
import { apiPost, apiPut } from "../../util/api";
import { AppContext } from "../../AppContext";
import SectionModal from "../SectionModal";

const STATE_STYLE = {
  ready: { text: "AOI ready", color: "#107c10" },
  incomplete_imagery: { text: "Incomplete imagery coverage", color: "#bf6a00" },
  no_buildings: { text: "No buildings found in this AOI", color: "#a4262c" },
  no_imagery: { text: "No usable imagery found", color: "#a4262c" },
};

const AoiAssessmentModal = ({
  onClose,
  projectId,
  imageLayerId,
  aoiGeoJson,
  onAssessmentCreated,
}) => {
  AoiAssessmentModal.propTypes = {
    onClose: PropTypes.func.isRequired,
    projectId: PropTypes.string.isRequired,
    imageLayerId: PropTypes.string.isRequired,
    aoiGeoJson: PropTypes.object,
    onAssessmentCreated: PropTypes.func.isRequired,
  };

  const { setDialog, setIsLoading } = useContext(AppContext);
  const [name, setName] = useState(() => "aoi-assessment-" + Date.now());
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function fetchPreview() {
      if (!aoiGeoJson) {
        setPreview(null);
        return;
      }
      setPreviewLoading(true);
      setPreviewError("");
      try {
        const result = await apiPost("GetAoiAssessmentPreview", {
          imageLayerId,
          aoi: aoiGeoJson,
        });
        if (!cancelled) {
          setPreview(result);
        }
      } catch (err) {
        if (!cancelled) {
          setPreviewError(err?.message || "Failed to preview AOI.");
          console.error("Error previewing AOI:", err);
        }
      } finally {
        if (!cancelled) {
          setPreviewLoading(false);
        }
      }
    }

    fetchPreview();
    return () => {
      cancelled = true;
    };
  }, [aoiGeoJson, imageLayerId]);

  async function handleSubmit() {
    if (!aoiGeoJson) return;
    setIsLoading(true, "Submitting Large-AOI assessment...");
    try {
      const body = {
        projectId,
        imageLayerId,
        aoi: aoiGeoJson,
        name: name.trim() || undefined,
      };
      const result = await apiPut("PutAoiAssessment", body);
      if (result === 409) {
        setDialog(
          "Already running",
          "An assessment for this exact area is already in progress."
        );
        return;
      }
      onClose();
      if (onAssessmentCreated) {
        onAssessmentCreated(result.assessmentId);
      }
      setDialog(
        "Assessment Submitted",
        `Assessment ${result.assessmentId} has been successfully submitted and queued for batched inference.`
      );
    } catch (err) {
      console.error("Error submitting assessment:", err);
      setDialog("Submission Failed", err?.message || "Could not submit assessment.");
    } finally {
      setIsLoading(false);
    }
  }

  const cannotSubmit = preview && preview.canSubmit === false;

  return (
    <SectionModal
      title="Large-AOI Building Damage Assessment"
      icon="MapLayers"
      onClose={onClose}
      body={
        <>
          <div className="row mb-3">
            <div className="col-12">
              <TextField
                id="aoiAssessmentName"
                label="Assessment Name"
                value={name}
                onChange={(e, v) => setName(v || "")}
              />
            </div>
          </div>

          <div className="row mb-3">
            <div className="col-12">
              <div
                style={{
                  backgroundColor: "#f4f6f8",
                  padding: "12px",
                  borderRadius: "6px",
                  border: "1px solid #e1e4e8",
                }}
              >
                <div className="d-flex align-items-center mb-2">
                  <FontIcon iconName="View" style={{ marginRight: 6, color: "#0078d4" }} />
                  <strong>Scope Preview</strong>
                </div>

                {!aoiGeoJson ? (
                  <p style={{ color: "#a4262c", margin: 0, fontSize: 13 }}>
                    No AOI drawn yet. Please draw an AOI polygon or rectangle first.
                  </p>
                ) : previewLoading ? (
                  <div className="d-flex align-items-center gap-2 py-2">
                    <Spinner size={SpinnerSize.small} />
                    <span style={{ fontSize: 13, color: "#666" }}>
                      Inspecting imagery coverage and building footprints...
                    </span>
                  </div>
                ) : previewError ? (
                  <p style={{ color: "#a4262c", margin: 0, fontSize: 13 }}>
                    {previewError}
                  </p>
                ) : preview ? (
                  <div className="d-flex flex-column gap-1" style={{ fontSize: 13 }}>
                    <div
                      style={{
                        color: STATE_STYLE[preview.state]?.color || "#333",
                        fontWeight: 600,
                      }}
                    >
                      {STATE_STYLE[preview.state]?.text || "Assessment unavailable"}
                    </div>
                    <div>
                      <strong>AOI Area:</strong> {preview.areaKm2} km²
                    </div>
                    <div>
                      <strong>Imagery:</strong> {preview.imageryCount} raster
                      {preview.imageryCount === 1 ? "" : "s"} (
                      {preview.fullyCoveredTiles ?? 0} full,{" "}
                      {preview.partiallyCoveredTiles ?? 0} partial)
                    </div>
                    <div>
                      <strong>Imagery coverage:</strong>{" "}
                      {preview.coveragePercent ?? 0}%
                      {preview.missingAreaKm2 > 0
                        ? ` (${preview.missingAreaKm2} km² has no imagery and will not be assessed)`
                        : ""}
                    </div>
                    <div>
                      <strong>Buildings:</strong>{" "}
                      {preview.buildingCount.toLocaleString()} footprints
                    </div>
                    <div>
                      <strong>Processing units (est. max):</strong>{" "}
                      {(preview.estimatedUnits ?? 0).toLocaleString()}
                    </div>
                    {preview.exceedsMaxUnits && (
                      <p style={{ color: "#a4262c", margin: "6px 0 0 0", fontSize: 12 }}>
                        This AOI needs more than the allowed {preview.maxUnits} units.
                        Draw a smaller area.
                      </p>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <p style={{ fontSize: 12, color: "#666", marginBottom: 16 }}>
            The assessment will run in parallel 1024-px tile batches across all intersecting TIFs
            using DamageTriageFormer. One persistent classification result per building will be stored.
          </p>

          <div className="row">
            <div className="col-12 d-flex justify-content-end">
              <PrimaryButton
                id="submitAoiAssessment"
                className="me-2"
                onClick={handleSubmit}
                disabled={!aoiGeoJson || previewLoading || !!previewError || cannotSubmit}
              >
                Submit Assessment
              </PrimaryButton>
              <DefaultButton onClick={onClose}>Cancel</DefaultButton>
            </div>
          </div>
        </>
      }
    />
  );
};

export default AoiAssessmentModal;
