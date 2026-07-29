// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modal to launch an AOI-scoped DamageTriageFormer inference job (Option
// B, AOI-scoped — docs/OptionB_AOI_Integration_Plan.md). Sibling of
// CreateEditDTFModal.jsx (the whole-layer path) — same shape, but POSTs
// the drawn AOI polygon and shows a pre-submit building-count preview
// (via GetBuildingsInAoi) so an analyst isn't surprised by how large a
// per-building-crop run they're about to queue.
import { useContext, useEffect, useState } from "react";
import { TextField } from "@fluentui/react";
import {
  DefaultButton,
  PrimaryButton,
} from "@fluentui/react/lib/Button";
import proptypes from "prop-types";

import { apiPut, apiPost } from "../../util/api";
import { AppContext } from "../../AppContext";
import SectionModal from "../SectionModal";

// Above this many buildings, warn before submit — per-building-crop
// inference is slower per-building than the whole-layer tile-batch path
// (docs/OptionB_AOI_Integration_Plan.md "Biggest pitfalls" #5).
const LARGE_AOI_WARNING_THRESHOLD = 5000;

const RunDtfAoiButton = ({
  onClose,
  projectId,
  imageLayerId,
  aoiGeoJson,
  selectedBuildingIds,
  fetchProjectDetails,
}) => {
  RunDtfAoiButton.propTypes = {
    onClose: proptypes.func.isRequired,
    projectId: proptypes.string.isRequired,
    imageLayerId: proptypes.string.isRequired,
    aoiGeoJson: proptypes.object,
    selectedBuildingIds: proptypes.array,
    fetchProjectDetails: proptypes.func,
  };

  const { setDialog, appParams, setIsLoading } = useContext(AppContext);
  const [state, setState] = useState({
    name: "dtf-aoi-" + Date.now(),
    nameError: "",
  });
  const [buildingCount, setBuildingCount] = useState(null);
  const [countError, setCountError] = useState("");

  const hasSelection =
    Boolean(aoiGeoJson) ||
    Boolean(selectedBuildingIds && selectedBuildingIds.length);

  useEffect(() => {
    let cancelled = false;
    async function fetchCount() {
      if (!hasSelection) {
        setBuildingCount(null);
        return;
      }
      try {
        const body = { imageLayerId };
        if (selectedBuildingIds && selectedBuildingIds.length) {
          body.buildingIds = selectedBuildingIds;
        } else {
          body.aoi = aoiGeoJson;
        }
        const result = await apiPost("GetBuildingsInAoi", body);
        if (!cancelled) {
          setBuildingCount(result.count);
          setCountError("");
        }
      } catch (error) {
        if (!cancelled) {
          setCountError("Could not preview building count.");
          console.error("Error counting buildings in AOI:", error);
        }
      }
    }
    fetchCount();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aoiGeoJson, selectedBuildingIds]);

  function onField(value, key) {
    setState((s) => ({ ...s, [key]: value }));
  }

  async function submit() {
    if (!hasSelection) return;

    setIsLoading(true, "Starting AOI-scoped DamageTriageFormer inference...");
    try {
      const body = {
        projectId,
        imageLayerId,
        modelType: "dtf-aoi",
        name: state.name,
        userId: appParams.userId,
      };
      if (selectedBuildingIds && selectedBuildingIds.length) {
        body.buildingIds = selectedBuildingIds;
      } else {
        body.aoi = aoiGeoJson;
      }
      await apiPut("PutRunDTFAoiInferenceQueueMessage", body);
      onClose();
      if (fetchProjectDetails) fetchProjectDetails();
      setDialog(
        "Success",
        "AOI-scoped DamageTriageFormer inference started. Buildings inside " +
          "the drawn area will be classified into 5 damage typology " +
          "classes and queued for review.",
        [
          {
            type: "primary",
            key: "close",
            text: "Close",
            onClick: () => setDialog(),
          },
        ]
      );
    } catch (error) {
      console.error("Error starting AOI-scoped DamageTriageFormer job:", error);
      setDialog("Error", "Failed to start the AOI-scoped DamageTriageFormer job.");
    }
    setIsLoading(false);
  }

  return (
    <SectionModal
      title="Run DamageTriageFormer on AOI"
      icon="MapLayers"
      onClose={onClose}
      body={
        <>
          <div className="row mb-2">
            <div className="col-12">
              <TextField
                id="createDTFAoiName"
                label="Name"
                value={state.name}
                onChange={(e, v) => onField(v, "name")}
                errorMessage={state.nameError}
              />
            </div>
          </div>
          <div className="row mb-2">
            <div className="col-12">
              <p style={{ fontSize: 12, color: "#666", margin: "8px 0" }}>
                Classifies only the buildings inside the drawn AOI (or the
                explicitly selected buildings) into 5 damage typology
                classes using the pretrained DamageTriageFormer model.
                Predictions are queued for review — draw an AOI on the map
                before running this.
              </p>
              {!hasSelection ? (
                <p style={{ fontSize: 12, color: "#a4262c", margin: "8px 0" }}>
                  No AOI drawn and no buildings selected yet.
                </p>
              ) : buildingCount === null ? (
                countError ? (
                  <p style={{ fontSize: 12, color: "#a4262c", margin: "8px 0" }}>
                    {countError}
                  </p>
                ) : (
                  <p style={{ fontSize: 12, color: "#666", margin: "8px 0" }}>
                    Counting buildings in scope...
                  </p>
                )
              ) : (
                <p
                  style={{
                    fontSize: 12,
                    color:
                      buildingCount > LARGE_AOI_WARNING_THRESHOLD
                        ? "#a4262c"
                        : "#666",
                    margin: "8px 0",
                  }}
                >
                  {buildingCount} building{buildingCount === 1 ? "" : "s"} in
                  scope.
                  {buildingCount > LARGE_AOI_WARNING_THRESHOLD
                    ? " This is a large AOI — per-building inference will " +
                      "take a while. Consider drawing a smaller area."
                    : ""}
                </p>
              )}
            </div>
          </div>
          <div className="row">
            <div className="col-12 d-flex justify-content-end">
              <PrimaryButton
                className="me-2"
                onClick={submit}
                disabled={!hasSelection}
                id="createDTFAoiSubmit"
              >
                Run
              </PrimaryButton>
              <DefaultButton onClick={onClose}>Cancel</DefaultButton>
            </div>
          </div>
        </>
      }
    />
  );
};

export default RunDtfAoiButton;
