// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import {
  ActionButton,
  DefaultButton,
  PrimaryButton,
} from "@fluentui/react";
import DrawingToolbar from "./DrawingToolbar";
import AoiDrawTool from "./AoiDrawTool";
import RunDtfAoiButton from "./RunDtfAoiButton";
import AoiAssessmentModal from "./AoiAssessmentModal";
import PropType from "prop-types";
import { useEffect, useState } from "react";
import { apiPost } from "../../util/api";

const LabelingToolRightPanel = ({
  drawingManager,
  setDrawingManager,
  setModalComponent,
  projectId,
  imageLayerId,
  aoiGeoJson,
  setAoiGeoJson,
  buildingSelectActive,
  setBuildingSelectActive,
  selectedBuildingIds,
  setSelectedBuildingIds,
  assessmentId,
  onAssessmentCreated,
  setShowAssessmentDashboard,
  getStudyArea,
  onFitImagery,
  layerBuildings,
}) => {
  LabelingToolRightPanel.propTypes = {
    drawingManager: PropType.object.isRequired,
    setDrawingManager: PropType.func.isRequired,
    setModalComponent: PropType.func.isRequired,
    projectId: PropType.string.isRequired,
    imageLayerId: PropType.string.isRequired,
    aoiGeoJson: PropType.object,
    setAoiGeoJson: PropType.func.isRequired,
    buildingSelectActive: PropType.bool.isRequired,
    setBuildingSelectActive: PropType.func.isRequired,
    selectedBuildingIds: PropType.array.isRequired,
    setSelectedBuildingIds: PropType.func.isRequired,
    assessmentId: PropType.string,
    onAssessmentCreated: PropType.func,
    setShowAssessmentDashboard: PropType.func,
    getStudyArea: PropType.func,
    onFitImagery: PropType.func,
    layerBuildings: PropType.object,
  };

  // How many footprints fall inside the drawn area, shown right under it so
  // the analyst sees what a run will cover before starting one.
  const [aoiCount, setAoiCount] = useState({ state: "idle", count: 0 });
  useEffect(() => {
    if (!aoiGeoJson) {
      setAoiCount({ state: "idle", count: 0 });
      return undefined;
    }
    let cancelled = false;
    setAoiCount({ state: "loading", count: 0 });
    apiPost("GetBuildingsInAoi", { imageLayerId, aoi: aoiGeoJson })
      .then((result) => {
        if (!cancelled) setAoiCount({ state: "done", count: result.count || 0 });
      })
      .catch((error) => {
        console.error("Error counting buildings in the area:", error);
        if (!cancelled) setAoiCount({ state: "error", count: 0 });
      });
    return () => {
      cancelled = true;
    };
  }, [aoiGeoJson, imageLayerId]);

  function handleRunDtfAoi() {
    setModalComponent(
      <RunDtfAoiButton
        onClose={() => setModalComponent(null)}
        projectId={projectId}
        imageLayerId={imageLayerId}
        aoiGeoJson={aoiGeoJson}
        selectedBuildingIds={selectedBuildingIds}
        fetchProjectDetails={() => {}}
      />
    );
  }

  function handleRunLargeAoiAssessment() {
    setModalComponent(
      <AoiAssessmentModal
        onClose={() => setModalComponent(null)}
        projectId={projectId}
        imageLayerId={imageLayerId}
        aoiGeoJson={aoiGeoJson}
        onAssessmentCreated={(newAssessmentId) => {
          if (onAssessmentCreated) {
            onAssessmentCreated(newAssessmentId);
          }
        }}
      />
    );
  }

  const nothingToRun = !aoiGeoJson && selectedBuildingIds.length === 0;

  return (
    <>
      <DrawingToolbar
        drawingManager={drawingManager}
        setDrawingManager={setDrawingManager}
        onDrawingStart={() => setBuildingSelectActive(false)}
        onFitImagery={onFitImagery}
      />
      <div
        style={{
          position: "absolute",
          right: 10,
          top: 70,
          width: 260,
          backgroundColor: "rgba(255, 255, 255, 1)",
          padding: "10px",
          borderRadius: "5px",
          zIndex: 1000,
        }}
        id="rightPanel"
      >
        <div className="col-12 d-flex flex-column">
          <div id="layerBuildingsInfo" style={{ fontSize: 12, color: "#333", marginBottom: 6 }}>
            {!layerBuildings ? (
              <span style={{ color: "#666" }}>Loading building footprints...</span>
            ) : layerBuildings.error ? (
              <span style={{ color: "#a4262c" }}>Could not load the building footprints.</span>
            ) : layerBuildings.total === 0 ? (
              <span style={{ color: "#a4262c" }}>
                This layer has no building footprints, so there is nothing to
                assess. Re-create it with a footprints file.
              </span>
            ) : (
              <>
                <span className="fw-semibold">
                  {layerBuildings.total.toLocaleString()} buildings
                </span>{" "}
                in this image
                {layerBuildings.truncated
                  ? ` (showing the first ${layerBuildings.shown.toLocaleString()})`
                  : ""}
                <div className="d-flex gap-3 mt-1" style={{ color: "#555" }}>
                  <span><span style={{ border: "2px solid #888", display: "inline-block", width: 10, height: 10, marginRight: 4 }} />building</span>
                  <span><span style={{ background: "#ffb900", display: "inline-block", width: 10, height: 10, marginRight: 4 }} />in area</span>
                  <span><span style={{ background: "#0078d4", display: "inline-block", width: 10, height: 10, marginRight: 4 }} />selected</span>
                </div>
              </>
            )}
          </div>
          <AoiDrawTool
            aoiGeoJson={aoiGeoJson}
            setAoiGeoJson={setAoiGeoJson}
            getStudyArea={getStudyArea}
          />
          {aoiGeoJson ? (
            <div id="aoiBuildingCount" style={{ fontSize: 13, marginTop: 2 }}>
              {aoiCount.state === "loading" ? (
                <span style={{ color: "#666" }}>Counting buildings in this area...</span>
              ) : aoiCount.state === "error" ? (
                <span style={{ color: "#a4262c" }}>Could not count the buildings in this area.</span>
              ) : aoiCount.state === "done" ? (
                <>
                  <span className="fw-semibold">
                    {aoiCount.count.toLocaleString()} building
                    {aoiCount.count === 1 ? "" : "s"}
                  </span>{" "}
                  in this area
                  {aoiCount.count === 0 ? (
                    <div style={{ fontSize: 12, color: "#666" }}>
                      No building footprints here. Draw over buildings, or
                      check the layer&apos;s footprints.
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>
          ) : null}

          <DefaultButton
            className="w-100 mt-2"
            text={
              buildingSelectActive
                ? "Stop selecting buildings"
                : "Select buildings"
            }
            iconProps={{ iconName: "TouchPointer" }}
            checked={buildingSelectActive}
            onClick={() => {
              if (!buildingSelectActive && drawingManager) {
                drawingManager.isAoiDrawing = false;
                drawingManager.setOptions({ mode: "none" });
              }
              setBuildingSelectActive((prev) => !prev);
            }}
          />
          <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
            {buildingSelectActive
              ? "Click a building to select it, click again to deselect."
              : "Touch individual buildings to pick them one by one."}
          </div>
          {selectedBuildingIds.length > 0 ? (
            <div className="d-flex align-items-center justify-content-between mt-1">
              <span style={{ fontSize: 12, color: "#333" }}>
                {selectedBuildingIds.length} building
                {selectedBuildingIds.length === 1 ? "" : "s"} selected
              </span>
              <ActionButton
                iconProps={{ iconName: "Cancel" }}
                onClick={() => setSelectedBuildingIds([])}
                styles={{ root: { height: "auto" } }}
              >
                Clear
              </ActionButton>
            </div>
          ) : null}
        </div>

        <div className="col-12 d-flex flex-column pt-2 mt-2 border-top">
          <PrimaryButton
            id="runDtfAoiButton"
            className="w-100"
            text="Run DamageTriageFormer"
            disabled={nothingToRun}
            title={
              nothingToRun
                ? "Draw an area or select buildings first"
                : "Classify the selected buildings"
            }
            onClick={handleRunDtfAoi}
          />

          <PrimaryButton
            id="runLargeAoiAssessmentButton"
            className="w-100 mt-2"
            text="Run Large-AOI Assessment"
            disabled={!aoiGeoJson}
            title={
              !aoiGeoJson
                ? "Draw a polygon or rectangle first"
                : "Launch batched large-AOI assessment"
            }
            onClick={handleRunLargeAoiAssessment}
          />

          {assessmentId ? (
            <DefaultButton
              id="viewAssessmentDashboardButton"
              className="w-100 mt-2"
              text="View Active Assessment"
              iconProps={{ iconName: "BIDashboard" }}
              onClick={() =>
                setShowAssessmentDashboard && setShowAssessmentDashboard(true)
              }
            />
          ) : null}
        </div>
      </div>
    </>
  );
};

export default LabelingToolRightPanel;
