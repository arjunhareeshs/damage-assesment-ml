// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Components
import { Text } from "@fluentui/react";
import PropTypes from "prop-types";
import { DefaultButton } from "@fluentui/react";
import React from "react";
import StatusIndicator from "../OtherComponents/StatusIndicator";
import { useNavigate } from "react-router-dom";

const ImageLayerInfoMobile = ({ item }) => {
  ImageLayerInfoMobile.propTypes = {
    item: PropTypes.object.isRequired,
    // Callers still pass these; the buttons that used them (training,
    // embedding, the whole-layer DTF modal) were removed.
    setModalComponent: PropTypes.func,
    fetchProjectDetails: PropTypes.func,
    setComponentState: PropTypes.func,
    eventTypes: PropTypes.array,
  };

  const navigate = useNavigate();

  const dtfModels =
    (item.models || []).filter((m) => m.modelType === "dtf-aoi") || [];

  // DamageTriageFormer runs from the Labeling Tool as a Large-AOI assessment
  // (draw an area or use the whole image). The whole-layer and per-AOI DTF
  // workflows were removed.
  return (
    <React.Fragment key={"imageLayerInfoMobile_" + item.projectId + "_" + item.imageLayerId}>
      <tr>
        <td
          className="dashboard-table-for-inner-table-td custom-text-no-wrap"
        >
          <div className="pb-2">
            <Text
              variant="small"
              className="me-4 fw-semibold custom-text-color"
            >
              Layer Tools
            </Text>
          </div>

          <div style={{ borderBottom: "1px solid #ccc" }} className="pb-2">
            <StatusIndicator
              id={"singleProjectImageLayerStatus" + item.imageLayerId}
              currentStep={item.currentStep}
              totalSteps={item.totalSteps}
              progressPct={item.progressPct}
              status={item.status}
              statusMessage={item.statusMessage}
              prefix="Imagery"
            />
          </div>

          <>
              <div style={{ borderBottom: "1px solid #ccc" }} className="pb-2 pt-2">
                <DefaultButton
                  id={"singleProjectLabelingToolLaunch" + item.imageLayerId}
                  className="dashboard-button"
                  styles={{ root: { width: "100%" } }}
                  onClick={() =>
                    navigate(`/labeling-tool/${item.projectId}/${item.imageLayerId}`)
                  }
                  disabled={item.status !== "Processed"}
                >
                  Launch Labeling Tool
                </DefaultButton>
                <Text className="d-block pt-1" variant="small">
                  Label projects: {item.labelProjectCount}
                </Text>
              </div>

              <div style={{ borderBottom: "1px solid #ccc" }} className="pb-2 pt-2">
                {/* Model training was removed (pre-trained models only). */}
                <Text className="d-block pt-1" variant="small">
                  Models: {item.models && item.models.length > 0 ? item.models.length : 0}
                </Text>
              </div>
          </>

          <div className="pb-2 pt-2">
            <DefaultButton
              id={"singleProjectRunDTF" + item.imageLayerId}
              className="dashboard-button"
              styles={{ root: { width: "100%" } }}
              onClick={() =>
                navigate(`/labeling-tool/${item.projectId}/${item.imageLayerId}`)
              }
              disabled={
                item.status !== "Processed" ||
                !item.buildingFootprintsUrl ||
                Boolean(item.buildingFootprintsError)
              }
            >
              Run DamageTriageFormer
            </DefaultButton>
            <Text className="d-block pt-1" variant="small">
              DTF runs: {dtfModels.length}
            </Text>
          </div>
        </td>
      </tr>
    </React.Fragment >
  );
};

export default ImageLayerInfoMobile;
