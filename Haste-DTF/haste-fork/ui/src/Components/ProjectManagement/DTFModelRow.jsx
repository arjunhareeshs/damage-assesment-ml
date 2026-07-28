// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Sub-row for a DamageTriageFormer inference model (Option B). Mirrors
// EmbeddingModelRow's column layout, but there is no "Interactive Label"
// step — DTF predicts the full 5-class typology automatically — so the row
// only exposes Results (gpkg download + reports) once model.gpkgUrl is set.
import {
  IconButton,
  PrimaryButton,
  Text,
  TooltipHost,
} from "@fluentui/react";
import { useContext, useState } from "react";
import React from "react";
import PropTypes from "prop-types";
import { useNavigate } from "react-router-dom";
import { apiDelete, buildUrl } from "../../util/api";
import { AppContext } from "../../AppContext";
import StatusIndicator from "../OtherComponents/StatusIndicator";
import ValidationReportModal from "../BuildingValidation/ValidationReportModal";
import AssessmentReportModal from "../BuildingValidation/AssessmentReportModal";
import { fileDownload } from "../../util/file";
import { limitTextLength } from "../../util/conversion";

// Keep in sync with DTF_CATEGORIES in
// hastelib/src/hastegeo/workflows/damage_triage_infer.py.
const DTF_CATEGORIES = [
  "Undamaged",
  "Partial Roof Damage",
  "Total Roof Damage",
  "Partial Structural Collapse",
  "Total Structural Collapse",
];

function dtfInfoMetadata() {
  return [
    { label: "Model", value: "DamageTriageFormer" },
    { label: "Classes", value: DTF_CATEGORIES.join(", ") },
  ];
}

const DTFModelRow = ({
  model,
  projectId,
  imageLayerId,
  index,
  fetchProjectDetails,
  validationLabelCount = 0,
  mobile = false,
}) => {
  DTFModelRow.propTypes = {
    model: PropTypes.object.isRequired,
    projectId: PropTypes.string.isRequired,
    imageLayerId: PropTypes.string.isRequired,
    index: PropTypes.number.isRequired,
    fetchProjectDetails: PropTypes.func.isRequired,
    validationLabelCount: PropTypes.number,
    mobile: PropTypes.bool,
  };

  const { setDialog, setIsLoading } = useContext(AppContext);
  const navigate = useNavigate();
  const [showValidationReport, setShowValidationReport] = useState(false);
  const [showAssessmentReport, setShowAssessmentReport] = useState(false);

  const hasPredictions = !!model.gpkgUrl;
  const hasMapLayer = !!model.predictedDamageLayerUrl;
  const createdDate = model.creationDate
    ? `${model.creationDate.substring(0, 10)} ${model.creationDate.substring(
        11,
        19
      )}`
    : "";

  async function handleDeletion() {
    setDialog();
    setIsLoading(true, "Removing DamageTriageFormer run...");
    try {
      await apiDelete(
        `DeleteModel?projectId=${projectId}&modelId=${model.modelId}`
      );
      fetchProjectDetails();
    } catch (error) {
      console.error("Error removing DTF run:", error);
      setDialog("Error", "There was an error removing the DTF run.");
    }
    setIsLoading(false);
  }

  const resultsMenu = {
    items: [
      {
        key: "viewOnMap",
        text: "View on Map",
        iconProps: { iconName: "Forward" },
        disabled: !hasMapLayer,
        onClick: () => {
          navigate(
            `/visualizer/${projectId}/${imageLayerId}/${model.modelId}`
          );
        },
      },
      {
        key: "downloadGeopackage",
        text: "Download Geopackage (.gpkg)",
        iconProps: { iconName: "download" },
        disabled: !hasPredictions,
        onClick: () => {
          fileDownload(
            buildUrl(
              `GetModelArtifact?projectId=${projectId}` +
                `&modelId=${model.modelId}&kind=gpkg`
            ),
            setDialog
          );
        },
      },
      {
        key: "validationReport",
        text: "Validation Report",
        iconProps: { iconName: "ReportDocument" },
        disabled: !hasPredictions || !(validationLabelCount > 0),
        onClick: () => setShowValidationReport(true),
      },
      {
        key: "assessmentReport",
        text: "Assessment Report",
        iconProps: { iconName: "AnalyticsReport" },
        disabled: !hasPredictions,
        onClick: () => setShowAssessmentReport(true),
      },
    ],
  };

  const moreMenuOptions = {
    items: [
      {
        key: "remove",
        text: "Remove",
        iconProps: { iconName: "Delete" },
        onClick: () => {
          setDialog(
            "Important",
            `Do you want to remove this DamageTriageFormer run?`,
            [
              {
                type: "primary",
                key: "yes",
                text: "Yes",
                onClick: handleDeletion,
              },
              {
                type: "default",
                key: "no",
                text: "No",
                onClick: () => setDialog(),
              },
            ]
          );
        },
      },
    ],
  };

  const reportModals = (
    <>
      {showValidationReport && (
        <ValidationReportModal
          projectId={projectId}
          imageLayerId={imageLayerId}
          modelId={model.modelId}
          modelName={model.name}
          onDismiss={() => setShowValidationReport(false)}
        />
      )}
      {showAssessmentReport && (
        <AssessmentReportModal
          projectId={projectId}
          imageLayerId={imageLayerId}
          modelId={model.modelId}
          modelName={model.name}
          onDismiss={() => setShowAssessmentReport(false)}
        />
      )}
    </>
  );

  if (mobile) {
    return (
      <React.Fragment key={"dtfMobile_" + projectId + "_" + model.modelId}>
        <tr>
          <td className="custom-text-no-wrap pt-1">
            <Text variant="small">
              <span className="fw-semibold">Name:</span>{" "}
              <span>{model.name}</span>
            </Text>
          </td>
        </tr>
        <tr>
          <td>
            <Text variant="small">
              <span className="fw-semibold">Model:</span> DamageTriageFormer
            </Text>
          </td>
        </tr>
        <tr>
          <td>
            <Text variant="small">
              <span className="fw-semibold">Run:</span> {createdDate}
            </Text>
          </td>
        </tr>
        <tr>
          <td className="pe-3 custom-text-no-wrap">
            <Text variant="small">
              <span className="fw-semibold">User:</span>{" "}
              {limitTextLength(model.userId, false, 35)}
            </Text>
          </td>
        </tr>
        <tr>
          <td className="pe-3 custom-text-no-wrap d-flex align-items-center">
            <StatusIndicator
              currentStep={model.currentStep}
              totalSteps={model.totalSteps}
              progressPct={model.progressPct}
              status={model.status}
              statusMessage={model.statusMessage}
              id={`singleDTFStatus${index}`}
              prefix="DamageTriageFormer"
              infoMetadata={dtfInfoMetadata()}
            />
          </td>
        </tr>
        <tr>
          <td className="pb-2 pt-2">
            <PrimaryButton
              id={"dtfResults" + index}
              text="Results"
              menuProps={resultsMenu}
              allowDisabledFocus
              className="dashboard-button"
              disabled={!hasPredictions}
            />
          </td>
        </tr>
        <tr className="model-mobile-row">
          <td className="pb-2">
            <IconButton
              id={`singleDTFMoreOptions${index}`}
              className="no-dropdown-icon"
              menuProps={moreMenuOptions}
              iconProps={{ iconName: "more" }}
              title="Menu"
              ariaLabel="Menu"
            />
          </td>
        </tr>
        {reportModals}
      </React.Fragment>
    );
  }

  return (
    <tr>
      <td className="pe-3 custom-text-no-wrap">
        <TooltipHost content={model.name} delay={2}>
          <Text variant="small">
            <span>{limitTextLength(model.name, false, 59)}</span>
          </Text>
        </TooltipHost>
      </td>
      <td className="pe-3 custom-text-no-wrap d-none d-xxl-table-cell">
        <Text variant="small">
          <span className="fw-semibold">Run:</span> {createdDate}
        </Text>
      </td>
      <td className="pe-3 custom-text-no-wrap d-none d-xxl-table-cell">
        <Text variant="small">
          <span className="fw-semibold">User: </span>
          {limitTextLength(model.userId, false, 35)}
        </Text>
      </td>
      <td className="pe-3 custom-text-no-wrap">
        <Text variant="medium">DamageTriageFormer</Text>
      </td>
      <td className="pe-3 custom-text-no-wrap d-flex align-items-center">
        <StatusIndicator
          currentStep={model.currentStep}
          totalSteps={model.totalSteps}
          progressPct={model.progressPct}
          status={model.status}
          statusMessage={model.statusMessage}
          id={`singleDTFStatus${index}`}
          prefix="DamageTriageFormer"
          infoMetadata={dtfInfoMetadata()}
        />
      </td>
      <td className="pe-3 custom-text-no-wrap">
        <div className="d-flex align-items-center pt-1 pb-1">
          <PrimaryButton
            id={"dtfResults" + index}
            text="Results"
            menuProps={resultsMenu}
            allowDisabledFocus
            className="dashboard-button"
            disabled={!hasPredictions}
          />
        </div>
      </td>
      <td>
        <IconButton
          id={`singleDTFMoreOptions${index}`}
          className="no-dropdown-icon"
          menuProps={moreMenuOptions}
          iconProps={{ iconName: "more" }}
          title="Menu"
          ariaLabel="Menu"
        />
      </td>
      {reportModals}
    </tr>
  );
};

export default DTFModelRow;
