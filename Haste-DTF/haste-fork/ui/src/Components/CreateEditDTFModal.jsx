// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modal to launch a DamageTriageFormer inference job (Option B). Mirrors
// CreateEditEmbeddingModal but with no tunable pipeline params — DTF is a
// single pretrained checkpoint (no backbone choice, no training), so the
// only user input is a name. POSTs to PutRunDTFInferenceQueueMessage.
import { useContext, useState } from "react";
import { TextField } from "@fluentui/react";
import {
  DefaultButton,
  PrimaryButton,
} from "@fluentui/react/lib/Button";
import proptypes from "prop-types";

import { apiPut } from "../util/api";
import { AppContext } from "../AppContext";
import SectionModal from "./SectionModal";

const CreateEditDTFModal = ({
  onClose,
  projectId,
  imageLayer,
  fetchProjectDetails,
}) => {
  CreateEditDTFModal.propTypes = {
    onClose: proptypes.func.isRequired,
    projectId: proptypes.string.isRequired,
    imageLayer: proptypes.object.isRequired,
    fetchProjectDetails: proptypes.func,
  };

  const { setDialog, appParams, setIsLoading } = useContext(AppContext);
  const [state, setState] = useState({
    name: "dtf-" + Date.now(),
    nameError: "",
  });

  function onField(value, key) {
    setState((s) => ({ ...s, [key]: value }));
  }

  async function submit() {
    setIsLoading(true, "Starting DamageTriageFormer inference...");
    try {
      const body = {
        projectId,
        imageLayerId: imageLayer.imageLayerId,
        modelType: "dtf",
        name: state.name,
        userId: appParams.userId,
      };
      await apiPut("PutRunDTFInferenceQueueMessage", body);
      onClose();
      if (fetchProjectDetails) fetchProjectDetails();
      setDialog(
        "Success",
        "DamageTriageFormer inference started. Buildings across the full " +
          "layer will be classified into 5 damage typology classes — no " +
          "labeling needed.",
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
      console.error("Error starting DamageTriageFormer job:", error);
      setDialog("Error", "Failed to start the DamageTriageFormer job.");
    }
    setIsLoading(false);
  }

  return (
    <SectionModal
      title="Run DamageTriageFormer"
      icon="OpenFolderHorizontal"
      onClose={onClose}
      body={
        <>
          <div className="row mb-2">
            <div className="col-12">
              <TextField
                id="createDTFName"
                label="Name"
                value={state.name}
                onChange={(e, v) => onField(v, "name")}
                errorMessage={state.nameError}
              />
            </div>
          </div>
          <div className="row mb-4">
            <div className="col-12">
              <p style={{ fontSize: 12, color: "#666", margin: "8px 0" }}>
                Classifies every building footprint in this layer into 5
                damage typology classes (Undamaged, Partial/Total Roof
                Damage, Partial/Total Structural Collapse) using the
                pretrained DamageTriageFormer model — fully automated, no
                analyst labeling required.
              </p>
            </div>
          </div>
          <div className="row">
            <div className="col-12 d-flex justify-content-end">
              <PrimaryButton
                className="me-2"
                onClick={submit}
                id="createDTFSubmit"
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

export default CreateEditDTFModal;
