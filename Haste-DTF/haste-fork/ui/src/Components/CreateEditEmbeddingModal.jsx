// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modal to customize and launch a building-embedding job (building labeling
// workflow). Mirrors CreateEditModelTrainingModal — collects the embedding
// backbone + per-backbone parameters and POSTs to PutRunEmbeddingQueueMessage.
import { useContext, useMemo, useState } from "react";
import { Dropdown, TextField } from "@fluentui/react";
import {
  DefaultButton,
  PrimaryButton,
} from "@fluentui/react/lib/Button";
import proptypes from "prop-types";

import { apiPut } from "../util/api";
import { validateInt } from "../util/validation";
import { AppContext } from "../AppContext";
import SectionModal from "./SectionModal";

// Keep this list in sync with build_embedding_model() in
// hastelib/src/hastegeo/workflows/embed_buildings.py — the backend rejects
// anything not on its supported list.
const EMBEDDING_MODEL_OPTIONS = [
  {
    key: "mosaiks",
    text: "MOSAIKS (random conv features)",
    description:
      "Fast, untrained random convolution features. Output dim is configurable.",
  },
  {
    key: "dinov2_vits14",
    text: "DINOv2 ViT-S/14 (384-dim)",
    description:
      "Self-supervised ViT trained by Meta — strong general-purpose visual features. ~22M params, 384-dim output.",
  },
  {
    key: "dinov2_vitb14",
    text: "DINOv2 ViT-B/14 (768-dim)",
    description:
      "Larger DINOv2 variant. ~86M params, 768-dim output — slower but often more discriminative than ViT-S.",
  },
];

const CreateEditEmbeddingModal = ({
  onClose,
  projectId,
  imageLayer,
  fetchProjectDetails,
}) => {
  CreateEditEmbeddingModal.propTypes = {
    onClose: proptypes.func.isRequired,
    projectId: proptypes.string.isRequired,
    imageLayer: proptypes.object.isRequired,
    fetchProjectDetails: proptypes.func,
  };

  const { setDialog, appParams, setIsLoading } = useContext(AppContext);
  const [state, setState] = useState({
    name: "embedding-" + Date.now(),
    nameError: "",
    embeddingModel: "mosaiks",
    numFeatures: "1024",
    numFeaturesError: "",
    resizeFactor: "4",
    resizeFactorError: "",
    batchSize: "16",
    batchSizeError: "",
  });

  const isMosaiks = state.embeddingModel === "mosaiks";
  const modelHelp = useMemo(
    () =>
      EMBEDDING_MODEL_OPTIONS.find((o) => o.key === state.embeddingModel)
        ?.description || "",
    [state.embeddingModel]
  );

  function onField(value, key) {
    setState((s) => ({ ...s, [key]: value }));
  }

  function onModelChange(_e, option) {
    if (!option) return;
    // Switching to DINOv2 picks per-backbone defaults that match what the
    // server-side preprocessor would fill in (resizeFactor=1, no num_feats).
    setState((s) => {
      if (option.key === "mosaiks") {
        return {
          ...s,
          embeddingModel: "mosaiks",
          resizeFactor: s.resizeFactor || "4",
          numFeatures: s.numFeatures || "1024",
        };
      }
      return {
        ...s,
        embeddingModel: option.key,
        resizeFactor: "1",
      };
    });
  }

  async function submit() {
    const resizeFactorError = validateInt("Resize factor", state.resizeFactor);
    const batchSizeError = validateInt("Batch size", state.batchSize);
    const numFeaturesError = isMosaiks
      ? validateInt("Number of features", state.numFeatures)
      : "";
    if (numFeaturesError || resizeFactorError || batchSizeError) {
      setState((s) => ({
        ...s,
        numFeaturesError,
        resizeFactorError,
        batchSizeError,
      }));
      return;
    }

    setIsLoading(true, "Starting embedding job...");
    try {
      const body = {
        projectId,
        imageLayerId: imageLayer.imageLayerId,
        modelType: "embedding",
        name: state.name,
        embeddingModel: state.embeddingModel,
        resizeFactor: parseInt(state.resizeFactor, 10),
        batchSize: state.batchSize,
        userId: appParams.userId,
      };
      // num_feats is MOSAIKS-only. DINOv2 variants have a fixed output dim
      // determined by the variant — sending a value would be misleading
      // since the workflow ignores it.
      if (isMosaiks) {
        body.numFeatures = parseInt(state.numFeatures, 10);
      }
      await apiPut("PutRunEmbeddingQueueMessage", body);
      onClose();
      if (fetchProjectDetails) fetchProjectDetails();
      setDialog("Success", "Embedding job started.", [
        {
          type: "primary",
          key: "close",
          text: "Close",
          onClick: () => setDialog(),
        },
      ]);
    } catch (error) {
      console.error("Error starting embedding job:", error);
      setDialog("Error", "Failed to start the embedding job.");
    }
    setIsLoading(false);
  }

  return (
    <SectionModal
      title="New Embedding"
      icon="OpenFolderHorizontal"
      onClose={onClose}
      body={
        <>
          <div className="row mb-2">
            <div className="col-12">
              <TextField
                id="createEmbeddingName"
                label="Name"
                value={state.name}
                onChange={(e, v) => onField(v, "name")}
                errorMessage={state.nameError}
              />
            </div>
          </div>
          <div className="row mb-2">
            <div className="col-12">
              <Dropdown
                id="createEmbeddingModel"
                label="Embedding model"
                selectedKey={state.embeddingModel}
                options={EMBEDDING_MODEL_OPTIONS.map((o) => ({
                  key: o.key,
                  text: o.text,
                }))}
                onChange={onModelChange}
              />
              <p style={{ fontSize: 12, color: "#666", margin: "8px 0" }}>
                {modelHelp}
              </p>
            </div>
          </div>
          <div className="row mb-4">
            <div className="col-12 flex-column flex-md-row d-flex">
              {isMosaiks && (
                <TextField
                  id="createEmbeddingNumFeatures"
                  label="Number of features"
                  className="me-0 me-md-4 mb-2"
                  value={state.numFeatures}
                  onChange={(e, v) => onField(v, "numFeatures")}
                  errorMessage={state.numFeaturesError}
                  required
                />
              )}
              <TextField
                id="createEmbeddingResizeFactor"
                label="Resize factor"
                className="me-0 me-md-4 mb-2"
                value={state.resizeFactor}
                onChange={(e, v) => onField(v, "resizeFactor")}
                errorMessage={state.resizeFactorError}
                required
              />
              <TextField
                id="createEmbeddingBatchSize"
                label="Batch size"
                className="mb-2"
                value={state.batchSize}
                onChange={(e, v) => onField(v, "batchSize")}
                errorMessage={state.batchSizeError}
                required
              />
            </div>
          </div>
          <div className="row">
            <div className="col-12 d-flex justify-content-end">
              <PrimaryButton
                className="me-2"
                onClick={submit}
                id="createEmbeddingSubmit"
              >
                Embed
              </PrimaryButton>
              <DefaultButton onClick={onClose}>Cancel</DefaultButton>
            </div>
          </div>
        </>
      }
    />
  );
};

export default CreateEditEmbeddingModal;
