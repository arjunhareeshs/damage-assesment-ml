// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// "Update imagery": add newer TIFs to an existing image layer (e.g. fresh
// post-disaster imagery for part of the area). The server puts each new
// TIF on the layer's pixel grid, registers it to the old imagery, retires
// the TIFs it fully covers and rebuilds the mosaic and footprints; building
// damage results are kept, and the ones under the updated area are flagged
// for a re-run. See PutLayerImageryUpdate / hastegeo imagery_update.
import { useContext, useState } from "react";
import proptypes from "prop-types";
import { DefaultButton, MessageBar, MessageBarType, PrimaryButton, Text } from "@fluentui/react";

import SectionModal from "./SectionModal";
import CreateEditImageLayerFormImagerySources from "./CreateEditImageLayerFormImagerySources";
import { getUrlList, imageryOriginOptions, onFormChange } from "./CreateEditImageLayerHelper";
import { apiPut } from "../util/api";
import { AppContext } from "../AppContext";

const UpdateImageryModal = ({ imageLayer, projectId, onClose, onUpdated }) => {
  const { setIsLoading } = useContext(AppContext);
  const [error, setError] = useState("");
  const [componentState, setComponentState] = useState({
    projectId: projectId,
    postEventImageryUrls: [],
    currentPostEventImageryUrlControl: imageryOriginOptions[0].key,
    currentPostEventImageryUrlControlError: "",
    imageryOriginOptions: imageryOriginOptions,
  });

  const files = componentState.postEventImageryUrls;
  // A file entry turns into type "url" once its upload has finished.
  const uploading = files.some((f) => f.type !== "url");
  const urls = getUrlList(files).filter((u) => u);
  const currentCount = (imageLayer.postEventImageryUrls || []).length;
  const lastUpdate = (imageLayer.imageryUpdates || []).slice(-1)[0];

  async function submit() {
    setError("");
    setIsLoading(true, "Queuing imagery update");
    try {
      const result = await apiPut("PutLayerImageryUpdate", {
        projectId: projectId,
        imageLayerId: imageLayer.imageLayerId,
        postEventImageryUrls: urls,
      });
      if (result === 409) {
        setError("The image layer is still processing. Try again when it has finished.");
      } else {
        onUpdated();
        onClose();
      }
    } catch {
      setError("The imagery update could not be queued. Check that every file is a valid GeoTIFF URL or upload.");
    }
    setIsLoading(false);
  }

  const body = (
    <div className="d-flex flex-column" style={{ maxWidth: 720 }}>
      <Text variant="medium" className="mb-3">
        Add newer imagery for part of this layer&apos;s area. Each new file is aligned to the
        existing imagery (same pixel grid, registered to the old image), the files it fully
        covers are retired, and files it covers only in part stay underneath it. Damage results
        are kept; buildings inside the updated area are marked for a re-run.
      </Text>
      <Text variant="small" className="mb-3">
        This layer has {currentCount} active imagery file{currentCount === 1 ? "" : "s"}
        {(imageLayer.retiredPostEventImageryUrls || []).length > 0 &&
          ` and ${imageLayer.retiredPostEventImageryUrls.length} retired`}
        .
        {lastUpdate &&
          ` Last update: ${new Date(lastUpdate.date).toLocaleString()}, ` +
            `${(lastUpdate.newUrls || []).length} new file(s) replaced ` +
            `${(lastUpdate.retiredUrls || []).length}.`}
      </Text>
      <CreateEditImageLayerFormImagerySources
        onFormChange={onFormChange}
        setComponentState={setComponentState}
        componentState={componentState}
        field="postEventImageryUrls"
        currentEventImageryUrlControl="currentPostEventImageryUrlControl"
      />
      {error && (
        <MessageBar messageBarType={MessageBarType.error} className="mb-3">
          {error}
        </MessageBar>
      )}
      <div className="d-flex justify-content-end mt-2">
        <DefaultButton text="Cancel" onClick={onClose} className="me-2" />
        <PrimaryButton
          text={uploading ? "Uploading..." : "Update imagery"}
          onClick={submit}
          disabled={uploading || urls.length === 0}
        />
      </div>
    </div>
  );

  return (
    <SectionModal
      title={`Update imagery: ${imageLayer.name || ""}`}
      body={body}
      onClose={onClose}
      icon="Photo2Add"
    />
  );
};

UpdateImageryModal.propTypes = {
  imageLayer: proptypes.object.isRequired,
  projectId: proptypes.string.isRequired,
  onClose: proptypes.func.isRequired,
  onUpdated: proptypes.func.isRequired,
};

export default UpdateImageryModal;
