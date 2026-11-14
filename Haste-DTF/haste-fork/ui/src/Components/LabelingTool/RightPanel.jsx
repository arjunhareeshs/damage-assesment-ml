import { useEffect, useState } from "react";
import { ChoiceGroup, PrimaryButton } from "@fluentui/react";
import { saveLabels } from "./LabelingToolHelper";
import DrawingToolbar from "./DrawingToolbar";
import PropType from "prop-types";
import { updateShape } from "./LabelingToolHelper";

const RightPanel = ({
  primaryClasses,
  selectedPrimaryClass,
  setSelectedPrimaryClass,
  drawingManager,
  setDrawingManager,
  setDialog,
  labelingToolDataRef,
  setIsLoading,
  setHasUnsavedChanges,
  selectedShape,
}) => {
  RightPanel.propTypes = {
    primaryClasses: PropType.array.isRequired,
    selectedPrimaryClass: PropType.string.isRequired,
    setSelectedPrimaryClass: PropType.func.isRequired,
    drawingManager: PropType.object.isRequired,
    setDrawingManager: PropType.func.isRequired,
    setDialog: PropType.func.isRequired,
    labelingToolDataRef: PropType.object.isRequired,
    setIsLoading: PropType.func.isRequired,
    setHasUnsavedChanges: PropType.func.isRequired,
    selectedShape: PropType.object,
  };

  useEffect(() => {
    if (selectedShape !== null) {
      updateShape(
        drawingManager,
        selectedShape,
        selectedPrimaryClass,
      );
      setHasUnsavedChanges(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPrimaryClass]);

  async function handleSave() {
    const isSaved = await saveLabels(
      drawingManager,
      labelingToolDataRef,
      setIsLoading,
      setHasUnsavedChanges
    );

    var message = "Labels saved successfully";

    if (!isSaved) {
      message = "There was an error saving the labels";
    }

    const buttons = [
      {
        type: "primary",
        key: "close",
        text: "Close",
        onClick: () => {
          setDialog("", "", []);
        },
      },
    ];

    setDialog("Important", message, buttons);
  }

  function handlePrimaryClassChange(newClass) {
    if(selectedShape !== null) {
      const buttons = [
        {
          type: "primary",
          key: "yes",
          text: "Yes",
          onClick: () => {
            setSelectedPrimaryClass(newClass);
            setDialog("", "", []);
            setTimeout(() => {
            document.getElementById("pointer").click();
            }, 100);
          }
        },
        {
          type: "default",
          key: "no",
          text: "No",
          onClick: () => {
            setDialog("", "", []);
          },
        },        
      ];
      setDialog("Important", "You are about to change the primary class of the selected label. Do you want to continue?", buttons);
    }else{  
      setSelectedPrimaryClass(newClass);
    }
  }

  return (
    <>
      <DrawingToolbar drawingManager={drawingManager} setDrawingManager={setDrawingManager} />
      <div
        style={{
          position: "absolute",
          right: 10,
          top: 70,
          backgroundColor: "rgba(255, 255, 255, 1)",
          padding: "5px 10px",
          borderRadius: "5px",
          zIndex: 1000,
        }}
        id="rightPanel"
      >
        <div className="col-12 d-flex">
          <ChoiceGroup
            options={
              primaryClasses && primaryClasses.length > 0 ? primaryClasses : []
            }
            selectedKey={selectedPrimaryClass}
            label="Primary Class"
            onChange={(e, option) => {
              handlePrimaryClassChange(option.key);
            }}
          />
        </div>
        <div className="col-12 d-flex mt-2 flex-column pt-2 pb-2">
          <PrimaryButton
            id="saveAndTrainButton"
            iconProps={{ iconName: "Save" }}
            className="w-100"
            text="Save"
            onClick={async () => {
              await handleSave();
            }}
          />
        </div>
      </div>
    </>
  );
};

export default RightPanel;
