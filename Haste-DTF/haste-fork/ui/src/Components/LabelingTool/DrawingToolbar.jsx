// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { useState } from "react";
import PropTypes from "prop-types";
import { FontIcon } from "@fluentui/react/lib/Icon";
import "../../assets/css/drawingToolbar.css";
import { AOI_DRAWING_MODE, AOI_POLYGON_MODE } from "./LabelingToolHelper.js";

const DrawingToolbar = ({ drawingManager, setDrawingManager, undo, redo }) => {
  DrawingToolbar.propTypes = {
    drawingManager: PropTypes.object.isRequired,
    setDrawingManager: PropTypes.func.isRequired,
    undo: PropTypes.func.isRequired,
    redo: PropTypes.func.isRequired,
  };

  const [activeMode, setActiveMode] = useState("none");

  const buttons = [
    {
      mode: "none",
      label: <FontIcon iconName="HandsFree" />,
      disabled: false,
      title: "Pointer",
      id: "pointer",
    },
    {
      mode: "draw-polygon",
      label: <FontIcon iconName="WebAppBuilderModule" />,
      disabled: false,
      title: "Draw Polygon",
      id: "draw-polygon",
    },
    {
      mode: "edit-geometry",
      label: <FontIcon iconName="Edit" />,
      disabled: false,
      title: "Edit Geometry",
      id: "edit-geometry",
    },
    {
      label: <FontIcon iconName="Undo" />,
      disabled: false,
      title: "Undo",
      id: "undo",
    },
    {
      label: <FontIcon iconName="Redo" />,
      disabled: false,
      title: "Redo",
      id: "redo",
    },
    {
      mode: "draw-line",      
      label: <FontIcon iconName="Cut" />,
      disabled: false,
      title: "Cut",
      id: "cut",
    },
    {
      mode: "erase-geometry",
      label: <FontIcon iconName="Delete" />,
      disabled: false,
      title: "Erase Geometry",
      id: "erase-geometry",
    },
    {
      // AOI-scoped DamageTriageFormer rectangle
      mode: AOI_DRAWING_MODE,
      label: <FontIcon iconName="MapLayers" />,
      disabled: false,
      title: "Draw AOI Rectangle (for DamageTriageFormer)",
      id: "draw-aoi",
    },
    {
      // AOI-scoped DamageTriageFormer polygon
      mode: AOI_POLYGON_MODE,
      label: <FontIcon iconName="PolygonalShape" />,
      disabled: false,
      title: "Draw AOI Polygon (for DamageTriageFormer)",
      id: "draw-aoi-polygon",
    },
  ];

  const handleClick = (mode, isDisabled) => {
    if (isDisabled) return;
    if (!drawingManager) return;

    if (mode === AOI_POLYGON_MODE) {
      drawingManager.isAoiDrawing = true;
      drawingManager.setOptions({ mode: "draw-polygon" });
      setDrawingManager(drawingManager);
      setActiveMode(mode);
      return;
    }

    if (mode === AOI_DRAWING_MODE) {
      drawingManager.isAoiDrawing = true;
      drawingManager.setOptions({ mode: AOI_DRAWING_MODE });
      setDrawingManager(drawingManager);
      setActiveMode(mode);
      return;
    }

    drawingManager.isAoiDrawing = false;
    drawingManager.setOptions({ mode });
    setDrawingManager(drawingManager);
    setActiveMode(mode);
  };

  return (
    <div
      style={{
        position: "absolute",
        right: 10,
        top: 10,
        backgroundColor: "rgba(255, 255, 255, 1)",
        padding: "10px 10px",
        borderRadius: "5px",
        zIndex: 1000,
        display: "flex",
        gap: "5px",
      }}
      id="drawingToolbar"
    >
      {buttons.map((btn) => {
        const classNames = ["drawing-button"];

        if (btn.id === "undo" || btn.id === "redo") {
          return (<button
            key={btn.id}
            onClick={() => {
              if (btn.id === "undo") {
                undo();
              } else {
                redo();
              }
            }}
            disabled={btn.disabled}
            title={btn.title}
            className={"drawing-button"}
            id={btn.id}
          >
            {btn.label}
          </button>);
        } else {

          if (activeMode === btn.mode) classNames.push("active");
          if (btn.disabled) classNames.push("disabled");

          return (
            <button
              key={btn.mode}
              onClick={() => handleClick(btn.mode, btn.disabled)}
              disabled={btn.disabled}
              title={btn.title}
              className={classNames.join(" ")}
              id={btn.id}
            >
              {btn.label}
            </button>
          );
        }
      })}

    </div>
  );
};

export default DrawingToolbar;
