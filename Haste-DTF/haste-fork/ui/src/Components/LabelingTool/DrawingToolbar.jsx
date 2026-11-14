// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { FontIcon } from "@fluentui/react/lib/Icon";
import "../../assets/css/drawingToolbar.css";
import { AOI_DRAWING_MODE, AOI_POLYGON_MODE } from "./LabelingToolHelper.js";

const DrawingToolbar = ({ drawingManager, setDrawingManager, onDrawingStart, onFitImagery }) => {
  DrawingToolbar.propTypes = {
    drawingManager: PropTypes.object.isRequired,
    setDrawingManager: PropTypes.func.isRequired,
    onDrawingStart: PropTypes.func,
    // Zooms the map out to show the whole imagery.
    onFitImagery: PropTypes.func,
  };

  const [activeMode, setActiveMode] = useState("none");

  const buttons = [
    {
      // AOI-scoped DamageTriageFormer polygon
      mode: AOI_POLYGON_MODE,
      // "PolygonalShape" isn't a Fluent MDL2 icon, so this button used to
      // render blank and looked like there was no polygon tool at all.
      label: (
        <>
          <FontIcon iconName="Pentagon" /> Polygon
        </>
      ),
      disabled: false,
      title: "Polygon Tool: click multiple points on the map to draw a custom area",
      id: "draw-aoi-polygon",
    },
    {
      // AOI-scoped DamageTriageFormer rectangle
      mode: AOI_DRAWING_MODE,
      label: (
        <>
          <FontIcon iconName="RectangleShape" /> Rectangle
        </>
      ),
      disabled: false,
      title: "Rectangle Tool: click and drag to draw a rectangular area",
      id: "draw-aoi",
    },
  ];

  const stopDrawing = () => {
    if (!drawingManager) return;
    drawingManager.isAoiDrawing = false;
    drawingManager.setOptions({ mode: "none" });
    setActiveMode("none");
  };

  // Esc cancels the current drawing tool.
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "Escape") stopDrawing();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawingManager]);

  // The manager returns to "none" after a shape is finished; reflect that.
  useEffect(() => {
    if (!drawingManager) return;
    const timer = setInterval(() => {
      const mode = drawingManager.getOptions().mode;
      if (mode === "none") setActiveMode((m) => (m === "none" ? m : "none"));
    }, 300);
    return () => clearInterval(timer);
  }, [drawingManager]);

  const handleClick = (mode, isDisabled) => {
    if (isDisabled) return;
    if (!drawingManager) return;

    // Clicking the active tool again turns it off.
    if (activeMode === mode) {
      stopDrawing();
      return;
    }
    if (onDrawingStart) onDrawingStart();

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
            style={{ width: "auto", padding: "0 10px", display: "flex", alignItems: "center", gap: "6px" }}
          >
            {btn.label}
          </button>
        );
      })}

      {onFitImagery ? (
        <button
          onClick={onFitImagery}
          title="Zoom out to show the whole imagery"
          className="drawing-button"
          id="fit-imagery"
          style={{ width: "auto", padding: "0 10px", display: "flex", alignItems: "center", gap: "6px" }}
        >
          <FontIcon iconName="FullScreen" /> Full image
        </button>
      ) : null}
    </div>
  );
};

export default DrawingToolbar;
