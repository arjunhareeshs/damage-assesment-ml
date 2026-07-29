// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AOI-scoped DamageTriageFormer (Option B, AOI-scoped —
// docs/OptionB_AOI_Integration_Plan.md Module 5). A small status readout,
// rendered inside LabelingToolRightPanel's existing panel, showing whether
// an AOI has been drawn and offering a way to clear it.
//
// Drawing itself happens through the SAME drawingManager LabelingTool.jsx
// already uses for damage labels: DrawingToolbar.jsx's "Draw AOI" button
// puts it into AOI_DRAWING_MODE ("draw-rectangle"), and LabelingTool.jsx's
// handleDrawingComplete diverts the resulting shape's geometry into
// aoiGeoJson instead of the labeling shape source (see that file's
// comment) — so this component only reflects state, it never touches the
// map or the drawing manager directly.
import PropTypes from "prop-types";
import { FontIcon } from "@fluentui/react/lib/Icon";

const AoiDrawTool = ({ aoiGeoJson, setAoiGeoJson }) => {
  AoiDrawTool.propTypes = {
    aoiGeoJson: PropTypes.object,
    setAoiGeoJson: PropTypes.func.isRequired,
  };

  if (!aoiGeoJson) {
    return (
      <div style={{ fontSize: 12, color: "#666", margin: "4px 0" }}>
        No AOI drawn. Use the &ldquo;Draw AOI&rdquo; tool to select an area
        for DamageTriageFormer.
      </div>
    );
  }

  return (
    <div
      style={{
        fontSize: 12,
        color: "#333",
        margin: "4px 0",
        display: "flex",
        alignItems: "center",
        gap: "6px",
      }}
      id="aoiStatus"
    >
      <FontIcon iconName="MapLayers" />
      <span>AOI drawn</span>
      <button
        onClick={() => setAoiGeoJson(null)}
        title="Clear the drawn AOI"
        className="drawing-button"
        id="clear-aoi"
        style={{ marginLeft: "auto" }}
      >
        <FontIcon iconName="Cancel" />
      </button>
    </div>
  );
};

export default AoiDrawTool;
