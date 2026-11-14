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
import { useRef, useState } from "react";
import { FontIcon } from "@fluentui/react/lib/Icon";

// Accept a Polygon/MultiPolygon geometry, a Feature, or a FeatureCollection
// (several polygons are merged into one MultiPolygon).
function extractAoiGeometry(parsed) {
  const geoms = [];
  const collect = (node) => {
    if (!node) return;
    if (node.type === "FeatureCollection") (node.features || []).forEach(collect);
    else if (node.type === "Feature") collect(node.geometry);
    else if (node.type === "Polygon") geoms.push(node.coordinates);
    else if (node.type === "MultiPolygon") node.coordinates.forEach((c) => geoms.push(c));
  };
  collect(parsed);
  if (geoms.length === 0) {
    throw new Error("No Polygon or MultiPolygon found in the file.");
  }
  const bad = geoms.some((poly) =>
    poly.some((ring) =>
      ring.some(([x, y]) => !(x >= -180 && x <= 180 && y >= -90 && y <= 90))
    )
  );
  if (bad) {
    throw new Error("Coordinates must be longitude/latitude (EPSG:4326).");
  }
  return geoms.length === 1
    ? { type: "Polygon", coordinates: geoms[0] }
    : { type: "MultiPolygon", coordinates: geoms };
}

const AoiDrawTool = ({ aoiGeoJson, setAoiGeoJson, getStudyArea }) => {
  AoiDrawTool.propTypes = {
    aoiGeoJson: PropTypes.object,
    setAoiGeoJson: PropTypes.func.isRequired,
    // Returns the layer's study area (a FeatureCollection covering all of
    // the imagery), used by "Whole image".
    getStudyArea: PropTypes.func,
  };

  const fileRef = useRef(null);
  const [importError, setImportError] = useState("");

  function handleFile(e) {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target.result);
        const geom = extractAoiGeometry(parsed);
        setAoiGeoJson(geom);
        setImportError("");
      } catch (err) {
        setImportError(err.message);
      }
    };
    reader.readAsText(file);
  }

  function useWholeImage() {
    try {
      const studyArea = getStudyArea ? getStudyArea() : null;
      if (!studyArea) throw new Error("The image extent isn't loaded yet.");
      setAoiGeoJson(extractAoiGeometry(studyArea));
      setImportError("");
    } catch (err) {
      setImportError(err.message);
    }
  }

  const importControls = (
    <>
      <button
        onClick={useWholeImage}
        title="Use the full extent of the imagery as the area"
        className="drawing-button me-2"
        id="whole-image-aoi"
        style={{ width: "auto", padding: "0 8px" }}
      >
        <FontIcon iconName="FullScreen" /> Whole image
      </button>
      <button
        onClick={() => fileRef.current && fileRef.current.click()}
        title="Import a GeoJSON Polygon/MultiPolygon as the area"
        className="drawing-button"
        id="import-aoi"
        style={{ width: "auto", padding: "0 8px" }}
      >
        <FontIcon iconName="Upload" /> Import GeoJSON
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".geojson,.json,application/geo+json,application/json"
        className="d-none"
        onChange={handleFile}
      />
      {importError ? (
        <div style={{ fontSize: 12, color: "#a4262c", marginTop: 4 }}>
          {importError}
        </div>
      ) : null}
    </>
  );

  if (!aoiGeoJson) {
    return (
      <div style={{ fontSize: 12, color: "#666", margin: "4px 0" }}>
        No area drawn. Use the Polygon or Rectangle tool (top right), take
        the whole image, or import a GeoJSON file.
        <div className="mt-2">{importControls}</div>
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
      <span>Area drawn</span>
      <button
        onClick={() => setAoiGeoJson(null)}
        title="Clear the drawn area"
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
