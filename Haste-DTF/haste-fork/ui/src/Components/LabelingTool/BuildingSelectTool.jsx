// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AOI-scoped DamageTriageFormer (Option B, AOI-scoped —
// docs/OptionB_AOI_Integration_Plan.md Module 6.1's "Case 2: analyst
// selected/clicked specific buildings"). Renders this layer's building
// footprints (via the existing GetBuildingFootprintsGeoJSON endpoint) as a
// clickable vector layer; clicking a building toggles it into
// selectedBuildingIds. This is a real, working implementation of the
// building-click-selection path — CustomSelection.jsx/PolygonMultiEdit
// (also in this folder) is a single-polygon vertex-dragging tool, not a
// multi-building click-selector, and isn't reused here for that reason.
//
// Renders no visible React output itself (map layers only) — mirrors the
// shape of loadStudyArea/loadPredictedDamageLayer in Visualizer.jsx, just
// as a component so it can own its own effect lifecycle independent of
// LabelingTool.jsx's drawingManager effect.
import { useEffect, useRef } from "react";
import PropTypes from "prop-types";
import { apiGet } from "../../util/api";

const SELECTED_COLOR = "#0078d4";
const UNSELECTED_COLOR = "rgba(255, 255, 255, 0.01)";

const BuildingSelectTool = ({
  mapRef,
  projectId,
  imageLayerId,
  active,
  setSelectedBuildingIds,
}) => {
  BuildingSelectTool.propTypes = {
    // The Azure Maps atlas.Map instance is read from mapRef.current inside
    // the effect below, not dereferenced at render time (React refs should
    // only be read in effects/handlers, not during render) — mirrors how
    // LabelingTool.jsx passes mapRef to LabelingToolLeftPanel.
    mapRef: PropTypes.object.isRequired,
    projectId: PropTypes.string.isRequired,
    imageLayerId: PropTypes.string.isRequired,
    active: PropTypes.bool.isRequired,
    setSelectedBuildingIds: PropTypes.func.isRequired,
  };

  const layersRef = useRef([]);
  const loadedRef = useRef(false);

  // Lazily load footprints the first time selection mode is turned on —
  // most sessions never use this path, so there's no reason to pay for it
  // on every map load.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !active || loadedRef.current) return;
    loadedRef.current = true;
    let cancelled = false;

    async function load() {
      try {
        const featureCollection = await apiGet(
          `GetBuildingFootprintsGeoJSON?projectId=${projectId}&imageLayerId=${imageLayerId}&sample=2000`
        );
        if (cancelled || !featureCollection?.features?.length) return;

        const dataSource = new window.atlas.source.DataSource();
        map.sources.add(dataSource);
        dataSource.add(featureCollection);

        const fillLayer = new window.atlas.layer.PolygonLayer(dataSource, null, {
          fillColor: [
            "case",
            ["boolean", ["get", "selected"], false],
            SELECTED_COLOR,
            UNSELECTED_COLOR,
          ],
          fillOpacity: 0.5,
          visible: active,
        });
        fillLayer.customId = "buildingSelectFill";
        map.layers.add(fillLayer);

        const outlineLayer = new window.atlas.layer.LineLayer(dataSource, null, {
          strokeColor: SELECTED_COLOR,
          strokeWidth: 1,
          visible: active,
        });
        outlineLayer.customId = "buildingSelectOutline";
        map.layers.add(outlineLayer);

        layersRef.current = [fillLayer, outlineLayer];

        map.events.add("click", fillLayer, (e) => {
          if (!e.shapes || !e.shapes.length) return;
          const shape = e.shapes[0];
          const shapeProperties = shape.getProperties();
          const buildingId = shapeProperties.id;
          const wasSelected = Boolean(shapeProperties.selected);
          shape.setProperties({ ...shapeProperties, selected: !wasSelected });
          setSelectedBuildingIds((prev) =>
            wasSelected
              ? prev.filter((id) => id !== buildingId)
              : [...prev, buildingId]
          );
        });
      } catch (error) {
        console.error(
          "Error loading building footprints for selection:",
          error
        );
      }
    }
    load();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapRef, active]);

  // Show/hide the layer as selection mode is toggled, without reloading it.
  useEffect(() => {
    layersRef.current.forEach((layer) => layer.setOptions({ visible: active }));
  }, [active]);

  return null;
};

export default BuildingSelectTool;
