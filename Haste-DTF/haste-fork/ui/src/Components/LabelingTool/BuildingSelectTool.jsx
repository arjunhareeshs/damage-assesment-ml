// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// The layer's building footprints on the labeling map.
//
// What the analyst sees here is exactly what a DamageTriageFormer run works
// on: the footprints come from hg_buildings (GetLayerBuildings) -- the same
// rows, with the same ids, that an AOI run resolves -- rather than the old
// random 2,000-building sample, which silently left buildings off the map.
//
//   * the footprints of the current view are outlined (re-fetched as the map
//     moves, capped server-side), so it is clear what can be assessed;
//   * buildings intersecting the drawn area are filled amber -- the backend
//     decides which, with the run's own ST_Intersects predicate;
//   * in "Select buildings" mode, clicking toggles a building (blue). The map
//     follows selectedBuildingIds, so "Clear" in the panel clears it here too.
//
// Renders no React output itself (map layers only).
import { useEffect, useRef } from "react";
import PropTypes from "prop-types";
import { apiGet, apiPost } from "../../util/api";

const SELECTED_COLOR = "#0078d4";
const IN_AOI_COLOR = "#ffb900";
const OUTLINE_COLOR = "#ffffff";
// Fully transparent polygons don't receive clicks; this is invisible but hit-testable.
const CLICKABLE_TRANSPARENT = "rgba(255, 255, 255, 0.01)";

const BuildingSelectTool = ({
  mapRef,
  imageLayerId,
  active,
  aoiGeoJson,
  selectedBuildingIds,
  setSelectedBuildingIds,
  onBuildingsLoaded,
}) => {
  BuildingSelectTool.propTypes = {
    // The atlas.Map instance, read inside effects only.
    mapRef: PropTypes.object.isRequired,
    projectId: PropTypes.string,
    imageLayerId: PropTypes.string.isRequired,
    // "Select buildings" mode: clicks toggle selection only while true.
    active: PropTypes.bool.isRequired,
    aoiGeoJson: PropTypes.object,
    selectedBuildingIds: PropTypes.array.isRequired,
    setSelectedBuildingIds: PropTypes.func.isRequired,
    // Called with { total, shown, truncated } once footprints are loaded,
    // or { error } if they could not be.
    onBuildingsLoaded: PropTypes.func,
  };

  const dataSourceRef = useRef(null);
  const activeRef = useRef(active);

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  // Footprints of the current viewport, re-fetched whenever the view
  // settles. A large layer is never sent whole: without a bbox the server
  // could only return the first N buildings by id, scattered anywhere.
  const selectedRef = useRef(selectedBuildingIds);
  useEffect(() => {
    selectedRef.current = selectedBuildingIds;
  }, [selectedBuildingIds]);
  const inAoiRef = useRef(new Set());

  useEffect(() => {
    const map = mapRef.current;
    if (!map || dataSourceRef.current) return undefined;
    let cancelled = false;
    let requestId = 0;

    const dataSource = new window.atlas.source.DataSource();
    map.sources.add(dataSource);
    dataSourceRef.current = dataSource;

    const fillLayer = new window.atlas.layer.PolygonLayer(dataSource, null, {
      fillColor: [
        "case",
        ["boolean", ["get", "selected"], false],
        SELECTED_COLOR,
        ["boolean", ["get", "inAoi"], false],
        IN_AOI_COLOR,
        CLICKABLE_TRANSPARENT,
      ],
      fillOpacity: [
        "case",
        ["boolean", ["get", "selected"], false],
        0.55,
        ["boolean", ["get", "inAoi"], false],
        0.45,
        1,
      ],
    });
    fillLayer.customId = "buildingSelectFill";
    map.layers.add(fillLayer);

    const outlineLayer = new window.atlas.layer.LineLayer(dataSource, null, {
      strokeColor: [
        "case",
        ["boolean", ["get", "selected"], false],
        SELECTED_COLOR,
        ["boolean", ["get", "inAoi"], false],
        IN_AOI_COLOR,
        OUTLINE_COLOR,
      ],
      strokeWidth: 1.5,
    });
    outlineLayer.customId = "buildingSelectOutline";
    map.layers.add(outlineLayer);

    const onClick = (e) => {
      if (!activeRef.current || !e.shapes || !e.shapes.length) return;
      const buildingId = e.shapes[0].getProperties().id;
      setSelectedBuildingIds((prev) =>
        prev.includes(buildingId)
          ? prev.filter((id) => id !== buildingId)
          : [...prev, buildingId]
      );
    };
    map.events.add("click", fillLayer, onClick);

    async function load() {
      const bounds = map.getCamera().bounds;
      const id = ++requestId;
      try {
        const bboxParam = bounds ? `&bbox=${bounds.join(",")}` : "";
        const collection = await apiGet(
          `GetLayerBuildings?imageLayerId=${encodeURIComponent(imageLayerId)}${bboxParam}`
        );
        if (cancelled || id !== requestId) return;
        const selected = new Set(selectedRef.current);
        const features = ((collection && collection.features) || []).map((f) => ({
          ...f,
          properties: {
            ...f.properties,
            selected: selected.has(f.properties.id),
            inAoi: inAoiRef.current.has(f.properties.id),
          },
        }));
        dataSource.setShapes(features);
        if (onBuildingsLoaded) {
          onBuildingsLoaded({
            total: collection.total ?? features.length,
            shown: features.length,
            truncated: Boolean(collection.truncated),
          });
        }
      } catch (error) {
        console.error("Error loading building footprints:", error);
        if (onBuildingsLoaded && !cancelled) onBuildingsLoaded({ error: true });
      }
    }
    map.events.add("moveend", load);
    load();
    return () => {
      cancelled = true;
      map.events.remove("moveend", load);
      map.events.remove("click", fillLayer, onClick);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageLayerId]);

  function applySelection(ids) {
    const dataSource = dataSourceRef.current;
    if (!dataSource) return;
    const selected = new Set(ids);
    dataSource.getShapes().forEach((shape) => {
      const shapeProps = shape.getProperties();
      const isSelected = selected.has(shapeProps.id);
      if (Boolean(shapeProps.selected) !== isSelected) {
        shape.setProperties({ ...shapeProps, selected: isSelected });
      }
    });
  }

  function setInAoi(ids) {
    inAoiRef.current = ids;
    const dataSource = dataSourceRef.current;
    if (!dataSource) return;
    dataSource.getShapes().forEach((shape) => {
      const shapeProps = shape.getProperties();
      const inAoi = ids.has(shapeProps.id);
      if (Boolean(shapeProps.inAoi) !== inAoi) {
        shape.setProperties({ ...shapeProps, inAoi });
      }
    });
  }

  const aoiRequestRef = useRef(0);
  function refreshAoi(aoi) {
    if (!dataSourceRef.current) return;
    const request = ++aoiRequestRef.current;
    if (!aoi) {
      setInAoi(new Set());
      return;
    }
    apiPost("GetBuildingsInAoi", { imageLayerId, aoi, returnIds: true })
      .then((result) => {
        if (request !== aoiRequestRef.current) return; // a newer AOI won
        setInAoi(new Set((result && result.buildingIds) || []));
      })
      .catch((error) => {
        console.error("Error finding buildings in the drawn area:", error);
      });
  }

  // Highlight the buildings inside the drawn area whenever it changes.
  useEffect(() => {
    refreshAoi(aoiGeoJson);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aoiGeoJson]);

  // Keep the map's blue highlight in step with the selection (incl. "Clear").
  useEffect(() => {
    applySelection(selectedBuildingIds);
  }, [selectedBuildingIds]);

  return null;
};

export default BuildingSelectTool;
