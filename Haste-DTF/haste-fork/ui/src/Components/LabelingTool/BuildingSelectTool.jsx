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
//   * every footprint is outlined as soon as the map loads, so it is clear
//     what can be assessed;
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

  // Load all footprints once the map exists.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || dataSourceRef.current) return undefined;
    let cancelled = false;

    async function load() {
      try {
        const collection = await apiGet(
          `GetLayerBuildings?imageLayerId=${encodeURIComponent(imageLayerId)}`
        );
        if (cancelled) return;
        const features = (collection && collection.features) || [];
        features.forEach((f) => {
          f.properties = { ...f.properties, selected: false, inAoi: false };
        });

        const dataSource = new window.atlas.source.DataSource();
        map.sources.add(dataSource);
        dataSource.add({ type: "FeatureCollection", features });
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

        map.events.add("click", fillLayer, (e) => {
          if (!activeRef.current || !e.shapes || !e.shapes.length) return;
          const buildingId = e.shapes[0].getProperties().id;
          setSelectedBuildingIds((prev) =>
            prev.includes(buildingId)
              ? prev.filter((id) => id !== buildingId)
              : [...prev, buildingId]
          );
        });

        if (onBuildingsLoaded) {
          onBuildingsLoaded({
            total: collection.total ?? features.length,
            shown: features.length,
            truncated: Boolean(collection.truncated),
          });
        }
        // Apply any AOI / selection that arrived before the footprints did.
        applySelection(selectedBuildingIds);
        refreshAoi(aoiGeoJson);
      } catch (error) {
        console.error("Error loading building footprints:", error);
        if (onBuildingsLoaded) onBuildingsLoaded({ error: true });
      }
    }
    load();
    return () => {
      cancelled = true;
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
