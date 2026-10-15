// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { useEffect, useRef } from "react";
import PropTypes from "prop-types";
import { apiGet } from "../../util/api";
import { DTF_CLASS_COLORS, RESULT_STATUS_COLORS } from "./LabelingToolHelper.js";

// Fill used for a completed building with no known class (should not
// happen; kept visible rather than transparent).
const UNKNOWN_CLASS_COLOR = "#b0bec5";

const AoiAssessmentMapLayer = ({
  mapRef,
  assessmentId,
  onBuildingClick,
  refreshTrigger,
}) => {
  AoiAssessmentMapLayer.propTypes = {
    mapRef: PropTypes.object.isRequired,
    assessmentId: PropTypes.string,
    onBuildingClick: PropTypes.func.isRequired,
    refreshTrigger: PropTypes.number,
  };

  const dataSourceRef = useRef(null);
  const layersRef = useRef([]);
  // The parent passes a new callback on every render; read it through a
  // ref so the map layers are not torn down and rebuilt each time.
  const onBuildingClickRef = useRef(onBuildingClick);
  useEffect(() => {
    onBuildingClickRef.current = onBuildingClick;
  }, [onBuildingClick]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !assessmentId) return;

    let cancelled = false;
    const dataSource = new window.atlas.source.DataSource();
    dataSourceRef.current = dataSource;
    map.sources.add(dataSource);

    // Vector polygons for buildings (detailed zoom)
    const fillLayer = new window.atlas.layer.PolygonLayer(dataSource, null, {
      fillColor: [
        "match",
        ["get", "result_status"],
        "insufficient_data", RESULT_STATUS_COLORS.insufficient_data,
        "failed", RESULT_STATUS_COLORS.failed,
        "uncertain", RESULT_STATUS_COLORS.uncertain,
        // completed: colour by damage class (same ramp as the DTF rasters)
        [
          "match",
          ["get", "damage_class"],
          ...DTF_CLASS_COLORS.flatMap((color, idx) => [idx, color]),
          UNKNOWN_CLASS_COLOR,
        ],
      ],
      fillOpacity: 0.75,
      filter: ["any", ["==", ["geometry-type"], "Polygon"], ["==", ["geometry-type"], "MultiPolygon"]],
    });
    fillLayer.customId = "aoiAssessmentFillLayer";
    map.layers.add(fillLayer);

    const outlineLayer = new window.atlas.layer.LineLayer(dataSource, null, {
      strokeColor: "#222222",
      strokeWidth: 1.5,
      filter: ["any", ["==", ["geometry-type"], "Polygon"], ["==", ["geometry-type"], "MultiPolygon"]],
    });
    outlineLayer.customId = "aoiAssessmentOutlineLayer";
    map.layers.add(outlineLayer);

    // Bubble layer for aggregated clusters when zoomed out
    const bubbleLayer = new window.atlas.layer.BubbleLayer(dataSource, null, {
      color: "#0078d4",
      radius: [
        "interpolate",
        ["linear"],
        ["get", "point_count"],
        1, 6,
        50, 14,
        500, 22
      ],
      opacity: 0.8,
      strokeColor: "#ffffff",
      strokeWidth: 2,
      filter: ["any", ["==", ["geometry-type"], "Point"], ["==", ["geometry-type"], "MultiPoint"]],
    });
    bubbleLayer.customId = "aoiAssessmentBubbleLayer";
    map.layers.add(bubbleLayer);

    layersRef.current = [fillLayer, outlineLayer, bubbleLayer];

    async function fetchMapData() {
      if (cancelled || !map) return;
      try {
        const camera = map.getCamera();
        const bounds = camera.bounds;
        const zoom = Math.round(camera.zoom);
        const bboxStr = bounds ? bounds.join(",") : "";
        const url = `GetAoiAssessmentMapData?assessmentId=${assessmentId}&bbox=${bboxStr}&zoom=${zoom}`;
        const data = await apiGet(url);

        if (!cancelled && data && data.features) {
          dataSource.clear();
          dataSource.add(data);
        }
      } catch (err) {
        if (!cancelled) {
          console.error("Error fetching assessment map data:", err);
        }
      }
    }

    const handleMoveEnd = () => {
      fetchMapData();
    };

    const handleClick = (e) => {
      if (!e.shapes || !e.shapes.length) return;
      // Reads the stored result only; never starts inference.
      const properties = e.shapes[0].getProperties();
      if (properties && (properties.building_id || properties.id)) {
        onBuildingClickRef.current(properties);
      }
    };

    map.events.add("moveend", handleMoveEnd);
    map.events.add("click", fillLayer, handleClick);

    // Initial load
    fetchMapData();

    return () => {
      cancelled = true;
      if (map && map.events) {
        map.events.remove("moveend", handleMoveEnd);
        map.events.remove("click", fillLayer, handleClick);
      }
      layersRef.current.forEach((layer) => {
        try {
          map.layers.remove(layer);
        } catch {
          // layer already removed with the map
        }
      });
      layersRef.current = [];
      try {
        map.sources.remove(dataSource);
      } catch {
        // source already removed with the map
      }
      dataSourceRef.current = null;
    };
  }, [mapRef, assessmentId, refreshTrigger]);

  return null;
};

export default AoiAssessmentMapLayer;
