// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { apiPut } from "../../util/api";
import settings from "../../assets/json/settings.json";
import { getAzureMapsAuthOptions, isAzureMapsPlaceholder } from "../../util/azureMapsAuth";

// AOI-scoped DamageTriageFormer (Option B, AOI-scoped —
// docs/OptionB_AOI_Integration_Plan.md Module 5). "draw-rectangle" is not
// used by any other DrawingToolbar button, so it's free to repurpose
// exclusively for area-of-interest capture: an AOI is a spatial filter,
// not a damage label, so it must never enter the labeling shape source
// that createShape()/saveLabels() operate on. Shared between
// LabelingTool.jsx (which diverts the drawingcomplete handler for this
// mode) and DrawingToolbar.jsx (which offers the toolbar button) so the
// two never drift out of sync.
export const AOI_DRAWING_MODE = "draw-rectangle";
export const AOI_POLYGON_MODE = "draw-polygon-aoi";

// DamageTriageFormer 5-class damage classification scale and colors
export const DTF_CATEGORIES = [
  "Undamaged",
  "Partial Roof Damage",
  "Total Roof Damage",
  "Partial Structural Collapse",
  "Total Structural Collapse",
];

export const DTF_CLASS_COLORS = [
  "#ffffff", // 0: Undamaged
  "#fcbea5", // 1: Partial Roof Damage
  "#fb7050", // 2: Total Roof Damage
  "#d32020", // 3: Partial Structural Collapse
  "#67000d", // 4: Total Structural Collapse
];

export const RESULT_STATUS_COLORS = {
  completed: "#4caf50",
  uncertain: "#ff9800",
  insufficient_data: "#9e9e9e",
  failed: "#e53935",
};

export function createShape(drawingManager, selectedPrimaryClass, setDrawingCount) {
  const source = drawingManager.getSource();
  const shapes = source.getShapes();
  const lastShape = shapes[shapes.length - 1];

  if (!lastShape) return;

  const props = {
    ...lastShape.getProperties?.(),
    primaryClass: selectedPrimaryClass,
    source: "Drawn|Imagery",
  };
  lastShape.setProperties(props);

  setDrawingCount(source.getShapes().length);
}

export function updateShape(drawingManager, selectedShape, selectedClass) {
  const source = drawingManager.getSource();
  const shape = source.getShapeById(selectedShape.getId());
  if (shape) {
    const props = {
      ...shape.getProperties?.(),
      primaryClass: selectedClass,
    };
    shape.setProperties(props);
  }
}

export const updateDrawingLayerStyles = (drawingManager, primaryClasses) => {
  const layers = drawingManager.getLayers();

  var colorExpression = [
    "match",
    ["get", "primaryClass"],
  ];

  var strokeColorExpression = [
    "match",
    ["get", "primaryClass"],
  ];

  primaryClasses.forEach((shapeType) => {
    colorExpression.push(shapeType.text);
    colorExpression.push(shapeType.color);

    strokeColorExpression.push(shapeType.text);
    strokeColorExpression.push(shapeType.color);
  });

  colorExpression.push("#FFFFFF");
  strokeColorExpression.push("#FF0000");

  layers.polygonLayer.setOptions({
    fillColor: colorExpression,
  });

  layers.polygonOutlineLayer.setOptions({
    strokeColor: strokeColorExpression,
    strokeWidth: 3,
  });
};


export const layerTypeOptions = [
  {
    key: "azureMapsSatellite",
    text: "Azure Maps Satellite",
  },
  {
    key: "imagery",
    text: "Imagery",
  },
];


export function loadImagery(tileUrl, map, imageryRef, customId, isVisible) {

  var tempTileUrlPath = tileUrl;
  if (tempTileUrlPath === "") {
    if (isAzureMapsPlaceholder) {
      // No real Azure Maps key — skip fallback to avoid 401 errors.
      // The map background will be blank; only COG imagery will show.
      return;
    }
    tempTileUrlPath = `https://atlas.microsoft.com/map/tile?api-version=2.1&tilesetId=microsoft.imagery&zoom={z}&x={x}&y={y}`;
  }


  imageryRef.current = new window.atlas.layer.TileLayer({
    tileUrl: tempTileUrlPath,
  });

  try {
    imageryRef.current.setOptions({ visible: isVisible });
    imageryRef.current.customId = customId;
    map.layers.add(imageryRef.current);
  } catch (error) {
    console.error("Error loading imagery layer:", error);
  }
}

export function centrateMap(bbox, map, duration = 2500) {
  map.setCamera({
    bounds: bbox,
    padding: 0,
    type: "fly",
    pitch: 0,
    duration: duration,
    maxPitch: 0
  });
}

export function loadStudyArea(map, imageLayer) {

  // Create Data Source
  var dataSource = new window.atlas.source.DataSource();
  map.sources.add(dataSource);

  // Add data
  var geoJsonData = {
    "type": "FeatureCollection",
    "features": imageLayer.features
  };
  dataSource.add(geoJsonData);


  // Create linelayer to define workspace
  var lineLayer = new window.atlas.layer.LineLayer(dataSource, null, {
    strokeColor: settings.labelingToolSettings.grid.gridStrokeColor,
    strokeWidth: 4,
    strokeDashArray: [1.5, 1.5]
  });
  map.layers.add(lineLayer);

  return window.atlas.data.BoundingBox.fromData(geoJsonData);
}

export async function saveLabels(drawingManager, labelingToolDataRef, setIsLoading, setHasUnsavedChanges) {

  var labels = [];
  drawingManager.source.shapes.map((shape) => {
    labels.push(shape.data);
  });



  labelingToolDataRef.current.labels = labels;

  setIsLoading(true, "Saving Labels...");
  try {
    await apiPut("PutLabelsFromLabelTool?imageLayerId", labelingToolDataRef.current);
    setHasUnsavedChanges(false);
    setIsLoading(false);
    return (true);
  } catch (error) {
    setIsLoading(false);
    return (false);
  }
}

export function checkLabelsState(drawingManager) {
  for (const shape of drawingManager.source.shapes) {
    const properties = shape.getProperties();
    const coordinates = shape.data.geometry.coordinates[0];

    var isADot = coordinates.length == 2 && coordinates[0][0] === coordinates[1][0] && coordinates[0][1] === coordinates[1][1];

    if (!properties.primaryClass && !isADot) {
      return false;
    }
  }
  return true;
}

export function parsePrimaryClasses(primaryClasses) {
  return primaryClasses.map((primaryClass) => {
    return {
      key: primaryClass.name,
      text: primaryClass.name,
      color: primaryClass.color,
      styles: {
        root: {
          selectors: {
            ".ms-ChoiceFieldLabel": {
              color: primaryClass.color
            },
          },
        },
      },
    };
  });
}