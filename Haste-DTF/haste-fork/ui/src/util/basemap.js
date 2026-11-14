// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Optional background basemap for deployments without Azure Maps.
//
// OFF by default: the maps show only the analyst's imagery (and its
// buildings), kept in view by the Labeling Tool's camera bounds. A world
// basemap around the imagery was found distracting, so it is opt-in.
//
// To enable it, set VITE_BASEMAP_TILE_URL to an XYZ tile URL with
// {z}/{x}/{y} placeholders (e.g. Esri World Imagery,
// https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x})
// and VITE_BASEMAP_ATTRIBUTION to the credit that provider requires.
// It is only used when Azure Maps isn't configured.
import { isAzureMapsPlaceholder } from "./azureMapsAuth";

const configuredUrl = import.meta.env.VITE_BASEMAP_TILE_URL;

export const OPEN_BASEMAP_URL =
  configuredUrl && configuredUrl !== "none" ? configuredUrl : "";

const ATTRIBUTION = import.meta.env.VITE_BASEMAP_ATTRIBUTION || "";

function addAttribution(map) {
  if (!ATTRIBUTION) return;
  const container = map.getMapContainer();
  if (!container || container.querySelector(".haste-basemap-attribution")) return;
  const credit = document.createElement("div");
  credit.className = "haste-basemap-attribution";
  credit.textContent = ATTRIBUTION;
  Object.assign(credit.style, {
    position: "absolute",
    left: "4px",
    bottom: "2px",
    padding: "0 4px",
    fontSize: "10px",
    color: "#333",
    background: "rgba(255, 255, 255, 0.7)",
    pointerEvents: "none",
    zIndex: 1,
  });
  container.appendChild(credit);
}

/**
 * Adds the open basemap as the bottom-most layer of `map` (call it before
 * adding imagery). Returns the layer, or null when Azure Maps provides the
 * basemap or the basemap is disabled.
 */
export function addOpenBasemap(map, customId = "openBasemap") {
  if (!isAzureMapsPlaceholder || !OPEN_BASEMAP_URL) return null;
  const layer = new window.atlas.layer.TileLayer({
    tileUrl: OPEN_BASEMAP_URL,
    tileSize: 256,
    maxSourceZoom: 19,
  });
  layer.customId = customId;
  try {
    map.layers.add(layer);
    addAttribution(map);
  } catch (error) {
    console.error("Error adding the open basemap:", error);
    return null;
  }
  return layer;
}
