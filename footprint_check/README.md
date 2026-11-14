# Building footprints for the three DamageTriage-Bench events

Downloaded 2026-10-27 from the public ArcGIS services of the agencies cited in the paper. All files are in EPSG:4326 (longitude/latitude) with every attribute the services provide.

## Contents

| File | Source | Coverage | Features |
|---|---|---|---|
| `data/la_county_buildings_2023_dins.gpkg` | Los Angeles County Enterprise GIS, "2023 Buildings with DINS data" (building polygons with CAL FIRE damage-inspection attributes) | entire layer: Palisades and Eaton fire areas | 45,433 |
| `data/bay_county_fl_buildings.gpkg` | Bay County, Florida GIS, "Building Footprints" (`Hosted/FC_BuildingFootprints`) | entire county | 117,646 |
| `data/nc_composite_buildings_helene_core.gpkg` | NC Emergency Management, "North Carolina Composite Building Footprints" | bounding box −82.7, 35.4, −82.2, 35.8 (Asheville and Swannanoa area) only | 133,869 |
| `data/check_areas/*.geojson` | the same three services | the six ~320 m areas shown in the overlays | 16–93 each |
| `*.jpg` | footprints (cyan) over NOAA Emergency Response Imagery web tiles (zoom 19, about 0.25 m per pixel) | the six check areas | |

Service URLs:
- LA County: `https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/Buildings_2023_with_DINS/FeatureServer/4`
- Bay County: `https://gis.baycountyfl.gov/arcgis/rest/services/Hosted/FC_BuildingFootprints/FeatureServer/22`
- NC: `https://services1.arcgis.com/YBWrN5qiESVpqi92/ArcGIS/rest/services/North_Carolina_Composite_Building_Footprints/FeatureServer/0`

NOAA imagery tiles: `https://stormscdn.ngs.noaa.gov/<flight>/<z>/<x>/<y>`, flights `20181011a` (Michael), `20241005a` and `20241006a` (Helene), `20250128a` (LA fires).

## Scripts

- `fetch_footprints.py`: download a layer (or a bounding box of it) to GeoPackage, for example `python fetch_footprints.py --layer nc --bbox W S E N --out out.gpkg`. The full NC layer is statewide (about 5.4 million buildings), so download it by bounding box.
- `save_check_areas.py`: save the footprints of the six check areas.
- `overlay.py`: download the NOAA tiles for the six areas and draw the overlays.

All three need the project Python environment (`/work/nvme/bhhj/yxiao11/envs/dinov3/bin/python`).

## Caveats

- These are the current versions of the layers, not necessarily the snapshots used to build DamageTriage-Bench (the paper cites a 2019 Bay County layer). The benchmark's own labels exist only as rasterized per-tile masks without georeference.
- The current Bay County layer omits many buildings destroyed by Hurricane Michael (see `michael_mexico_beach.jpg`), so it undercounts the most damaged buildings. For a new event, use a footprint layer from before the event.
- Visual check of the overlays: alignment is very good in Los Angeles and good in Panama City. On the wooded hillsides of Asheville several polygons are offset by about 3–5 m, and the NC and Bay County layers miss some buildings (sheds, some commercial buildings).
- Each layer is subject to its publisher's terms of use.
