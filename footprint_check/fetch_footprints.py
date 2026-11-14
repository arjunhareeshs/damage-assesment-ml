"""Download building footprints from an ArcGIS FeatureServer layer into a GeoPackage.

Pages through the layer (2,000 features per request), optionally limited to a
bounding box in longitude/latitude, and writes EPSG:4326 polygons with all
attributes.

  python fetch_footprints.py --layer la --out data/la_county_buildings_2023_dins.gpkg
  python fetch_footprints.py --layer nc --bbox -82.7 35.4 -82.2 35.8 --out data/nc_helene_core.gpkg
"""
import argparse, json, time, urllib.parse, urllib.request
import geopandas as gpd

LAYERS = {
    "la": "https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/Buildings_2023_with_DINS/FeatureServer/4",
    "nc": "https://services1.arcgis.com/YBWrN5qiESVpqi92/ArcGIS/rest/services/North_Carolina_Composite_Building_Footprints/FeatureServer/0",
    "bay": "https://gis.baycountyfl.gov/arcgis/rest/services/Hosted/FC_BuildingFootprints/FeatureServer/22",
}


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, help="la | nc | bay, or a FeatureServer layer URL")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = LAYERS.get(a.layer, a.layer)
    oid = get(url + "?f=json").get("objectIdField") or "OBJECTID"
    base = {"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson", "orderByFields": oid}
    if a.bbox:
        base.update({"geometry": ",".join(map(str, a.bbox)), "geometryType": "esriGeometryEnvelope",
                     "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
    feats, off = [], 0
    while True:
        d = get(url + "/query?" + urllib.parse.urlencode({**base, "resultOffset": off, "resultRecordCount": 2000}))
        page = d.get("features", [])
        feats += page
        off += len(page)
        print(f"\r{off} features", end="", flush=True)
        if not page:   # some servers drop the exceededTransferLimit flag early, so page until empty
            break
    print()
    g = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    g = g[g.geometry.notna()]
    oid_col = next((c for c in g.columns if c.lower() == oid.lower()), None)
    if oid_col:
        g = g.drop_duplicates(subset=oid_col)
    g.to_file(a.out, driver="GPKG")
    print(f"wrote {len(g)} footprints -> {a.out}")


if __name__ == "__main__":
    main()
