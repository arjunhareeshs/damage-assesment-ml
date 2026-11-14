"""Save the footprints of the six overlay areas (same extents as overlay.py) as GeoJSON."""
import json, math, os, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "overlay.py")).read()
ns = {}
exec(src[src.index("Z, N ="):src.index("def get(")], ns)          # Z, N, FP, SITES
Z, N, FP, SITES = ns["Z"], ns["N"], ns["FP"], ns["SITES"]


def tile_xy(lon, lat, z):
    n = 2 ** z
    return ((lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def lonlat(x, y, z):
    n = 2 ** z
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


os.makedirs(os.path.join(HERE, "data", "check_areas"), exist_ok=True)
for name, s, layers, lon, lat in SITES:
    cx, cy = tile_xy(lon, lat, Z)
    x0, y0 = int(cx) - N // 2, int(cy) - N // 2
    w, n = lonlat(x0, y0, Z); e, so = lonlat(x0 + N, y0 + N, Z)
    q = {"geometry": f"{w},{so},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
         "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326, "f": "geojson"}
    req = urllib.request.Request(FP[s] + "/query?" + urllib.parse.urlencode(q), headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    d["bbox_lonlat"] = [w, so, e, n]
    json.dump(d, open(os.path.join(HERE, "data", "check_areas", f"{name}.geojson"), "w"))
    print(name, len(d["features"]), [round(v, 5) for v in (w, so, e, n)])
