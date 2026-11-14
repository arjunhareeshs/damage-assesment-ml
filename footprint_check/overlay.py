"""Overlay official footprint polygons on NOAA ERI tiles for a few small areas."""
import io, json, math, os, sys, urllib.parse, urllib.request
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
Z, N = 19, 5          # zoom 19 (~0.26 m/px at 30N), 5x5 tiles (~1280 px)
FP = {
    "la": "https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/Buildings_2023_with_DINS/FeatureServer/4",
    "nc": "https://services1.arcgis.com/YBWrN5qiESVpqi92/ArcGIS/rest/services/North_Carolina_Composite_Building_Footprints/FeatureServer/0",
    "bay": "https://gis.baycountyfl.gov/arcgis/rest/services/Hosted/FC_BuildingFootprints/FeatureServer/22",
}
SITES = [  # name, footprint source, NOAA layers to try, lon, lat
    ("michael_panama_city", "bay", ["20181011a-rgb", "20181012a-rgb"], -85.575, 30.125),
    ("michael_mexico_beach", "bay", ["20181011a-rgb", "20181012a-rgb", "20181012b-rgb"], -85.414, 29.948),
    ("helene_swannanoa", "nc", ["20241005a-rgb", "20241006a-rgb", "20241007a-rgb"], -82.399, 35.601),
    ("helene_asheville_rad", "nc", ["20241005a-rgb", "20241006a-rgb", "20241007a-rgb"], -82.570, 35.586),
    ("lafire_altadena", "la", ["20250128a-rgb"], -118.135, 34.190),
    ("lafire_palisades", "la", ["20250128a-rgb"], -118.531, 34.046),
]

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()

def tile_xy(lon, lat, z):
    n = 2 ** z
    return ((lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

def lonlat(x, y, z):
    n = 2 ** z
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))

def mosaic(layer, x0, y0):
    img = Image.new("RGB", (256 * N, 256 * N)); filled = 0
    for i in range(N):
        for j in range(N):
            p = os.path.join(HERE, "tiles", f"{layer}_{Z}_{x0+i}_{y0+j}.png")
            if not os.path.exists(p):
                try: open(p, "wb").write(get(f"https://stormscdn.ngs.noaa.gov/{layer}/{Z}/{x0+i}/{y0+j}"))
                except Exception: continue
            try:
                t = Image.open(p).convert("RGB")
            except Exception:
                continue
            if t.size == (256, 256) and t.getextrema() != ((0, 0), (0, 0), (0, 0)):
                img.paste(t, (256 * i, 256 * j)); filled += 1
    return img, filled

def footprints(src, w, s, e, n):
    q = {"geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
         "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326, "f": "geojson",
         "resultRecordCount": 2000}
    return json.loads(get(FP[src] + "/query?" + urllib.parse.urlencode(q)))["features"]

for name, src, layers, lon, lat in SITES:
    cx, cy = tile_xy(lon, lat, Z)
    x0, y0 = int(cx) - N // 2, int(cy) - N // 2
    w, n = lonlat(x0, y0, Z); e, s = lonlat(x0 + N, y0 + N, Z)
    img, used = None, None
    for L in layers:
        im, filled = mosaic(L, x0, y0)
        if filled >= N * N * 0.8:
            img, used = im, L; break
    if img is None:
        print(f"{name}: no imagery coverage in {layers}"); continue
    feats = footprints(src, w, s, e, n)
    d = ImageDraw.Draw(img)
    for f in feats:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ring = [((tile_xy(a, b, Z)[0] - x0) * 256, (tile_xy(a, b, Z)[1] - y0) * 256) for a, b in poly[0]]
            d.line(ring + [ring[0]], fill=(0, 255, 255), width=2)
    mpp = 156543.03 * math.cos(math.radians(lat)) / 2 ** Z
    out = os.path.join(HERE, "out", f"{name}.jpg")
    img.save(out, quality=88)
    print(f"{name}: layer {used}, {len(feats)} footprints, {256*N}px at {mpp:.2f} m/px ({256*N*mpp:.0f} m wide) -> {out}")
