
from pathlib import Path
from urllib.parse import urljoin
import json
import requests
import zipfile

ROOT = Path("data")

FOOTPRINTS = ROOT / "footprints"
IMAGERY = ROOT / "imagery" / "naip_2020"
METADATA = ROOT / "metadata"

for folder in (FOOTPRINTS, IMAGERY, METADATA):
    folder.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "DamageAssessmentDataDownloader/1.0"

def download(url, destination):
    """Stream a download to disk without holding it in memory."""
    destination = Path(destination)
    temp = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination}")
        return

    print(f"\nDownloading: {destination.name}")

    with SESSION.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with open(temp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    temp.replace(destination)
    print(f"Saved: {destination}")


# --------------------------------------------------
# 1. MICROSOFT TEXAS BUILDING FOOTPRINTS
# --------------------------------------------------

footprint_url = (
    "https://minedbuildings.z5.web.core.windows.net/"
    "legacy/usbuildings-v2/Texas.geojson.zip"
)

archive = FOOTPRINTS / "Texas.geojson.zip"
geojson = FOOTPRINTS / "Texas.geojson"

download(footprint_url, archive)

if not geojson.exists():
    print("\nExtracting Texas building footprints...")
    with zipfile.ZipFile(archive) as z:
        members = [
            name for name in z.namelist()
            if name.lower().endswith(".geojson")
        ]

        if not members:
            raise RuntimeError("No GeoJSON found in Texas archive.")

        # Stream extraction instead of loading the whole file into RAM.
        with z.open(members[0]) as src, open(geojson, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

    print(f"Footprints extracted: {geojson}")

# Remove archive after successful extraction to save disk space.
if geojson.exists() and archive.exists():
    archive.unlink()


# --------------------------------------------------
# 2. USDA NAIP 2020 TEXAS IMAGERY
# --------------------------------------------------

base_url = (
    "https://coast.noaa.gov/htdata/raster7/imagery/"
    "TX_NAIP_2020_9520/"
)

url_list = urljoin(base_url, "urllist9520.txt")

print("\nReading NOAA imagery URL index...")
response = SESSION.get(url_list, timeout=60)
response.raise_for_status()

lines = response.text.splitlines()

# The first filename was abbreviated in the project notes.
# Match its prefix; use the second filename exactly.
targets = {
    "tile_3510133": "m_3510133",
    "tile_3510141": "m_3510141_ne_14_060_20200915.tif",
}

urls = {}

for line in lines:
    line = line.strip().strip('"')
    if not line or line.startswith("#"):
        continue

    candidate = urljoin(base_url, line)
    filename = candidate.split("?")[0].rsplit("/", 1)[-1]

    if filename.startswith(targets["tile_3510133"]) and filename.lower().endswith(".tif"):
        urls.setdefault("tile_3510133", candidate)

    if filename == targets["tile_3510141"]:
        urls["tile_3510141"] = candidate

missing = set(targets) - set(urls)

if missing:
    raise RuntimeError(
        "Could not find imagery in NOAA's URL list: "
        + ", ".join(missing)
        + ". Check the URL index and exact tile filenames."
    )

for key, url in urls.items():
    filename = url.split("?")[0].rsplit("/", 1)[-1]
    download(url, IMAGERY / filename)

# Record source and file inventory.
manifest = {
    "footprints": {
        "source": footprint_url,
        "local_file": str(geojson),
        "dataset": "Microsoft US Building Footprints - Texas",
    },
    "imagery": [
        {"source": url, "local_file": str(
            IMAGERY / url.split("?")[0].rsplit("/", 1)[-1]
        )}
        for url in urls.values()
    ],
}

with open(METADATA / "download_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nAll downloads completed.")
print(f"Data directory: {ROOT.resolve()}")
