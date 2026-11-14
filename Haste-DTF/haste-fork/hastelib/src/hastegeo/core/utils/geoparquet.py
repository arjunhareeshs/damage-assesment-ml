# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Write GeoParquet 1.0 files (WKB geometry, EPSG:4326) with pyarrow only."""

from __future__ import annotations

import json
import os
from typing import Dict, List

def _wgs84_projjson() -> dict:
    """Complete PROJJSON for EPSG:4326.

    A hand-written stub (name + id only, no datum) used to be embedded here;
    pyproj -- and so geopandas, QGIS and GDAL -- rejects it ("Missing
    datum_ensemble"), which made the files' CRS unreadable.
    """
    from pyproj import CRS

    return CRS.from_epsg(4326).to_json_dict()


def write_geoparquet(path: str, columns: Dict[str, object], geometries_wkb: List[bytes]) -> int:
    """Write ``columns`` plus a WKB ``geometry`` column as GeoParquet.

    ``columns`` maps names to lists or ``pyarrow.Array`` of equal length to
    ``geometries_wkb``. The file is written to ``path + ".tmp"`` and moved
    into place, so readers never see a half-written file. Returns the row
    count.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely

    bounds = [shapely.from_wkb(g).bounds for g in geometries_wkb]
    bbox = (
        [
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        ]
        if bounds
        else None
    )
    table = pa.table({**columns, "geometry": [bytes(g) for g in geometries_wkb]})
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon", "MultiPolygon"],
                "crs": _wgs84_projjson(),
                **({"bbox": bbox} if bbox else {}),
            }
        },
    }
    table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return table.num_rows
