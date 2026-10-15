# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Test-session environment fixes.

A Windows PostgreSQL/PostGIS install exports PROJ_LIB / GDAL_DATA pointing
at its own (older) PROJ database, which makes pyproj/rasterio in the test
environment fail CRS lookups. Drop those entries for the test run only;
library code never touches these variables.
"""

import os

for _var in ("PROJ_LIB", "GDAL_DATA", "PROJ_DATA"):
    if "PostgreSQL" in os.environ.get(_var, ""):
        os.environ.pop(_var, None)
