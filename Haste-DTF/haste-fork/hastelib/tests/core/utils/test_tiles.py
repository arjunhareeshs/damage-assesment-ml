# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from hastegeo.core.utils.tiles import mask_nodata

T = "http://t/cog/tiles/WebMercatorQuad/{z}/{x}/{y}"


class TestMaskNodata(unittest.TestCase):
    def test_appends_to_tile_url(self):
        self.assertEqual(mask_nodata(T + "?scale=1&url=a"), T + "?scale=1&url=a&nodata=0")
        self.assertEqual(mask_nodata(T), T + "?nodata=0")

    def test_leaves_other_urls_alone(self):
        for u in ("", None, "https://atlas.microsoft.com/map/tile?x=1",
                  T + "?url=a&nodata=255"):
            self.assertEqual(mask_nodata(u), u)

    def test_idempotent(self):
        once = mask_nodata(T + "?url=a")
        self.assertEqual(mask_nodata(once), once)


if __name__ == "__main__":
    unittest.main()
