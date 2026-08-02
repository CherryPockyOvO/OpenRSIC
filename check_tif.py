#!/usr/bin/env python3
"""Utility script to inspect bit-depth, channels, and GeoTIFF coordinate system information of any TIF image.

Usage:
    python check_tif.py path/to/image.tif
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def check_tif_info(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print("=" * 65)
    print(f"🔍 Inspecting TIF File: {path.name}")
    print("=" * 65)

    # 1. Inspect Raw Image Stats & Bit Depth
    if cv2 is not None:
        raw_arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        raw_arr = None

    if raw_arr is None:
        try:
            img_pil = Image.open(str(path))
            raw_arr = np.array(img_pil)
        except Exception as e:
            print(f"❌ Failed to open image: {e}")
            return

    print(f"  • Data Type (Bit Depth): {raw_arr.dtype}")
    print(f"  • Image Shape (H, W, C): {raw_arr.shape}")
    print(f"  • Min Pixel Value:       {raw_arr.min()}")
    print(f"  • Max Pixel Value:       {raw_arr.max()}")

    # 2. Inspect GeoTIFF Coordinate Tags
    print("-" * 65)
    try:
        img = Image.open(str(path))
        tags = getattr(img, "tag_v2", getattr(img, "tag", {}))

        # GeoTIFF Standard Tags
        geo_tag_names = {
            33550: "ModelPixelScaleTag",
            33922: "ModelTiepointTag",
            34735: "GeoKeyDirectoryTag",
            34736: "GeoDoubleParamsTag",
            34737: "GeoAsciiParamsTag",
        }

        found_geotags = {geo_tag_names[tag]: tags[tag] for tag in geo_tag_names if tag in tags}

        if found_geotags:
            print("  ✅ GeoTIFF Coordinates Found! (文件頭包含地理坐標系統標籤):")
            for tag_name, tag_val in found_geotags.items():
                print(f"     - {tag_name}: {str(tag_val)[:80]}...")
        else:
            # Check for sidecar .tfw file
            tfw_path = path.with_suffix(".tfw")
            if tfw_path.exists():
                print(f"  ⚠️ 文件頭未包含 GeoTIFF 標籤，但發現外置座標檔: {tfw_path.name}")
            else:
                print("  ❌ 普通 TIF 檔（文件頭未包含 GeoTIFF 地理坐標系統標籤）")
    except Exception as e:
        print(f"  ⚠️ Error reading TIFF tags: {e}")

    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_tif.py <path_to_image.tif>")
        sys.exit(1)

    for target_path in sys.argv[1:]:
        check_tif_info(target_path)
