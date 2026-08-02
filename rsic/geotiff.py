#!/usr/bin/env python3
"""GeoTIFF Metadata Extractor & Injector for RSIC Neural Compression.

Extracts GeoTIFF tags (ModelPixelScaleTag, ModelTiepointTag, GeoKeyDirectoryTag)
into a tiny JSON header (~70 to 500 bytes) and re-injects them upon decompression.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, TiffImagePlugin

GEO_TAGS = (33550, 33922, 34735, 34736, 34737)


def extract_geotiff_metadata(path: str | Path) -> str:
    """Extract GeoTIFF coordinate tags into a compact JSON string (~70-500 bytes)."""
    path_str = str(path)
    try:
        img = Image.open(path_str)
        tags = getattr(img, "tag_v2", getattr(img, "tag", {}))
        meta_dict = {}
        for tag_id in GEO_TAGS:
            if tag_id in tags:
                val = tags[tag_id]
                if isinstance(val, (tuple, list)):
                    meta_dict[int(tag_id)] = list(val)
                else:
                    meta_dict[int(tag_id)] = val
        return json.dumps(meta_dict)
    except Exception:
        return "{}"


def inject_geotiff_metadata(image_pil: Image.Image, metadata_json: str) -> TiffImagePlugin.ImageFileDirectory_v2:
    """Reconstruct PIL TIFF ImageDirectory header with extracted GeoTIFF metadata."""
    info = TiffImagePlugin.ImageFileDirectory_v2()
    if not metadata_json:
        return info
    try:
        parsed = json.loads(metadata_json)
        for k, v in parsed.items():
            tag_id = int(k)
            info[tag_id] = tuple(v) if isinstance(v, list) else v
    except Exception:
        pass
    return info
