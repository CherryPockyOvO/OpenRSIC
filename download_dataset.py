#!/usr/bin/env python3
"""Fast Direct Downloader for High-Quality RGB Image Datasets (COCO 2017 / DIV2K / CLIC) on AutoDL.

Uses single high-speed zip stream downloads (100~150 MB/s with AutoDL academic turbo).
Extracts images directly into /root/autodl-tmp/datasets2/train.
"""

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

DATASET_SOURCES = {
    "coco2017": {
        "name": "COCO 2017 Train (118,287 RGB Images, ~18 GB)",
        "url": "http://images.cocodataset.org/zips/train2017.zip",
    },
    "div2k": {
        "name": "DIV2K 2K High Resolution (800 2K RGB Images, ~3.5 GB)",
        "url": "https://huggingface.co/datasets/eugenesiow/Div2k/resolve/main/data/DIV2K_train_HR.zip",
    },
}


def download_dataset(source_key: str = "coco2017", export_dir: str = "/root/autodl-tmp/datasets2/train") -> None:
    if source_key not in DATASET_SOURCES:
        raise ValueError(f"Unknown source_key: {source_key}. Choices: {list(DATASET_SOURCES.keys())}")

    info = DATASET_SOURCES[source_key]
    url = info["url"]
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    zip_name = Path(url).name
    temp_zip = Path("/root/autodl-tmp") / zip_name

    print(f"\n🚀 Selected Source: {info['name']}")
    print(f"📥 Downloading ZIP file from: {url}")
    print("💡 Tip: Make sure you ran 'source /etc/network_turbo' for 100+ MB/s download speeds!")

    cmd = ["wget", "-c", "-O", str(temp_zip), url]
    subprocess.run(cmd, check=True)

    print(f"\n📦 Extracting {zip_name} to {export_path}...")
    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(export_path)

    # Flatten nested folder if unzipped into train2017/ or DIV2K_train_HR/
    for item in export_path.iterdir():
        if item.is_dir() and item.name in ("train2017", "DIV2K_train_HR", "val2017"):
            print(f"🚚 Flattening nested directory {item.name} into {export_path}...")
            for f in item.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(export_path / f.name))
            shutil.rmtree(item)

    print(f"\n✅ Completed! Dataset successfully extracted to: {export_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download High-Speed RGB Datasets for CompressAI Training.")
    parser.add_argument(
        "--source",
        "-s",
        choices=["coco2017", "div2k"],
        default="coco2017",
        help="Dataset source: 'coco2017' (118k images) or 'div2k' (800 2K images). Default: coco2017.",
    )
    parser.add_argument(
        "--export-dir",
        "-o",
        type=str,
        default="/root/autodl-tmp/datasets2/train",
        help="Output target directory. Default: /root/autodl-tmp/datasets2/train.",
    )
    args = parser.parse_args()
    download_dataset(args.source, args.export_dir)


if __name__ == "__main__":
    main()
