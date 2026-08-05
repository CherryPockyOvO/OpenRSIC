#!/usr/bin/env python3
"""Unzip AutoDL public COCO2017 dataset from /autodl-pub/COCO2017 directly into datasets2.

Zero internet download required! Uses local NVMe disk speed (500+ MB/s).
"""

import os
import shutil
import subprocess
from pathlib import Path


def find_public_zip(filename: str) -> Path | None:
    candidate_roots = [
        Path("/autodl-pub"),
        Path("/root/autodl-pub"),
        Path("/autodl-fs"),
        Path("/root/autodl-fs"),
    ]
    for root in candidate_roots:
        if root.exists():
            for p in root.rglob(filename):
                if p.is_file():
                    return p
    return None


def unzip_file(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Found local public zip: {zip_path}")
    print(f"🚀 Unzipping into {target_dir}...")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(target_dir)], check=True)

    # Flatten if unzipped into a subfolder
    subfolders = [d for d in target_dir.iterdir() if d.is_dir()]
    for sub in subfolders:
        if sub.name in ("train2017", "val2017", "DIV2K_train_HR", "DIV2K_valid_HR"):
            print(f"🚚 Flattening nested directory {sub.name} -> {target_dir}...")
            for f in sub.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(target_dir / f.name))
            shutil.rmtree(sub)
    print(f"✅ Extracted {zip_path.name} successfully!")


def setup_autodl_pub_coco() -> None:
    train_target = Path("/root/autodl-tmp/datasets2/train")
    val_target = Path("/root/autodl-tmp/datasets2/val")

    # 1. Search and unzip training zips
    found_train = False
    for train_name in ["DIV2K_train_HR.zip", "train2017.zip"]:
        z = find_public_zip(train_name)
        if z:
            unzip_file(z, train_target)
            found_train = True

    if not found_train:
        print("⚠️ No training zips (DIV2K_train_HR.zip / train2017.zip) found under /autodl-pub.")

    # 2. Search and unzip validation zips
    found_val = False
    for val_name in ["DIV2K_valid_HR.zip", "val2017.zip"]:
        z = find_public_zip(val_name)
        if z:
            unzip_file(z, val_target)
            found_val = True

    if not found_val:
        print("⚠️ No validation zips (DIV2K_valid_HR.zip / val2017.zip) found under /autodl-pub.")

    print("\n🎉 All AutoDL public dataset extraction finished under /root/autodl-tmp/datasets2!")


if __name__ == "__main__":
    setup_autodl_pub_coco()
