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


def setup_autodl_pub_coco() -> None:
    train_zip = find_public_zip("train2017.zip")
    val_zip = find_public_zip("val2017.zip")

    train_target = Path("/root/autodl-tmp/datasets2/train")
    val_target = Path("/root/autodl-tmp/datasets2/val")

    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)

    if train_zip and train_zip.exists():
        print(f"📦 Found local public zip: {train_zip}")
        print(f"🚀 Unzipping training images into {train_target}...")
        subprocess.run(["unzip", "-q", "-o", str(train_zip), "-d", str(train_target)], check=True)

        nested_train = train_target / "train2017"
        if nested_train.exists() and nested_train.is_dir():
            print(f"🚚 Flattening nested directory {nested_train.name} -> {train_target}...")
            for f in nested_train.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(train_target / f.name))
            shutil.rmtree(nested_train)
        print("✅ Training dataset unzipped and ready!")
    else:
        print(f"⚠️ train2017.zip not found in candidate public directories (/autodl-pub, /root/autodl-pub).")

    if val_zip and val_zip.exists():
        print(f"\n📦 Found local public zip: {val_zip}")
        print(f"🚀 Unzipping validation images into {val_target}...")
        subprocess.run(["unzip", "-q", "-o", str(val_zip), "-d", str(val_target)], check=True)

        nested_val = val_target / "val2017"
        if nested_val.exists() and nested_val.is_dir():
            print(f"🚚 Flattening nested directory {nested_val.name} -> {val_target}...")
            for f in nested_val.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(val_target / f.name))
            shutil.rmtree(nested_val)
        print("✅ Validation dataset unzipped and ready!")
    else:
        print(f"⚠️ val2017.zip not found in candidate public directories (/autodl-pub, /root/autodl-pub).")

    print("\n🎉 All AutoDL public COCO2017 datasets check finished under /root/autodl-tmp/datasets2!")


if __name__ == "__main__":
    setup_autodl_pub_coco()
