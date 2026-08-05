#!/usr/bin/env python3
"""Unzip AutoDL public COCO2017 dataset from /autodl-pub/COCO2017 directly into datasets2.

Zero internet download required! Uses local NVMe disk speed (500+ MB/s).
"""

import os
import shutil
import subprocess
from pathlib import Path


def setup_autodl_pub_coco() -> None:
    pub_dir = Path("/autodl-pub/COCO2017")
    train_zip = pub_dir / "train2017.zip"
    val_zip = pub_dir / "val2017.zip"

    train_target = Path("/root/autodl-tmp/datasets2/train")
    val_target = Path("/root/autodl-tmp/datasets2/val")

    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)

    if train_zip.exists():
        print(f"📦 Found local public zip: {train_zip}")
        print(f"🚀 Unzipping 118,287 training images into {train_target}...")
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
        print(f"⚠️ {train_zip} not found in public directory.")

    if val_zip.exists():
        print(f"\n📦 Found local public zip: {val_zip}")
        print(f"🚀 Unzipping 5,000 validation images into {val_target}...")
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
        print(f"⚠️ {val_zip} not found in public directory.")

    print("\n🎉 All AutoDL public COCO2017 datasets are ready under /root/autodl-tmp/datasets2!")


if __name__ == "__main__":
    setup_autodl_pub_coco()
