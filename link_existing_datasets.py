#!/usr/bin/env python3
"""Instantly create symlinks from existing server datasets (COCO2017, DIV2K, DOTA, ImageNet) into datasets2/train.

Zero download time, zero extra disk space usage!
"""

import argparse
import os
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def link_dataset_folder(src_dir: Path, target_dir: Path, max_samples: int | None = None) -> int:
    src_dir = Path(src_dir).resolve()
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"⚠️ Source directory does not exist: {src_dir}")
        return 0

    print(f"🔍 Searching for image files in: {src_dir}...")
    files = sorted(
        p for p in src_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_samples and max_samples > 0:
        files = files[:max_samples]

    linked_count = 0
    for file_path in files:
        # Create unique link name using parent folder name if needed
        link_name = f"{src_dir.name}_{file_path.name}"
        link_path = target_dir / link_name
        if not link_path.exists() and not link_path.is_symlink():
            try:
                link_path.symlink_to(file_path)
                linked_count += 1
            except Exception as e:
                print(f"Failed to link {file_path}: {e}")

    print(f"✅ Created {linked_count} symlinks from {src_dir.name} into {target_dir}")
    return linked_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Link existing server datasets into datasets2/train instantly.")
    parser.add_argument(
        "--source-dirs",
        "-s",
        nargs="+",
        required=True,
        help="Paths to existing dataset directories (e.g. /autodl-pub/COCO2017 /autodl-pub/DIV2K /autodl-pub/DOTA).",
    )
    parser.add_argument(
        "--target-dir",
        "-t",
        type=str,
        default="/root/autodl-tmp/datasets2/train",
        help="Target training directory. Default: /root/autodl-tmp/datasets2/train.",
    )
    parser.add_argument(
        "--max-samples",
        "-m",
        type=int,
        default=None,
        help="Optional max samples limit per source directory.",
    )
    args = parser.parse_args()

    total_linked = 0
    target_path = Path(args.target_dir)
    for src in args.source_dirs:
        total_linked += link_dataset_folder(Path(src), target_path, max_samples=args.max_samples)

    print(f"\n🎉 Finished! Total {total_linked} images linked to {target_path}")


if __name__ == "__main__":
    main()
