"""Turn a COCO detection set (with ``sku`` on product annotations) into per-SKU crops.

Usage: ``uv run python -m training.build_embedding_dataset --coco ann.json --images dir/
--out /private/crops``. Output: ``<out>/<sku>/<n>.png``; keep it outside the git repo.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image


def build(coco_path: Path, images_dir: Path, out_dir: Path, min_size: int = 16) -> dict[str, int]:
    """Crop every SKU-labelled product box and save it under ``out_dir/<sku>/``.

    Returns:
        Number of crops written per SKU. Boxes smaller than ``min_size`` px are skipped.
    """
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in data["images"]}
    counts: dict[str, int] = {}
    cache: dict[int, Image.Image] = {}
    for ann in data["annotations"]:
        sku = ann.get("sku")
        if not sku:
            continue
        x, y, w, h = ann["bbox"]
        if w < min_size or h < min_size:
            continue
        img_id = ann["image_id"]
        if img_id not in cache:
            cache.clear()  # keep one image in memory
            cache[img_id] = Image.open(images_dir / files[img_id]).convert("RGB")
        folder = out_dir / sku
        folder.mkdir(parents=True, exist_ok=True)
        n = counts.get(sku, 0)
        cache[img_id].crop((int(x), int(y), int(x + w), int(y + h))).save(folder / f"{n:05d}.png")
        counts[sku] = n + 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    counts = build(args.coco, args.images, args.out)
    print(f"{sum(counts.values())} crops, {len(counts)} SKUs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
