"""Generate a synthetic detection dataset (COCO) from a gallery, or from demo placeholders.

Usage: ``uv run python tools/gen_synthetic.py --out /private/synth --count 500
[--gallery /private/gallery --catalog catalog.yaml]``. Output must stay outside the repo.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hello_vision.catalog.gallery import Gallery
from hello_vision.catalog.models import Catalog
from hello_vision.catalog.packshots import augment_packshot, placeholder_sprites
from hello_vision.synth.compose import generate_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args(argv)

    if args.gallery and args.catalog:
        import numpy as np

        rng = np.random.default_rng(args.seed)
        gallery = Gallery(args.gallery)
        sprites = {}
        for sku in Catalog.load(args.catalog).ids():
            images = gallery.load_rgba(sku)
            if images:
                sprites[sku] = augment_packshot(images[0], rng)
    else:
        sprites = placeholder_sprites()
    ann = generate_dataset(args.out, sprites, args.count, args.seed)
    print(f"wrote {args.count} images + {ann}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
