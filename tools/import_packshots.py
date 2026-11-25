"""Import packshots (``<sku>.png|jpg``) into a gallery with background removal.

Usage: ``uv run python tools/import_packshots.py --catalog catalog.yaml --src packshots/
--gallery /private/gallery``. Packshot and gallery folders must live outside the git repo.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hello_vision.catalog.gallery import Gallery
from hello_vision.catalog.models import Catalog
from hello_vision.catalog.packshots import import_packshots


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exits 1 if any catalog SKU has no packshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    args = parser.parse_args(argv)
    report = import_packshots(args.src, Catalog.load(args.catalog), Gallery(args.gallery))
    print(f"imported: {len(report.imported)}")
    print(f"missing:  {report.missing}")
    print(f"unmatched files: {report.unmatched}")
    return 1 if report.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
