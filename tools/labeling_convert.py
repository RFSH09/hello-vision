"""Convert labeling-tool exports into our frame-label JSON.

Usage:
    uv run python tools/labeling_convert.py cvat --in annotations.json --out frames.json
    uv run python tools/labeling_convert.py label-studio --in export.json --out frames.json
Pre-annotations for CVAT come from ``hello_vision.datatools.labeling.preannotations`` +
``export_coco``. Labeling data lives in private storage, never in the repo.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from hello_vision.datatools.labeling import LabeledFrame, import_coco, import_label_studio


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["cvat", "label-studio"])
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    frames = import_coco(args.inp) if args.source == "cvat" else import_label_studio(args.inp)
    args.out.write_bytes(TypeAdapter(list[LabeledFrame]).dump_json(frames, indent=2))
    print(f"{len(frames)} frames, {sum(len(f.boxes) for f in frames)} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
