"""Build weak-label eval sessions from exported purchase history (see datatools.weak_labels).

Usage: ``uv run python tools/weak_labels.py --records purchases.jsonl --catalog catalog.yaml
--videos /private/videos --out /private/weak_sessions``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hello_vision.catalog.models import Catalog
from hello_vision.datatools.weak_labels import build_weak_sessions


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exits 2 if the export contains personal-looking fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rep = build_weak_sessions(args.records, Catalog.load(args.catalog), args.videos, args.out)
    except ValueError as exc:
        print(f"refused: {exc}")
        return 2
    print(f"accepted: {len(rep.accepted)}")
    for reason, ids in sorted(rep.rejected.items()):
        print(f"rejected ({reason}): {len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
