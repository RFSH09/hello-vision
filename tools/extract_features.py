"""Preprocess every downloaded order once: motion frames -> frozen DINOv2 features.

    uv run --extra vision python tools/extract_features.py \
        --records E:/hello-videos/orders.local.jsonl \
        --videos E:/hello-videos --out E:/hello-features

Resumable (existing ``.npz`` files are skipped). Prints speed and size.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from hello_vision.weak.features import DinoTokenEncoder, extract_session
from hello_vision.weak.records import load_records


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", default="facebook/dinov2-small")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    records = load_records(args.records)[: args.limit]
    todo = [r for r in records if not (args.out / f"{r.order_id}.npz").exists()]
    print(f"{len(records)} orders, {len(todo)} to process", flush=True)
    if not todo:
        return 0
    encoder = DinoTokenEncoder(args.backbone)
    start, frames, failed = time.perf_counter(), 0, []
    for n, r in enumerate(todo, start=1):
        try:
            videos = [args.videos / v for v in r.videos]
            frames += extract_session(videos, encoder, args.out / f"{r.order_id}.npz")
        except (OSError, ValueError) as exc:
            failed.append(f"{r.order_id}: {exc}")
        if n % 20 == 0 or n == len(todo):
            rate = (time.perf_counter() - start) / n
            size = sum(p.stat().st_size for p in args.out.glob("*.npz"))
            print(
                f"{n}/{len(todo)}: {rate:.1f}s per order, {frames / n:.0f} frames per order, "
                f"{size / 1e9:.2f} GB of features, "
                f"about {rate * (len(todo) - n) / 3600:.1f} h left",
                flush=True,
            )
    for line in failed[:20]:
        print("failed", line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
