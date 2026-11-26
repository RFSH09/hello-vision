"""Use confirmed labels: build the real-crop gallery and measure the gain on unseen orders.

    uv run --extra vision python tools/apply_labels.py --labels data/labels.json

Evaluation is leave-one-order-out: when an order is scored, NO crop from that order is in the
gallery. Packshots-only and packshots+real-crops are scored on exactly the same crops.
Labelled crops are also copied to ``data/real_crops/<sku>/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hello_vision.capture.types import Image
from hello_vision.datatools.label_sheet import NOT_PRODUCT, UNSURE
from hello_vision.identify.embedders import Vector
from hello_vision.identify.index import GalleryIndex
from sim.orders import OrderScore, count_items, merge_cameras, report

NEGATIVE = "__not_product__"


def labelled_crops(tasks: list[dict[str, Any]], labels: dict[str, Any]) -> dict[str, list[str]]:
    """Task id -> usable crop paths (excluded crops removed) for labelled tasks."""
    excluded = set(labels.get("excluded", []))
    out: dict[str, list[str]] = {}
    for t in tasks:
        if labels["labels"].get(t["id"]) not in (None, UNSURE):
            out[t["id"]] = [c for c in t["crops"] if c not in excluded]
    return out


def vote(index: GalleryIndex, vectors: Sequence[Vector]) -> str | None:
    """Similarity-weighted vote over a track's crops; ``None`` = unknown."""
    weights: Counter[str] = Counter()
    unknown = 0.0
    for v in vectors:
        m = index.query(v)
        if m.sku is None:
            unknown += m.score
        else:
            weights[m.sku] += m.score  # type: ignore[assignment]
    if not weights:
        return None
    sku, weight = weights.most_common(1)[0]
    return None if unknown > weight else sku


def evaluate(
    tasks: list[dict[str, Any]],
    labels: dict[str, str],
    vectors: dict[str, list[Vector]],
    base: tuple[list[str], list[Vector]],
    tracks: dict[str, dict[str, Any]],
    orders: dict[str, Counter[str]],
    use_real: bool,
    threshold: float,
    margin: float,
) -> tuple[list[OrderScore], int, int]:
    """Leave-one-order-out scoring.

    Returns:
        Order scores, correctly named labelled items, and labelled product items.
    """
    scores: list[OrderScore] = []
    right = total = 0
    for order in sorted(orders):
        index = GalleryIndex(threshold, margin)
        for ref, v in zip(*base, strict=True):
            index.add(ref, v)
        if use_real:
            for t in tasks:
                label = labels.get(t["id"])
                if t["order"] != order and label not in (None, UNSURE):
                    for v in vectors.get(t["id"], []):
                        index.add(str(label), v)
        by_camera: dict[str, list[dict[str, Any]]] = {}
        for t in (t for t in tasks if t["order"] == order):
            sku = vote(index, vectors.get(t["id"], []))
            sku = None if sku == NEGATIVE else sku
            truth = labels.get(t["id"])
            if truth not in (None, UNSURE, NOT_PRODUCT):
                total += 1
                right += sku == truth
            info = tracks.get(t["id"])
            if info and sku:
                by_camera.setdefault(t["video"], []).append(
                    {**info, "sku": sku, "taken_guess": True}
                )
        cart = merge_cameras([count_items(v) for v in by_camera.values()])
        seen = {x["sku"] for v in by_camera.values() for x in v}
        scores.append(OrderScore(order, cart, orders[order], seen))
    return scores, right, total


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    from hello_vision.identify.gallery import DinoEmbedder
    from sim.run_video import load_models, read_orders

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, default=Path("out/labeling"))
    parser.add_argument("--results", type=Path, default=Path("out"))
    parser.add_argument("--orders", type=Path, default=Path("data/orders.csv"))
    parser.add_argument("--real-crops", type=Path, default=Path("data/real_crops"))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--margin", type=float, default=0.02)
    args = parser.parse_args(argv)

    tasks = json.loads((args.sheet / "tasks.json").read_text(encoding="utf-8"))
    data = json.loads(args.labels.read_text(encoding="utf-8"))
    usable = labelled_crops(tasks, data)
    labels: dict[str, str] = data["labels"]
    models = load_models()
    embedder: DinoEmbedder = models.embedder

    shutil.rmtree(args.real_crops, ignore_errors=True)
    vectors: dict[str, list[Vector]] = {}
    for t in tasks:
        paths = usable.get(t["id"], t["crops"])
        images: list[Image] = []
        for p in paths:
            bgr = cv2.imread(str(args.sheet / p))
            if bgr is not None:
                images.append(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8))
        vectors[t["id"]] = embedder.embed_batch(images) if images else []
        label = labels.get(t["id"])
        if label not in (None, UNSURE):
            folder = args.real_crops / str(label)
            folder.mkdir(parents=True, exist_ok=True)
            for p in paths:
                name = f"{t['video']}_{Path(p).name}"
                shutil.copy2(args.sheet / p, folder / name)

    tracks: dict[str, dict[str, Any]] = {}
    for f in args.results.glob("order-*.json"):
        for tr in json.loads(f.read_text(encoding="utf-8"))["tracks"]:
            tracks[f"{f.stem}:{tr['track_id']}"] = tr
    orders = read_orders(args.orders)
    refs, vecs = models.index.export()
    counts = Counter(labels.get(t["id"], "unlabelled") for t in tasks)
    print("labels:", dict(counts.most_common()))
    lines: list[str] = []
    for use_real in (False, True):
        name = "packshots + real crops" if use_real else "packshots only"
        scores, right, total = evaluate(
            tasks, labels, vectors, (refs, vecs), tracks, orders, use_real,
            args.threshold, args.margin,
        )  # fmt: skip
        exact = sum(s.exact for s in scores)
        correct = sum(s.correct for s in scores)
        wrong = sum(s.wrong for s in scores)
        actual = sum(sum(s.actual.values()) for s in scores)
        lines.append(
            f"{name:24s} item naming {right}/{total} | carts exact {exact}/{len(scores)} | "
            f"items correct {correct}/{actual}, wrong {wrong}"
        )
        if use_real:
            (args.results / "summary_real_gallery.md").write_text(report(scores), "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
