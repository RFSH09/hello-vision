"""Train the session model and write the first evaluation report.

    uv run --extra vision python tools/train_mil.py --records E:/hello-videos/orders.local.jsonl \
        --features E:/hello-features --videos E:/hello-videos --out out/mil --top 10 \
        --sizes 1000 3000 10000

Writes ``report.md`` (test metrics, learning curve), ``disagreements.csv`` and, for those
orders, the frames the model attended to (``evidence/``; private, never commit).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hello_vision.weak.metrics import evaluate_carts, to_markdown
from hello_vision.weak.records import OrderRecord, load_records
from hello_vision.weak.train import (
    REVIEW_LOW,
    TrainConfig,
    choose_classes,
    choose_threshold,
    evaluate,
    predict,
    target_cart,
    train,
)


def save_evidence(
    record: OrderRecord,
    attention: np.ndarray[Any, Any],
    classes: Sequence[str],
    wanted: Sequence[str],
    features: Path,
    videos: Path,
    out: Path,
    top: int = 3,
) -> None:
    """Save the crops with the highest attention for each class in ``wanted``."""
    with np.load(features / f"{record.order_id}.npz") as d:
        cams, times, boxes = d["cam"], d["time"], d["box"]
    for sku in wanted:
        if sku not in classes or not len(times):
            continue
        weights = attention[list(classes).index(sku)][: len(times)]
        for rank, k in enumerate(np.argsort(weights)[::-1][:top], start=1):
            cap = cv2.VideoCapture(str(videos / record.videos[int(cams[k])]))
            cap.set(cv2.CAP_PROP_POS_MSEC, float(times[k]) * 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                x1, y1, x2, y2 = (int(v) for v in boxes[k])
                out.mkdir(parents=True, exist_ok=True)
                name = f"{record.order_id}_{sku}_{rank}_{weights[k]:.2f}.jpg"
                cv2.imwrite(str(out / name), frame[y1:y2, x1:x2])


def test_title(train_records: Sequence[OrderRecord], test_records: Sequence[OrderRecord]) -> str:
    """Name the test set truthfully: whether its fridges were seen in training."""
    shared = {r.fridge_key for r in train_records} & {r.fridge_key for r in test_records}
    return (
        "Test (new days; fridges also seen in training)"
        if shared
        else ("Test (unseen fridges and days)")
    )


def less_seen_section(
    train_records: Sequence[OrderRecord],
    test_records: Sequence[OrderRecord],
    truth: dict[str, dict[str, int]],
    carts: dict[str, dict[str, int]],
    classes: Sequence[str],
    flagged: set[str],
    limit: int = 50,
) -> list[str]:
    """Test results for fridges with few training orders next to the other fridges."""
    seen = Counter(r.fridge_key for r in train_records)
    groups = {
        f"fridges with fewer than {limit} training orders": [
            r for r in test_records if seen[r.fridge_key] < limit
        ],
        f"fridges with {limit} or more": [r for r in test_records if seen[r.fridge_key] >= limit],
    }
    lines = [
        "",
        "## Less-seen fridges (a rough signal)",
        "",
        "| test group | fridges | orders | wrong-charge rate | review rate | exact (all) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, members in groups.items():
        if not members:
            lines.append(f"| {name} | 0 | 0 | - | - | - |")
            continue
        ids = [r.order_id for r in members]
        part = evaluate_carts({i: truth[i] for i in ids}, carts, classes, flagged & set(ids))
        wrong = f"{part.false_charge_rate:.1%}" if part.auto_sessions else "nothing charged"
        lines.append(
            f"| {name} | {len({r.fridge_key for r in members})} | {len(members)} | {wrong} | "
            f"{part.review_rate:.1%} | {part.exact:.1%} |"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--splits", type=Path, help="default: splits.json next to the records")
    parser.add_argument("--out", type=Path, default=Path("out/mil"))
    parser.add_argument("--top", type=int, default=10, help="classes besides 'other' (30 = all)")
    parser.add_argument("--sizes", type=int, nargs="*", default=[], help="learning-curve sizes")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--less-seen", type=int, default=50, help="training orders below which a fridge is rare"
    )
    args = parser.parse_args(argv)

    records = [
        r for r in load_records(args.records) if (args.features / f"{r.order_id}.npz").exists()
    ]
    splits = json.loads((args.splits or args.records.with_name("splits.json")).read_text("utf-8"))
    parts = {
        s: [r for r in records if splits.get(r.order_id) == s] for s in ("train", "val", "test")
    }
    # an order with a dropped product line has an incomplete cart: fine to learn from,
    # not fine to be judged on
    incomplete = 0
    for name in ("val", "test"):
        kept = [r for r in parts[name] if not r.skipped_lines]
        incomplete += len(parts[name]) - len(kept)
        parts[name] = kept
    for name, members in parts.items():
        if not members:
            raise SystemExit(f"the {name} split is empty")
        print(f"{name}: {len(members)} orders")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    full = len(parts["train"])
    sizes = sorted({min(s, full) for s in args.sizes} | {full})
    rng = np.random.default_rng(0)
    shuffled = [parts["train"][i] for i in rng.permutation(full)]  # nested subsets
    classes = choose_classes(parts["train"], args.top)  # same classes for every size
    curve: list[tuple[int, float, float]] = []
    model = None
    for size in sizes:
        print(f"--- training on {size} orders", flush=True)
        run_dir = args.out / f"n{size}"
        model = train(shuffled[:size], parts["val"], args.features, classes, run_dir, cfg)
        device = str(next(model.parameters()).device)
        result = evaluate(model, parts["test"], args.features, classes, device)
        curve.append((size, result.exact, result.false_charge_rate))
    assert model is not None  # noqa: S101
    device = str(next(model.parameters()).device)
    threshold, val_safe = choose_threshold(model, parts["val"], args.features, classes, device)
    plain = evaluate(model, parts["test"], args.features, classes, device)
    flagged: set[str] = set()
    carts, attention = predict(
        model, parts["test"], args.features, classes, device,
        threshold=threshold, review_low=REVIEW_LOW, flagged=flagged,
    )  # fmt: skip
    truth = {r.order_id: target_cart(r.items, classes) for r in parts["test"]}
    final = evaluate_carts(truth, carts, classes, flagged)
    baseline = sum(target_cart(r.items, classes) == {classes[0]: 1} for r in parts["test"]) / len(
        parts["test"]
    )
    report = [
        "# Session model: first evaluation",
        "",
        to_markdown(final, test_title(parts["train"], parts["test"])),
    ]
    report += [
        "## Charging rule",
        "",
        f"An item is charged only at probability >= {threshold}. On validation data this "
        + (
            f"charged {val_safe.auto_sessions} of {val_safe.sessions} orders with a "
            f"wrong-charge rate of {val_safe.false_charge_rate:.1%}. "
            if val_safe.auto_sessions
            else "charged NOTHING at any threshold, so no safe threshold exists yet. "
        )
        + "An order with any "
        f"probability from {REVIEW_LOW} up to that threshold goes to review.",
        f"For comparison, charging everything at 0.5 with no review gives a wrong-charge rate "
        f"of {plain.false_charge_rate:.1%} and {plain.exact:.1%} exact carts.",
        "",
        f"Reference: always answering `{classes[0]} x1` would score {baseline:.1%} exact.",
        "",
        "## Accuracy vs number of training orders",
        "",
        "| training orders | test exact cart | wrong-charge rate (charging at 0.5) |",
        "| --- | --- | --- |",
        *[f"| {n} | {acc:.1%} | {wrong:.1%} |" for n, acc, wrong in curve],
        "",
        f"Split sizes: train {len(parts['train'])}, val {len(parts['val'])}, "
        f"test {len(parts['test'])}. Left out of val/test because a product line was "
        f"unknown: {incomplete}.",
    ]
    (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    report += less_seen_section(
        parts["train"], parts["test"], truth, carts, classes, flagged, args.less_seen
    )
    (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    by_id = {r.order_id: r for r in parts["test"]}
    with (args.out / "disagreements.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["order_id", "fridge_key", "date", "charged", "model"])
        for order in final.disagreements:
            r = by_id[order]
            charged = target_cart(r.items, classes)
            writer.writerow(
                [order, r.fridge_key, r.date, json.dumps(charged), json.dumps(carts[order])]
            )
            wanted = sorted(set(charged) | set(carts[order]))
            save_evidence(
                r, attention[order], classes, wanted, args.features, args.videos,
                args.out / "evidence",
            )  # fmt: skip
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
