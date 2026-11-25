"""M2 offline metrics: detection and identification on SYNTHETIC sessions.

These numbers measure the synthetic baseline only. They say nothing about real footage.

Usage: ``uv run python -m eval.offline_m2 [--out report.md] [--seed 0]``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.detect.baseline import BaselineDetector
from hello_vision.detect.metrics import MatchResult, match
from hello_vision.detect.types import Detection
from hello_vision.identify.embedders import ColorLayoutEmbedder
from hello_vision.identify.index import build_index, crop
from hello_vision.identify.metrics import confusion, overall_accuracy, per_sku_accuracy
from hello_vision.synth.session import SCENARIOS, render_session


def run(seed: int = 0, scenarios: Sequence[str] | None = None) -> str:
    """Evaluate the baseline detector + colour embedder on every scenario; return Markdown."""
    sprites = placeholder_sprites()
    embedder = ColorLayoutEmbedder()
    index = build_index({k: [v] for k, v in sprites.items()}, embedder, threshold=0.0)
    totals: dict[str, MatchResult] = {}
    truth_skus: list[str] = []
    pred_skus: list[str | None] = []
    for scenario in sorted(scenarios or SCENARIOS):
        clip = render_session(scenario, sprites, seed=seed)
        detector = BaselineDetector()
        detector.start_session(clip.frames[0])
        for img, boxes in zip(clip.frames, clip.boxes, strict=True):
            preds = detector.detect(img)
            gts = [Detection(b.label, 1.0, b.x1, b.y1, b.x2, b.y2) for b in boxes]
            results, pairs = match(gts, preds)
            for label, r in results.items():
                t = totals.setdefault(label, MatchResult())
                t.tp, t.fp, t.fn = t.tp + r.tp, t.fp + r.fp, t.fn + r.fn
            for gi, pi in pairs.items():
                if boxes[gi].label == "product" and boxes[gi].sku:
                    p = preds[pi]
                    m = index.query(embedder.embed(crop(img, (p.x1, p.y1, p.x2, p.y2))))
                    truth_skus.append(boxes[gi].sku or "")
                    pred_skus.append(m.best_sku)
    conf = confusion(truth_skus, pred_skus)
    lines = ["# M2 offline metrics (synthetic data only)", "", "## Detection (IoU >= 0.5)", ""]
    lines += [
        "| label | precision | recall | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, r in sorted(totals.items()):
        lines.append(f"| {label} | {r.precision:.3f} | {r.recall:.3f} | {r.tp} | {r.fp} | {r.fn} |")
    lines += ["", f"## Identification top-1 (on matched boxes): {overall_accuracy(conf):.3f}", ""]
    lines += ["| SKU | top-1 |", "| --- | --- |"]
    lines += [f"| {k} | {v:.3f} |" for k, v in per_sku_accuracy(conf).items()]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = run(args.seed)
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
