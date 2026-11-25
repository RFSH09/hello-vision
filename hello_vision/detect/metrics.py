"""Detection metrics: greedy IoU matching of predictions to ground truth."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hello_vision.detect.types import Detection, iou


@dataclass
class MatchResult:
    """Counts for one label plus the matched pairs (``gt index -> pred index``)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """TP / (TP + FP); 1.0 when nothing was predicted."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN); 1.0 when there was nothing to find."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0


def match(
    truth: Sequence[Detection], pred: Sequence[Detection], iou_threshold: float = 0.5
) -> tuple[dict[str, MatchResult], dict[int, int]]:
    """Match predictions to ground truth, per label, highest IoU first.

    Args:
        truth: Ground-truth boxes.
        pred: Predicted boxes.
        iou_threshold: Minimum IoU for a match.

    Returns:
        Per-label counts and a map from ground-truth index to matched prediction index.
    """
    results: dict[str, MatchResult] = {}
    pairs: dict[int, int] = {}
    labels = {d.label for d in truth} | {d.label for d in pred}
    for label in sorted(labels):
        res = results.setdefault(label, MatchResult())
        cands = sorted(
            (
                (iou(g, p), gi, pi)
                for gi, g in enumerate(truth)
                if g.label == label
                for pi, p in enumerate(pred)
                if p.label == label
            ),
            reverse=True,
        )
        used_g: set[int] = set()
        used_p: set[int] = set()
        for score, gi, pi in cands:
            if score < iou_threshold:
                break
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            pairs[gi] = pi
        n_g = sum(1 for g in truth if g.label == label)
        n_p = sum(1 for p in pred if p.label == label)
        res.tp, res.fn, res.fp = len(used_g), n_g - len(used_g), n_p - len(used_p)
    return results, pairs
