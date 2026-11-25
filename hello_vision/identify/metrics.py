"""Identification metrics: per-SKU top-1 accuracy and a confusion matrix."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

UNKNOWN = "<unknown>"


def confusion(truth: Sequence[str], predicted: Sequence[str | None]) -> dict[str, Counter[str]]:
    """Count predictions per true SKU (``None`` predictions are recorded as ``<unknown>``).

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    out: dict[str, Counter[str]] = {}
    for t, p in zip(truth, predicted, strict=True):
        out.setdefault(t, Counter())[p if p is not None else UNKNOWN] += 1
    return out


def per_sku_accuracy(conf: dict[str, Counter[str]]) -> dict[str, float]:
    """Top-1 accuracy per true SKU (unknowns count as errors)."""
    return {t: row[t] / sum(row.values()) for t, row in sorted(conf.items()) if sum(row.values())}


def overall_accuracy(conf: dict[str, Counter[str]]) -> float:
    """Top-1 accuracy over all samples (0.0 if empty)."""
    total = sum(sum(r.values()) for r in conf.values())
    correct = sum(r[t] for t, r in conf.items())
    return correct / total if total else 0.0
