"""Cart-level evaluation of the session model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

OTHER = "other"


@dataclass
class CartReport:
    """Evaluation over sessions."""

    sessions: int
    exact: float
    per_sku: dict[str, tuple[float, float, int]]  # precision, recall, support
    count_accuracy: float  # quantity right, among items correctly detected as present
    confusion: Counter[tuple[str, str]]
    disagreements: list[str] = field(default_factory=list)


def evaluate_carts(
    truth: Mapping[str, Mapping[str, int]],
    predicted: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
) -> CartReport:
    """Compare predicted and true carts (both ``order -> {class: qty}``)."""
    orders = sorted(truth)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    right_count = present_right = 0
    wrong: list[str] = []
    for o in orders:
        t = {k: v for k, v in truth[o].items() if v > 0}
        p = {k: v for k, v in predicted.get(o, {}).items() if v > 0}
        if t != p:
            wrong.append(o)
        for c in classes:
            if c in t and c in p:
                tp[c] += 1
                present_right += 1
                right_count += t[c] == p[c]
            elif c in p:
                fp[c] += 1
            elif c in t:
                fn[c] += 1
        for missed in set(t) - set(p):
            for extra in set(p) - set(t):
                confusion[(missed, extra)] += 1
    per_sku = {
        c: (
            tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0,
            tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0,
            tp[c] + fn[c],
        )
        for c in classes
    }
    n = len(orders)
    return CartReport(
        n,
        (n - len(wrong)) / n if n else 0.0,
        per_sku,
        right_count / present_right if present_right else 0.0,
        confusion,
        wrong,
    )


def to_markdown(report: CartReport, title: str) -> str:
    """Render a report."""
    lines = [
        f"## {title}",
        "",
        f"- sessions: {report.sessions}",
        f"- exact cart accuracy: {report.exact:.1%}",
        f"- count accuracy (items found): {report.count_accuracy:.1%}",
        "",
        "| class | precision | recall | sessions with it |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {c} | {p:.1%} | {r:.1%} | {s} |" for c, (p, r, s) in report.per_sku.items()]
    lines += ["", "Confused (true -> predicted):"]
    lines += [f"- {t} -> {p}: {n}" for (t, p), n in report.confusion.most_common(10)] or ["- none"]
    return "\n".join(lines) + "\n"
