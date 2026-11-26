"""Combine camera results into one cart per order and score it against the order list."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

_CAMERA = re.compile(r"_cam\d+$")


def order_id(video_stem: str) -> str:
    """``order-1_cam0`` -> ``order-1`` (names without a camera suffix are unchanged)."""
    return _CAMERA.sub("", video_stem)


def count_items(tracks: Sequence[dict[str, Any]], merge_gap_s: float = 4.0) -> Counter[str]:
    """Count taken items in ONE camera's tracks, joining broken-up tracks of the same item.

    Tracks of the same SKU that follow each other within ``merge_gap_s`` are one physical item
    (the tracker lost it and found it again); tracks that overlap in time are separate items.
    """
    cart: Counter[str] = Counter()
    by_sku: dict[str, list[dict[str, Any]]] = {}
    for t in tracks:
        if t["taken_guess"]:
            by_sku.setdefault(t["sku"] or "unknown", []).append(t)
    for sku, items in by_sku.items():
        ends: list[float] = []  # end time of each physical item
        for t in sorted(items, key=lambda x: float(x["first_seen_s"])):
            start, end = float(t["first_seen_s"]), float(t["last_seen_s"])
            joinable = [i for i, e in enumerate(ends) if 0 < start - e <= merge_gap_s]
            if joinable:
                ends[max(joinable, key=lambda i: ends[i])] = end
            else:
                ends.append(end)
        cart[sku] = len(ends)
    return cart


def merge_cameras(carts: Sequence[Counter[str]]) -> Counter[str]:
    """Per SKU, the largest count any camera saw (cameras watch the same items)."""
    out: Counter[str] = Counter()
    for c in carts:
        out |= c
    return out


@dataclass
class OrderScore:
    """Predicted vs true cart of one order."""

    order: str
    predicted: Counter[str]
    actual: Counter[str]
    seen: set[str] = field(default_factory=set)  # SKUs identified on any track
    unknown: int = 0  # held items the module could not identify (would go to review)
    multiple: bool = False  # items held together: quantity needs a human check

    @property
    def correct(self) -> int:
        """Items predicted and truly bought."""
        return sum((self.predicted & self.actual).values())

    @property
    def wrong(self) -> int:
        """Items predicted but not bought."""
        return sum((self.predicted - self.actual).values())

    @property
    def missed(self) -> int:
        """Items bought but not predicted."""
        return sum((self.actual - self.predicted).values())

    @property
    def exact(self) -> bool:
        """Whether the cart is exactly right."""
        return self.predicted == self.actual


def confused_pairs(scores: Sequence[OrderScore]) -> Counter[tuple[str, str]]:
    """Count ``(true SKU, predicted SKU)`` pairs among each order's missed and wrong items.

    This is an upper bound on confusion: an order with two errors pairs every missed item with
    every wrong one.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for s in scores:
        for truth in s.actual - s.predicted:
            for pred in s.predicted - s.actual:
                pairs[(truth, pred)] += 1
    return pairs


def fmt(cart: Counter[str]) -> str:
    """``a x1, b x2`` or ``-``."""
    return ", ".join(f"{s} x{n}" for s, n in sorted(cart.items())) or "-"


def report(scores: Sequence[OrderScore]) -> str:
    """Markdown table, totals and most confused pairs."""
    lines = [
        "| order | detected (naive taken) | actual (orders.csv) | correct | wrong | missed | "
        "unidentified | exact | actual SKU seen on any track | multiple? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in scores:
        seen = f"{len(set(s.actual) & s.seen)}/{len(set(s.actual))}"
        lines.append(
            f"| {s.order} | {fmt(s.predicted)} | {fmt(s.actual)} | {s.correct} | {s.wrong} | "
            f"{s.missed} | {s.unknown} | {'yes' if s.exact else 'no'} | {seen} | "
            f"{'REVIEW' if s.multiple else ''} |"
        )
    n = len(scores)
    exact = sum(s.exact for s in scores)
    correct = sum(s.correct for s in scores)
    wrong = sum(s.wrong for s in scores)
    actual = sum(sum(s.actual.values()) for s in scores)
    kinds = sum(len(set(s.actual)) for s in scores)
    seen_total = sum(len(set(s.actual) & s.seen) for s in scores)
    lines += [
        "",
        f"**Exact cart accuracy: {exact}/{n} = {exact / max(n, 1):.1%}**",
        f"Items: {correct}/{actual} correct (recall {correct / max(actual, 1):.1%}), "
        f"{wrong} wrong (precision {correct / max(correct + wrong, 1):.1%}), "
        f"{actual - correct} missed.",
        f"Bought SKU identified on at least one track: {seen_total}/{kinds}.",
        "",
        "Most confused (true -> predicted):",
    ]
    pairs = confused_pairs(scores).most_common(10)
    lines += [f"- {t} -> {p}: {c}" for (t, p), c in pairs] or ["- none"]
    return "\n".join(lines) + "\n"
