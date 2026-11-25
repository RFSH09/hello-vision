"""The proposed go/no-go gate before any real customer is charged by our model.

PROPOSED values from the project brief; the owner has not confirmed them yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.metrics import Metrics


@dataclass(frozen=True)
class GateThresholds:
    """Gate limits (proposed)."""

    min_exact_accuracy: float = 0.99
    max_false_charge_rate: float = 0.002
    max_review_rate: float = 0.03
    max_latency_s: float = 5.0
    max_realtime_factor: float = 1.0
    min_days: float = 14.0


@dataclass(frozen=True)
class GateRow:
    """One gate criterion and its outcome."""

    name: str
    value: str
    limit: str
    passed: bool


def check_gate(
    m: Metrics,
    days_covered: float,
    on_target_device: bool,
    t: GateThresholds | None = None,
) -> list[GateRow]:
    """Evaluate the gate.

    Args:
        m: Metrics computed on shadow data.
        days_covered: Number of days the shadow data spans.
        on_target_device: Whether latency was measured on the target edge device. If not,
            the latency criterion cannot pass.
        t: Thresholds.

    Returns:
        One row per criterion; the gate passes only if every row passed.
    """
    t = t or GateThresholds()
    return [
        GateRow("exact cart accuracy (auto-charged)", f"{m.auto_exact_accuracy:.4f}",
                f">= {t.min_exact_accuracy}", m.auto_exact_accuracy >= t.min_exact_accuracy),
        GateRow("false-charge rate", f"{m.false_charge_rate:.4f}",
                f"< {t.max_false_charge_rate}", m.false_charge_rate < t.max_false_charge_rate),
        GateRow("review rate", f"{m.review_rate:.4f}",
                f"< {t.max_review_rate}", m.review_rate < t.max_review_rate),
        GateRow("latency p95 (door close -> result)", f"{m.latency_p95_s:.2f}s"
                + ("" if on_target_device else " (NOT on target device)"),
                f"< {t.max_latency_s}s", on_target_device and m.latency_p95_s < t.max_latency_s),
        GateRow("keeps up with cameras (processing / session time)",
                f"{m.realtime_factor_max:.2f}"
                + ("" if on_target_device else " (NOT on target device)"),
                f"< {t.max_realtime_factor}",
                on_target_device and m.realtime_factor_max < t.max_realtime_factor),
        GateRow("shadow duration", f"{days_covered:.1f} days",
                f">= {t.min_days:.0f} days", days_covered >= t.min_days),
    ]  # fmt: skip
