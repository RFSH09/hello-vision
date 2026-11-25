"""Session-level evaluation metrics (pure functions over predicted vs true carts)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel


class SessionEval(BaseModel):
    """One evaluated session.

    Attributes:
        session_id: Session identifier.
        truth: True cart ``{sku: qty}``.
        pred: Predicted cart ``{sku: qty}`` (the proposal even if ``needs_review``).
        needs_review: Whether the module flagged the session for a human.
        latency_s: Seconds from door close to result.
        truth_actions: True actions as ``"take:<sku>"`` / ``"return:<sku>"`` (may be empty).
        pred_actions: Predicted actions in the same format.
        clip: Reference to the evidence clip or source video, if any.
    """

    session_id: str
    truth: dict[str, int]
    pred: dict[str, int]
    needs_review: bool = False
    latency_s: float = 0.0
    truth_actions: list[str] = []
    pred_actions: list[str] = []
    clip: str | None = None


def false_charged(truth: dict[str, int], pred: dict[str, int]) -> dict[str, int]:
    """Items predicted beyond what was truly taken: ``{sku: extra qty}``."""
    return {s: n - truth.get(s, 0) for s, n in pred.items() if n > truth.get(s, 0)}


def missed(truth: dict[str, int], pred: dict[str, int]) -> dict[str, int]:
    """Items truly taken but not predicted: ``{sku: missing qty}``."""
    return {s: n - pred.get(s, 0) for s, n in truth.items() if n > pred.get(s, 0)}


@dataclass
class Metrics:
    """Aggregated metrics over a set of sessions."""

    sessions: int = 0
    exact_cart_accuracy: float = 0.0  # all sessions, proposal vs truth
    auto_exact_accuracy: float = 0.0  # only sessions NOT flagged for review (what gets charged)
    review_rate: float = 0.0
    false_charge_rate: float = 0.0  # auto-charged sessions with any wrongly charged item
    false_charged_items: int = 0
    missed_items_in_auto: int = 0
    item_precision: float = 1.0
    item_recall: float = 1.0
    take_return_errors: int = 0
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    confusion: dict[str, Counter[str]] = field(default_factory=dict)  # true sku -> predicted sku
    failures: list[str] = field(default_factory=list)  # session ids that are not exact


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def compute(evals: Sequence[SessionEval]) -> Metrics:
    """Aggregate session results into the evaluation KPIs.

    The charged population is the sessions with ``needs_review == False``: only those would
    be billed automatically, so false charges and auto accuracy are measured on them.
    """
    m = Metrics(sessions=len(evals))
    if not evals:
        return m
    exact = [e for e in evals if e.pred == e.truth]
    auto = [e for e in evals if not e.needs_review]
    m.exact_cart_accuracy = len(exact) / len(evals)
    m.review_rate = (len(evals) - len(auto)) / len(evals)
    m.auto_exact_accuracy = sum(e.pred == e.truth for e in auto) / len(auto) if auto else 1.0
    bad_charge = [e for e in auto if false_charged(e.truth, e.pred)]
    m.false_charge_rate = len(bad_charge) / len(auto) if auto else 0.0
    m.false_charged_items = sum(sum(false_charged(e.truth, e.pred).values()) for e in auto)
    m.missed_items_in_auto = sum(sum(missed(e.truth, e.pred).values()) for e in auto)
    pred_n = sum(sum(e.pred.values()) for e in evals)
    true_n = sum(sum(e.truth.values()) for e in evals)
    tp = sum(min(n, e.truth.get(s, 0)) for e in evals for s, n in e.pred.items())
    m.item_precision = tp / pred_n if pred_n else 1.0
    m.item_recall = tp / true_n if true_n else 1.0
    for e in evals:
        extra, lost = false_charged(e.truth, e.pred), missed(e.truth, e.pred)
        if len(extra) == 1 and len(lost) == 1:  # one item mistaken for another
            m.confusion.setdefault(next(iter(lost)), Counter())[next(iter(extra))] += 1
        elif lost:
            for s in lost:
                m.confusion.setdefault(s, Counter())["<missed>"] += lost[s]
        if e.pred != e.truth:
            m.failures.append(e.session_id)
        if e.truth_actions or e.pred_actions:
            diff = Counter(e.truth_actions) - Counter(e.pred_actions)
            diff2 = Counter(e.pred_actions) - Counter(e.truth_actions)
            m.take_return_errors += sum(diff.values()) + sum(diff2.values())
    lat = [e.latency_s for e in evals]
    m.latency_p50_s, m.latency_p95_s = _percentile(lat, 0.5), _percentile(lat, 0.95)
    return m
