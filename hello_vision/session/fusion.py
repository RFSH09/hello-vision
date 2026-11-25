"""Fuse track events and the shelf diff into the final cart and the review decision."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from hello_vision.session.shelf import UNKNOWN_KEY
from hello_vision.session.types import Action, ActionEvent, CartLine


@dataclass(frozen=True)
class FusionConfig:
    """Thresholds for the review decision.

    Attributes:
        min_confidence: Lines or sessions below this confidence need review.
        empty_confidence: Confidence assigned to a verified empty session.
    """

    min_confidence: float = 0.75
    empty_confidence: float = 0.95


def fuse(
    events: list[ActionEvent],
    shelf: dict[str, int] | None,
    cfg: FusionConfig | None = None,
) -> tuple[list[CartLine], float, bool, list[str]]:
    """Build the cart and decide whether a human must review the session.

    Args:
        events: TAKE/RETURN events from tracking.
        shelf: ``before - after`` shelf counts per SKU, or ``None`` if unavailable.
        cfg: Thresholds.

    Returns:
        ``(cart lines, session confidence, needs_review, reasons)``. The cart is returned even
        when review is needed so the reviewer sees the proposal, but must not be charged then.
    """
    cfg = cfg or FusionConfig()
    reasons: list[str] = []
    net: dict[str, int] = defaultdict(int)
    conf: dict[str, float] = {}
    for e in events:
        if e.sku is None:
            reasons.append(f"unknown product ({e.action.value}, track {e.track_id})")
            continue
        net[e.sku] += 1 if e.action is Action.TAKE else -1
        conf[e.sku] = min(conf.get(e.sku, 1.0), e.confidence)
        if not e.hand_associated:
            reasons.append(f"motion without a hand nearby (track {e.track_id})")

    negative = sorted(s for s, n in net.items() if n < 0)
    if negative:
        reasons.append(f"returned items that were not taken in this session: {negative}")

    if shelf is None:
        reasons.append("shelf snapshots unavailable (no hand-free frames)")
    else:
        expected = {s: n for s, n in net.items() if n != 0}
        if UNKNOWN_KEY in shelf:
            reasons.append("shelf contains an unidentified product")
        if {s: n for s, n in shelf.items() if s != UNKNOWN_KEY} != expected:
            reasons.append(f"shelf diff {dict(shelf)} disagrees with tracked events {expected}")

    lines = [CartLine(s, n, conf.get(s, 0.0)) for s, n in sorted(net.items()) if n > 0]
    session_conf = min((ln.confidence for ln in lines), default=cfg.empty_confidence)
    if shelf is None:
        session_conf = min(session_conf, cfg.min_confidence - 0.01)
    low = [ln.sku for ln in lines if ln.confidence < cfg.min_confidence]
    if low:
        reasons.append(f"low identification confidence for {low}")
    return lines, session_conf, bool(reasons), reasons
