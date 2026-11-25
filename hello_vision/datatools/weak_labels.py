"""Weak labels from history: pair each purchase video with the item list that was charged.

IMPORTANT: a charged order is NOT ground truth. The system that produced it can be wrong
(that is the whole reason for shadow mode). These labels are a cheap starting point for
training/eval sessions; sessions where our module disagrees must be checked by a human
before they count as errors of ours.

Input: JSON Lines, one purchase per line, ONLY these fields:
``{"session_id", "video", "items": [{"sku", "qty"}], "started_at"?, "fridge_id"?}``.
Exports that contain anything that looks personal (phone, card, name, user ...) are refused
outright so such data never gets copied into the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hello_vision.catalog.models import Catalog
from hello_vision.datatools.labeling import SessionLabel

ALLOWED_FIELDS = {"session_id", "video", "items", "started_at", "fridge_id"}
PII_HINTS = ("phone", "card", "pan", "name", "email", "user", "customer", "client", "account")


class PrivacyError(ValueError):
    """The export contains fields that look like personal data."""


@dataclass
class WeakReport:
    """Outcome of :func:`build_weak_sessions`."""

    accepted: list[str] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)  # reason -> session ids


def _reject(rep: WeakReport, reason: str, sid: str) -> None:
    rep.rejected.setdefault(reason, []).append(sid)


def build_weak_sessions(
    records_path: Path, catalog: Catalog, videos_root: Path, out_dir: Path
) -> WeakReport:
    """Write one ``<session_id>.truth.json`` per usable purchase into ``out_dir``.

    The truth file references the video by path (videos are never copied) and is marked
    ``label_quality: weak``; it is readable by ``eval.run --sessions``.

    Raises:
        PrivacyError: If any record has a field that looks personal.
    """
    rep = WeakReport()
    out_dir.mkdir(parents=True, exist_ok=True)
    known = set(catalog.ids())
    seen: set[str] = set()
    root = videos_root.resolve()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        extra = set(rec) - ALLOWED_FIELDS
        bad = sorted(k for k in extra if any(h in k.lower() for h in PII_HINTS))
        if bad:
            raise PrivacyError(
                f"export has personal-looking fields {bad}; remove them at the source"
            )
        sid = str(rec.get("session_id", ""))
        if sid in seen:
            _reject(rep, "duplicate_session_id", sid)
            continue
        seen.add(sid)
        cart: dict[str, int] = {}
        for it in rec.get("items", []):
            cart[str(it["sku"])] = cart.get(str(it["sku"]), 0) + int(it.get("qty", 1))
        if any(s not in known for s in cart):
            _reject(rep, "unknown_sku", sid)
            continue
        if any(q < 1 for q in cart.values()):
            _reject(rep, "bad_quantity", sid)
            continue
        video = (videos_root / str(rec.get("video", ""))).resolve()
        if not video.is_relative_to(root) or not video.is_file():
            _reject(rep, "missing_video", sid)
            continue
        try:
            label = SessionLabel(session_id=sid, cart=cart)
        except ValueError:
            _reject(rep, "invalid_session_id", sid or "<empty>")
            continue
        truth = {
            "session_id": label.session_id,
            "cart": [{"sku": s, "qty": q} for s, q in sorted(cart.items())],
            "actions": [],
            "video": str(video),
            "label_quality": "weak",
            "source": "charged_order",
        }
        (out_dir / f"{sid}.truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        rep.accepted.append(sid)
    return rep
