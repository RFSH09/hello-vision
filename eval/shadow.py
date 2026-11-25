"""Shadow-mode comparison: our carts vs the carts of the system currently charging customers.

Both inputs are JSON Lines, one session per line:
``{"session_id": "...", "items": [{"sku": "...", "qty": 1}], "needs_review": false,
"started_at": "2026-01-01T10:00:00+00:00", "clip": "optional path or reference"}``.
Only ``session_id`` and ``items`` are required. The other system's SKU ids must be mapped to
ours before comparing (see ``--sku-map``).

Usage: ``uv run python -m eval.shadow --ours ours.jsonl --theirs theirs.jsonl --out report/``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from eval.metrics import false_charged, missed
from eval.report import to_html


@dataclass
class Disagreement:
    """A session where the two systems differ."""

    session_id: str
    ours: dict[str, int]
    theirs: dict[str, int]
    kind: str  # we_missed | we_added | swapped | quantity
    ours_flagged: bool
    clip: str | None


@dataclass
class ShadowReport:
    """Result of comparing the two systems."""

    common: int = 0
    only_ours: list[str] = field(default_factory=list)
    only_theirs: list[str] = field(default_factory=list)
    agree: int = 0
    flagged: int = 0  # ours flagged for review (excluded from the agreement rate)
    disagreements: list[Disagreement] = field(default_factory=list)
    days_covered: float = 0.0

    @property
    def agreement_rate(self) -> float:
        """Agreement over the sessions we did not flag (1.0 if none)."""
        decided = self.common - self.flagged
        return (self.agree / decided) if decided > 0 else 1.0


def _cart(items: list[dict[str, object]], sku_map: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        sku = sku_map.get(str(it["sku"]), str(it["sku"]))
        out[sku] = out.get(sku, 0) + int(str(it.get("qty", 1)))
    return {s: n for s, n in out.items() if n > 0}


def load_jsonl(path: Path, sku_map: dict[str, str] | None = None) -> dict[str, dict[str, object]]:
    """Read a JSONL file into ``{session_id: record}`` with the cart normalised to ``cart``."""
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["cart"] = _cart(rec["items"], sku_map or {})
        rows[str(rec["session_id"])] = rec
    return rows


def classify(ours: dict[str, int], theirs: dict[str, int]) -> str:
    """Name the kind of disagreement from our point of view."""
    extra, lost = false_charged(theirs, ours), missed(theirs, ours)
    if extra and lost:
        return (
            "swapped"
            if sum(extra.values()) == sum(lost.values()) and len(extra) == len(lost)
            else "quantity"
        )
    if lost:
        return "we_missed"
    if extra:
        return "we_added"
    return "quantity"


def compare(
    ours: dict[str, dict[str, object]], theirs: dict[str, dict[str, object]]
) -> ShadowReport:
    """Compare two loaded session sets."""
    rep = ShadowReport()
    rep.only_ours = sorted(set(ours) - set(theirs))
    rep.only_theirs = sorted(set(theirs) - set(ours))
    stamps: list[datetime] = []
    for sid in sorted(set(ours) & set(theirs)):
        rep.common += 1
        o, t = ours[sid], theirs[sid]
        if o.get("started_at"):
            stamps.append(datetime.fromisoformat(str(o["started_at"])))
        flagged = bool(o.get("needs_review", False))
        oc, tc = o["cart"], t["cart"]
        assert isinstance(oc, dict) and isinstance(tc, dict)  # noqa: S101
        if flagged:
            rep.flagged += 1
        if oc == tc:
            if not flagged:
                rep.agree += 1
            continue
        clip = o.get("clip") or t.get("clip")
        rep.disagreements.append(
            Disagreement(sid, oc, tc, classify(oc, tc), flagged, str(clip) if clip else None)
        )
    if len(stamps) > 1:
        rep.days_covered = (max(stamps) - min(stamps)).total_seconds() / 86400
    return rep


def to_markdown(rep: ShadowReport) -> str:
    """Render the shadow report as Markdown."""
    lines = [
        "# Shadow-mode comparison", "",
        f"- sessions in both systems: {rep.common}",
        f"- agreement (sessions we did not flag): {rep.agreement_rate:.4f}",
        f"- flagged for review by us: {rep.flagged}",
        f"- disagreements: {len(rep.disagreements)}",
        f"- only in ours: {len(rep.only_ours)}, only in theirs: {len(rep.only_theirs)}",
        f"- data spans: {rep.days_covered:.1f} days", "",
        "## Every disagreement", "",
        "| session | kind | ours | theirs | flagged | clip |",
        "| --- | --- | --- | --- | --- | --- |",
    ]  # fmt: skip
    for d in rep.disagreements:
        lines.append(
            f"| {d.session_id} | {d.kind} | {d.ours} | {d.theirs} | "
            f"{'yes' if d.ours_flagged else 'no'} | {d.clip or ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--theirs", type=Path, required=True)
    p.add_argument("--sku-map", type=Path, help="JSON {their_sku: our_sku}")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    sku_map = json.loads(args.sku_map.read_text(encoding="utf-8")) if args.sku_map else {}
    rep = compare(load_jsonl(args.ours), load_jsonl(args.theirs, sku_map))
    md = to_markdown(rep)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shadow.md").write_text(md, encoding="utf-8")
    (args.out / "shadow.html").write_text(to_html(md), encoding="utf-8")
    print(f"agreement={rep.agreement_rate:.4f} disagreements={len(rep.disagreements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
