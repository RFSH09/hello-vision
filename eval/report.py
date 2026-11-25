"""Render evaluation results as Markdown and HTML."""

from __future__ import annotations

import html
from collections.abc import Sequence

from eval.gate import GateRow
from eval.metrics import Metrics, SessionEval


def to_markdown(
    title: str,
    m: Metrics,
    evals: Sequence[SessionEval],
    gate: Sequence[GateRow] | None = None,
    note: str = "",
) -> str:
    """Build the Markdown report."""
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["## Summary", "", "| metric | value |", "| --- | --- |"]
    rows = [
        ("sessions", str(m.sessions)),
        ("exact cart accuracy (all sessions, proposal vs truth)", f"{m.exact_cart_accuracy:.4f}"),
        ("exact cart accuracy (auto-charged sessions only)", f"{m.auto_exact_accuracy:.4f}"),
        ("review rate", f"{m.review_rate:.4f}"),
        ("false-charge rate (auto-charged)", f"{m.false_charge_rate:.4f}"),
        ("falsely charged items", str(m.false_charged_items)),
        ("missed items in auto-charged sessions", str(m.missed_items_in_auto)),
        ("item precision / recall", f"{m.item_precision:.4f} / {m.item_recall:.4f}"),
        ("TAKE/RETURN action errors", str(m.take_return_errors)),
        (
            "latency p50 / p95 (door close -> result)",
            f"{m.latency_p50_s:.2f}s / {m.latency_p95_s:.2f}s",
        ),
        ("worst real-time factor (processing / session time)", f"{m.realtime_factor_max:.2f}"),
    ]
    lines += [f"| {a} | {b} |" for a, b in rows]
    if gate:
        lines += ["", "## Go/no-go gate (proposed, not yet confirmed)", ""]
        lines += ["| criterion | value | limit | result |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {g.name} | {g.value} | {g.limit} | {'PASS' if g.passed else 'FAIL'} |" for g in gate
        ]
        lines += ["", f"**Gate: {'PASS' if all(g.passed for g in gate) else 'NOT PASSED'}**"]
    if m.confusion:
        lines += ["", "## Per-SKU confusion (true -> predicted, counts)", ""]
        for true, row in sorted(m.confusion.items()):
            lines.append(f"- `{true}` -> " + ", ".join(f"`{p}` x{n}" for p, n in row.most_common()))
    if m.failures:
        lines += ["", "## Sessions that are not an exact match", ""]
        by_id = {e.session_id: e for e in evals}
        for sid in m.failures:
            e = by_id[sid]
            flag = " (flagged for review)" if e.needs_review else ""
            clip = f" clip: {e.clip}" if e.clip else ""
            lines.append(f"- `{sid}`: truth {e.truth} vs predicted {e.pred}{flag}{clip}")
    return "\n".join(lines) + "\n"


def to_html(markdown: str) -> str:
    """Wrap a Markdown report in a minimal standalone HTML page (tables rendered, text escaped)."""
    out: list[str] = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", " "} for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
                continue
            out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            out.append(f"<p>&bull; {html.escape(line[2:])}</p>")
        elif line.strip():
            out.append(f"<p>{html.escape(line.lstrip('> '))}</p>")
    if in_table:
        out.append("</table>")
    style = (
        "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:.3rem .6rem;text-align:left}"
    )
    return (
        f"<!doctype html><meta charset=utf-8><title>Evaluation</title><style>{style}</style>"
        + "\n".join(out)
    )
