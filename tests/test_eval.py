import json
from pathlib import Path

import pytest

from eval.gate import GateThresholds, check_gate
from eval.metrics import SessionEval, compute, false_charged, missed
from eval.report import to_html, to_markdown
from eval.run import evaluate_session, folder_evals, load_analyzer, main, write_reports
from eval.shadow import classify, compare, load_jsonl
from eval.shadow import main as shadow_main
from hello_vision.capture.types import Frame, Session
from hello_vision.synth.session import write_video
from sim.pipeline import build_demo_analyzer

from .conftest import cached_clip


def ev(
    sid: str,
    truth: dict[str, int],
    pred: dict[str, int],
    review: bool = False,
    lat: float = 1.0,
    **kw,
) -> SessionEval:  # type: ignore[no-untyped-def]
    return SessionEval(
        session_id=sid, truth=truth, pred=pred, needs_review=review, latency_s=lat, **kw
    )


def test_cart_diffs() -> None:
    assert false_charged({"a": 1}, {"a": 2, "b": 1}) == {"a": 1, "b": 1}
    assert missed({"a": 2, "b": 1}, {"a": 1}) == {"a": 1, "b": 1}


def test_metrics_kpis() -> None:
    evals = [
        ev("s1", {"a": 1}, {"a": 1}),
        ev("s2", {"a": 1}, {"b": 1}),  # one item mistaken for another, auto-charged
        ev("s3", {"a": 1}, {}, review=True),  # flagged, so not charged
        ev("s4", {}, {}),
        ev("s5", {"a": 1}, {"a": 1}, truth_actions=["take:a"], pred_actions=["take:a", "return:a"]),
    ]
    m = compute(evals)
    assert m.sessions == 5 and m.exact_cart_accuracy == pytest.approx(3 / 5)
    assert m.review_rate == pytest.approx(1 / 5)
    assert m.auto_exact_accuracy == pytest.approx(3 / 4)
    assert m.false_charge_rate == pytest.approx(1 / 4) and m.false_charged_items == 1
    assert m.missed_items_in_auto == 1
    assert m.confusion["a"]["b"] == 1 and m.confusion["a"]["<missed>"] == 1
    assert m.take_return_errors == 1
    assert m.item_precision == pytest.approx(2 / 3) and m.item_recall == pytest.approx(2 / 4)
    assert m.failures == ["s2", "s3"]


def test_metrics_empty_and_latency() -> None:
    assert compute([]).sessions == 0
    m = compute([ev(str(i), {}, {}, lat=float(i)) for i in range(1, 11)])
    assert m.latency_p50_s == pytest.approx(6.0, abs=1.0) and m.latency_p95_s == pytest.approx(
        10.0, abs=1.0
    )
    only_review = compute([ev("x", {"a": 1}, {}, review=True)])
    assert only_review.auto_exact_accuracy == 1.0 and only_review.false_charge_rate == 0.0


def test_gate_logic() -> None:
    good = compute([ev(str(i), {"a": 1}, {"a": 1}, lat=1.0) for i in range(200)])
    rows = check_gate(good, days_covered=15, on_target_device=True)
    assert all(r.passed for r in rows)
    rows = check_gate(good, days_covered=3, on_target_device=False)
    assert [r.name for r in rows if not r.passed] == [
        "latency p95 (door close -> result)",
        "keeps up with cameras (processing / session time)",
        "shadow duration",
    ]
    bad = compute([ev("a", {"a": 1}, {"b": 1})])
    assert not check_gate(bad, 20, True, GateThresholds())[0].passed


def test_reports(tmp_path: Path) -> None:
    evals = [ev("s1", {"a": 1}, {"a": 1}), ev("s2<script>", {"a": 1}, {"b": 1}, clip="clip.mp4")]
    m = write_reports(tmp_path, "T", evals, note="a note")
    md = (tmp_path / "report.md").read_text()
    assert "NOT PASSED" in md and "clip.mp4" in md and "a note" in md and "`a` -> `b` x1" in md
    html_text = (tmp_path / "report.html").read_text()
    assert "<table>" in html_text and "<script>" not in html_text and "&lt;script&gt;" in html_text
    assert json.loads((tmp_path / "results.json").read_text())[0]["session_id"] == "s1"
    assert m.sessions == 2
    assert "Go/no-go" not in to_markdown("T", m, evals)
    assert to_html("# x").startswith("<!doctype html>")


def test_evaluate_session_and_folder(tmp_path: Path) -> None:
    clip = cached_clip("take_one")
    frames = [Frame("cam0", i, i / 15, im) for i, im in enumerate(clip.frames)]
    analyzer = build_demo_analyzer()
    e = evaluate_session(
        analyzer,
        Session("s", 0, 1, frames),
        {c.sku: c.qty for c in clip.truth.cart},
        ["take:demo_tea_bottle"],
    )
    assert e.pred == e.truth and not e.needs_review and e.latency_s > 0
    assert e.pred_actions == ["take:demo_tea_bottle"]

    write_video(clip.frames, tmp_path / "a.mp4")
    (tmp_path / "a.truth.json").write_text(clip.truth.model_dump_json())
    write_video(clip.frames[:5], tmp_path / "orphan.mp4")  # no truth file: skipped
    results = folder_evals(analyzer, tmp_path)
    assert [r.session_id for r in results] == [clip.truth.session_id]
    assert results[0].pred == results[0].truth and results[0].clip.endswith("a.mp4")


def test_load_analyzer_and_cli(tmp_path: Path) -> None:
    assert load_analyzer("sim.pipeline:build_demo_analyzer") is not None
    clip = cached_clip("take_nothing")
    write_video(clip.frames, tmp_path / "n.mp4")
    (tmp_path / "n.truth.json").write_text(clip.truth.model_dump_json())
    assert main(["--sessions", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "report.md").exists()


def jl(path: Path, rows: list[dict]) -> Path:  # type: ignore[type-arg]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_shadow_compare_and_cli(tmp_path: Path) -> None:
    ours = jl(
        tmp_path / "o.jsonl",
        [
            {
                "session_id": "1",
                "items": [{"sku": "a", "qty": 1}],
                "started_at": "2026-01-01T10:00:00+00:00",
            },
            {"session_id": "2", "items": [], "clip": "c2.mp4"},
            {"session_id": "3", "items": [{"sku": "x", "qty": 1}]},
            {"session_id": "4", "items": [{"sku": "a", "qty": 1}], "needs_review": True},
            {
                "session_id": "7",
                "items": [{"sku": "a", "qty": 2}],
                "started_at": "2026-01-16T10:00:00+00:00",
            },
            {"session_id": "only-ours", "items": []},
        ],
    )
    theirs = jl(
        tmp_path / "t.jsonl",
        [
            {"session_id": "1", "items": [{"sku": "A_THEIR", "qty": 1}]},
            {"session_id": "2", "items": [{"sku": "b", "qty": 1}]},
            {"session_id": "3", "items": [{"sku": "y", "qty": 1}]},
            {"session_id": "4", "items": [{"sku": "b", "qty": 1}]},
            {"session_id": "7", "items": [{"sku": "A_THEIR", "qty": 1}]},
            {"session_id": "only-theirs", "items": []},
        ],
    )
    rep = compare(load_jsonl(ours), load_jsonl(theirs, {"A_THEIR": "a"}))
    assert (rep.common, rep.agree, rep.flagged) == (5, 1, 1)
    assert rep.only_ours == ["only-ours"] and rep.only_theirs == ["only-theirs"]
    kinds = {d.session_id: d.kind for d in rep.disagreements}
    assert kinds == {"2": "we_missed", "3": "swapped", "4": "swapped", "7": "we_added"}
    assert rep.disagreements[0].clip == "c2.mp4" and rep.days_covered == pytest.approx(15.0)
    assert rep.agreement_rate == pytest.approx(1 / 4)
    sku_map = tmp_path / "map.json"
    sku_map.write_text(json.dumps({"A_THEIR": "a"}))
    assert (
        shadow_main(
            [
                "--ours",
                str(ours),
                "--theirs",
                str(theirs),
                "--sku-map",
                str(sku_map),
                "--out",
                str(tmp_path / "r"),
            ]
        )
        == 0
    )
    assert "Every disagreement" in (tmp_path / "r" / "shadow.md").read_text()
    assert (tmp_path / "r" / "shadow.html").exists()


def test_classify_quantity() -> None:
    assert classify({"a": 1, "b": 1}, {"a": 3}) == "quantity"
    assert classify({"a": 1, "b": 1}, {"a": 2}) == "swapped"
    assert classify({"a": 1}, {"a": 1, "b": 1}) == "we_missed"
