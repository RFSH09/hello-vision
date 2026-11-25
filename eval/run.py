"""Evaluation harness: run an analyzer over sessions with ground truth and write a report.

Sessions come either from a folder (``<name>.mp4`` + ``<name>.truth.json``, as written by
``python -m sim.make_demo``) or are generated on the fly with ``--synthetic``.

Usage:
    uv run python -m eval.run --synthetic --out reports/
    uv run python -m eval.run --sessions /private/sessions --out /private/reports \
        --analyzer mypackage.module:build_analyzer
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from eval.gate import check_gate
from eval.metrics import Metrics, SessionEval, compute
from eval.report import to_html, to_markdown
from hello_vision.capture.types import Frame, Session
from hello_vision.session.analyzer import SessionAnalyzer
from hello_vision.session.types import SessionResult
from hello_vision.synth.session import SCENARIOS, render_session


def _actions(result: SessionResult) -> list[str]:
    return [f"{e.action.value}:{e.sku}" for e in result.events]


def evaluate_session(
    analyzer: SessionAnalyzer,
    session: Session,
    truth_cart: dict[str, int],
    truth_actions: list[str],
    clip: str | None = None,
) -> SessionEval:
    """Analyse one session (timing the call) and compare with the truth."""
    start = time.perf_counter()
    result = analyzer.analyze(session)
    latency = time.perf_counter() - start
    return SessionEval(
        session_id=session.session_id,
        truth=truth_cart,
        pred=result.cart_dict(),
        needs_review=result.needs_review,
        latency_s=latency,
        truth_actions=truth_actions,
        pred_actions=_actions(result),
        clip=clip,
    )


def synthetic_evals(
    analyzer: SessionAnalyzer, sprites_seeds: Sequence[int] = (0, 1, 2)
) -> list[SessionEval]:
    """Evaluate every synthetic scenario for each seed."""
    from hello_vision.catalog.packshots import placeholder_sprites

    sprites = placeholder_sprites()
    out: list[SessionEval] = []
    for scenario in sorted(SCENARIOS):
        for seed in sprites_seeds:
            clip = render_session(scenario, sprites, seed=seed)
            frames = [Frame("cam0", i, i / clip.truth.fps, im) for i, im in enumerate(clip.frames)]
            session = Session(
                clip.truth.session_id, 0.0, clip.truth.num_frames / clip.truth.fps, frames
            )
            out.append(
                evaluate_session(
                    analyzer,
                    session,
                    {c.sku: c.qty for c in clip.truth.cart},
                    [f"{a.kind}:{a.sku}" for a in clip.truth.actions],
                )
            )
    return out


def folder_evals(analyzer: SessionAnalyzer, folder: Path) -> list[SessionEval]:
    """Evaluate every ``<name>.truth.json`` in ``folder``.

    A truth file needs ``cart`` (list of ``{sku, qty}`` or a ``{sku: qty}`` map) and may have
    ``session_id``, ``actions`` (``{kind, sku}``) and ``video`` (path; default: the sibling
    ``<name>.mp4``). Weak labels from ``datatools.weak_labels`` use the same format.

    Raises:
        FileNotFoundError: A truth file points at a video that does not exist.
    """
    from sim.replay import replay_video

    out: list[SessionEval] = []
    for truth_path in sorted(folder.glob("*.truth.json")):
        raw = json.loads(truth_path.read_text(encoding="utf-8"))
        name = truth_path.name.removesuffix(".truth.json")
        video = Path(raw["video"]) if raw.get("video") else truth_path.with_name(name + ".mp4")
        if not video.is_file():
            raise FileNotFoundError(f"video for {truth_path.name} not found: {video}")
        cart = raw["cart"]
        truth_cart = {c["sku"]: c["qty"] for c in cart} if isinstance(cart, list) else dict(cart)
        session = replay_video(video, session_id=str(raw.get("session_id", name)))
        out.append(
            evaluate_session(
                analyzer,
                session,
                truth_cart,
                [f"{a['kind']}:{a['sku']}" for a in raw.get("actions", [])],
                clip=str(video),
            )
        )
    return out


def load_analyzer(spec: str) -> SessionAnalyzer:
    """Load ``module:function`` and call it to build an analyzer."""
    module, _, func = spec.partition(":")
    factory: Callable[[], SessionAnalyzer] = getattr(importlib.import_module(module), func)
    return factory()


def write_reports(out_dir: Path, title: str, evals: list[SessionEval], note: str = "") -> Metrics:
    """Write ``report.md``, ``report.html`` and ``results.json`` into ``out_dir``."""
    m = compute(evals)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Latency here is not measured on the target edge device and the data is not shadow
    # data spanning days, so the gate is shown but cannot pass.
    gate = check_gate(m, days_covered=0.0, on_target_device=False)
    md = to_markdown(title, m, evals, gate, note)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.html").write_text(to_html(md), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps([e.model_dump() for e in evals], indent=2), encoding="utf-8"
    )
    return m


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--sessions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analyzer", default="sim.pipeline:build_demo_analyzer")
    args = parser.parse_args(argv)
    analyzer = load_analyzer(args.analyzer)
    evals = synthetic_evals(analyzer) if args.synthetic else folder_evals(analyzer, args.sessions)
    note = "SYNTHETIC data and baseline stand-in models: proves the harness, not real accuracy."
    m = write_reports(args.out, "Evaluation report", evals, note if args.synthetic else "")
    print(
        f"sessions={m.sessions} exact={m.exact_cart_accuracy:.3f} "
        f"review={m.review_rate:.3f} false_charge={m.false_charge_rate:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
