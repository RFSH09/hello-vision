"""Latency / throughput benchmark for the pipeline stages on the machine it runs on.

Measures wall-clock time only. Power draw and cost cannot be measured by software and must
be recorded by hand (power meter, price list) in docs/EDGE.md.

Usage: ``uv run python -m hello_vision.edge.benchmark --out demo_out/bench.json``
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from hello_vision.capture.types import Frame, Session


@dataclass
class StageResult:
    """Timing of one stage."""

    name: str
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


def time_callable(
    name: str, fn: Callable[[], object], runs: int = 20, warmup: int = 2
) -> StageResult:
    """Time ``fn`` ``runs`` times after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    s = sorted(samples)
    mean = statistics.fmean(samples)
    return StageResult(
        name,
        runs,
        mean,
        s[len(s) // 2],
        s[min(len(s) - 1, round(0.95 * (len(s) - 1)))],
        1000 / mean,
    )


def device_info() -> dict[str, object]:
    """Describe the machine (power and cost are NOT measured)."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "power_w": None,
        "cost": None,
        "note": "wall-clock only; power and cost must be measured/recorded manually",
    }


def run_benchmark(runs: int = 20) -> dict[str, object]:
    """Benchmark detector, embedder and the whole session analysis on a synthetic session."""
    from hello_vision.catalog.packshots import placeholder_sprites
    from hello_vision.identify.index import crop
    from hello_vision.synth.session import render_session
    from sim.pipeline import build_demo_analyzer

    clip = render_session("take_one", placeholder_sprites(), seed=0)
    analyzer = build_demo_analyzer()
    frame = clip.frames[12]
    analyzer.detector.start_session(clip.frames[0])
    dets = [d for d in analyzer.detector.detect(frame) if d.label == "product"]
    patch = crop(frame, (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2))
    frames = [Frame("cam0", i, i / 15, im) for i, im in enumerate(clip.frames)]
    session = Session("bench", 0.0, len(frames) / 15, frames)

    stages = [
        time_callable("detect (1 frame, baseline)", lambda: analyzer.detector.detect(frame), runs),
        time_callable("embed (1 crop, baseline)", lambda: analyzer.embedder.embed(patch), runs),
        time_callable(
            f"analyze whole session ({len(frames)} frames)",
            lambda: analyzer.analyze(session),
            max(3, runs // 5),
            warmup=1,
        ),
    ]
    return {
        "device": device_info(),
        "models": "SYNTHETIC BASELINES (classical); learned models are not trained yet",
        "stages": [asdict(s) for s in stages],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args(argv)
    result = run_benchmark(args.runs)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
