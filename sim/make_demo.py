"""Generate synthetic demo sessions (video + ground-truth JSON). No real footage is used.

Usage: ``uv run python -m sim.make_demo [--out demo_out] [--scenario take_one] [--seed 0]``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.synth.session import SCENARIOS, render_session, write_video


def make_demo(out: Path, scenario: str, seed: int = 0) -> tuple[Path, Path]:
    """Render ``scenario`` to ``out/<scenario>.mp4`` and ``out/<scenario>.truth.json``.

    Returns:
        The video path and the ground-truth path.
    """
    clip = render_session(scenario, placeholder_sprites(), seed=seed)
    video = out / f"{scenario}.mp4"
    truth = out / f"{scenario}.truth.json"
    write_video(clip.frames, video)
    truth.write_text(clip.truth.model_dump_json(indent=2), encoding="utf-8")
    return video, truth


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo_out"))
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="take_one")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    video, truth = make_demo(args.out, args.scenario, args.seed)
    print(f"video: {video}\ntruth: {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
