"""Golden-file tests for the simulator.

Ground truth and the replay summary are compared with files in tests/golden/. Pixels are not
compared (video codecs differ by platform). Regenerate deliberately with UPDATE_GOLDEN=1.
"""

import json
import os
from pathlib import Path

import pytest

from hello_vision.synth.session import SCENARIOS, write_video
from sim.replay import replay_video, summarize

from .conftest import cached_clip

GOLDEN = Path(__file__).parent / "golden"


def _check(name: str, data: object) -> None:
    path = GOLDEN / name
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_truth_golden(scenario: str) -> None:
    clip = cached_clip(scenario)
    _check(f"{scenario}.truth.json", json.loads(clip.truth.model_dump_json()))


def test_replay_summary_golden(tmp_path: Path) -> None:
    clip = cached_clip("take_two")
    write_video(clip.frames, tmp_path / "v.mp4")
    summary = summarize(replay_video(tmp_path / "v.mp4"))
    _check("take_two.replay.json", summary.model_dump())
