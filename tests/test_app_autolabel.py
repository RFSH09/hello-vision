"""Demo-app helpers and auto-label selection (synthetic data only)."""

from pathlib import Path
from typing import Any

import numpy as np

from hello_vision.datatools.labeling import LabeledBox, LabeledFrame, import_label_studio
from hello_vision.review.demo_app import cart_text, thumbnails, track_rows
from hello_vision.synth.session import write_video
from sim.run_video import Models
from tools.autolabel import frame_uncertainty, label_studio_tasks, select


def track(tid: int, sku: str | None, held: bool, taken: bool) -> dict[str, Any]:
    return {
        "track_id": tid, "sku": sku, "best_guess": "x", "confidence": 0.5, "first_seen_s": 0.0,
        "last_seen_s": 0.5, "frames": 2, "hand_touched": held, "taken_guess": taken,
        "boxes": [[0.2, 10, 10, 40, 40, 0.3, int(held)], [0.4, 10, 10, 40, 40, 0.9, int(held)]],
    }  # fmt: skip


def test_rows_and_cart() -> None:
    tracks = [track(1, "a", False, False), track(2, "b", True, True), track(3, None, True, True)]
    assert [r[0] for r in track_rows(tracks)] == [2, 3, 1]
    text = cart_text(tracks)
    assert "b x1" in text and "unidentified: 1" in text


def test_thumbnails_only_for_held_tracks(tmp_path: Path) -> None:
    write_video([np.full((60, 80, 3), 90, np.uint8)] * 10, tmp_path / "v.mp4", fps=10)
    models = Models(None, None, None, "fake")  # type: ignore[arg-type]
    out = thumbnails(
        tmp_path / "v.mp4", [track(1, "a", False, False), track(2, None, True, True)], models
    )
    assert len(out) == 1 and out[0][1] == "#2 unknown" and out[0][0].size > 0


def test_uncertainty_prefers_held_low_score_frames() -> None:
    scores = frame_uncertainty([track(1, None, True, True), track(2, "a", False, False)])
    assert scores[0.2] > scores[0.4]


def test_select_round_robin_and_spacing() -> None:
    per = {"a": {0.0: 3, 0.1: 2.9, 2.0: 1}, "b": {0.0: 2, 5.0: 1}}
    got = select(per, 4)
    assert got == {"a": [0.0, 2.0], "b": [0.0, 5.0]}
    assert sum(len(v) for v in select(per, 99).values()) == 4  # 0.1 is too close to 0.0


def test_label_studio_export_can_be_read_back(tmp_path: Path) -> None:
    import json

    frame = LabeledFrame(
        file_name="f.jpg", width=200, height=100,
        boxes=[LabeledBox(label="product", sku="cola", x1=20, y1=10, x2=60, y2=60),
               LabeledBox(label="hand", x1=0, y1=0, x2=10, y2=10)],
    )  # fmt: skip
    tasks = label_studio_tasks([frame], "images/")
    exported = [{"data": t["data"], "annotations": t["predictions"]} for t in tasks]
    (tmp_path / "ls.json").write_text(json.dumps(exported))
    back = import_label_studio(tmp_path / "ls.json")[0]
    assert [(b.label, b.sku) for b in back.boxes] == [("product", "cola"), ("hand", None)]
    assert abs(back.boxes[0].x2 - 60) < 1e-6
