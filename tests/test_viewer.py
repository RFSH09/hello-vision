"""Results viewer and the labelling-sheet import controls (synthetic data only)."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hello_vision.datatools.label_sheet import render
from hello_vision.synth.session import write_video
from sim.viewer import build, mistakes


def track(tid: int, sku: str | None, taken: bool, guess: str = "b") -> dict[str, Any]:
    return {
        "track_id": tid, "sku": sku, "best_guess": guess, "confidence": 0.5, "first_seen_s": 0.0,
        "last_seen_s": 0.4, "frames": 3, "hand_touched": True, "taken_guess": taken,
        "multiple": False, "boxes": [[0.2, 10, 10, 30, 40, 0.9, 1]],
    }  # fmt: skip


def result(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"tracks": tracks, "processed_frames": 5, "seconds": 2.0, "detector": "gdino",
            "rejected_detections": 0, "hands": []}  # fmt: skip


def test_mistakes_explain_wrong_and_missed() -> None:
    run = {"results": {"o1_cam0": result([track(1, "x", True)]),
                       "o2_cam0": result([track(1, "b", False)])}}  # fmt: skip
    got = mistakes(run, {"o1": Counter(b=1), "o2": Counter(b=1)})
    kinds = {(m["order"], m["kind"], m["sku"]) for m in got}
    assert kinds == {("o1", "wrong", "x"), ("o1", "missed", "b"), ("o2", "missed", "b")}
    why = {m["order"]: m["why"] for m in got if m["kind"] == "missed"}
    assert "right" in why["o1"] and "not counted" in why["o2"]
    assert len(mistakes(run, {"o1": Counter(b=1), "o2": Counter(b=1)}, n=1)) == 1


def test_viewer_builds_offline_page(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    (data / "videos").mkdir(parents=True)
    (data / "gallery").mkdir()
    (out / "baseline").mkdir(parents=True)
    Image.new("RGB", (200, 200), (255, 255, 255)).save(data / "gallery" / "b.png")
    write_video([np.full((60, 80, 3), 90, np.uint8)] * 6, data / "videos" / "o1_cam0.mp4", 10)
    (data / "orders.csv").write_text("video_id,sku,qty\no1,b,1\n")
    (out / "o1_cam0.json").write_text(json.dumps(result([track(1, "b", True)])))
    old = result([track(1, None, True, "</script>")])
    del old["rejected_detections"]
    (out / "baseline" / "o1_cam0.json").write_text(json.dumps(old))
    page = build(data, out, use_models=False).read_text(encoding="utf-8")
    payload = json.loads(page.split("const D = ", 1)[1].split(";\nconst $", 1)[0])
    assert payload["order"] == ["gdino", "before"]
    assert payload["runs"]["gdino"]["name"].startswith("After fixes")
    assert payload["runs"]["gdino"]["summary"]["exact"] == 1
    assert payload["runs"]["before"]["summary"]["exact"] == 0
    assert payload["orders"][0]["videos"][0]["src"] == "../../data/videos/o1_cam0.mp4"
    assert "</script>b" not in page and (out / "viewer" / "thumbs" / "b.jpg").exists()
    assert (out / "viewer" / "crops" / "gdino_o1_cam0_t1.jpg").exists()


def test_sheet_has_import() -> None:
    page = render([], ["a"], "x")
    assert "Import labels.json" in page and "could not import" in page
