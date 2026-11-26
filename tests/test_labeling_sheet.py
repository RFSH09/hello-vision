"""Crop selection, the labelling page and leave-one-order-out scoring (synthetic data only)."""

from collections import Counter
from pathlib import Path

import numpy as np

from hello_vision.datatools.crops import (
    Candidate,
    held_outside,
    pad_crop,
    pick,
    track_candidates,
)
from hello_vision.datatools.label_sheet import UNSURE, render
from hello_vision.synth.session import write_video
from tools.apply_labels import evaluate, labelled_crops, vote


def row(t: float, y: float, held: int) -> list[float]:
    return [t, 10, y, 30, y + 20, 0.5, held]


def test_held_outside_needs_hand_and_below_shelf() -> None:
    rows = [row(0, 10, 1), row(1, 80, 0), row(2, 80, 1)]
    assert held_outside(rows, 100) == [rows[2]]


def test_pick_prefers_quality_spread_in_time() -> None:
    img = np.zeros((4, 4, 3), np.uint8)
    cands = [Candidate(t, img, q) for t, q in [(0.0, 9), (0.1, 8), (1.0, 5), (2.0, 1)]]
    assert [c.time_s for c in pick(cands, n=3)] == [0.0, 1.0, 2.0]
    assert len(pick(cands[:2], n=3)) == 2


def test_pad_crop_is_clipped() -> None:
    img = np.zeros((50, 50, 3), np.uint8)
    assert pad_crop(img, (-10, -10, 60, 60)).shape == (50, 50, 3)


def test_track_candidates_reads_the_right_frames(tmp_path: Path) -> None:
    frames = [np.full((100, 100, 3), i * 20, np.uint8) for i in range(10)]
    write_video(frames, tmp_path / "v.mp4", fps=10)
    tracks = [
        {"track_id": 1, "boxes": [row(0.2, 60, 1), row(0.5, 60, 1)]},
        {"track_id": 2, "boxes": [row(0.2, 60, 1)]},  # too few
    ]
    found = track_candidates(tmp_path / "v.mp4", tracks, prepare=lambda f: f)
    assert list(found) == [1]
    means = sorted(c.crop.mean() for c in found[1])
    assert abs(means[0] - 40) < 8 and abs(means[1] - 100) < 8


def test_page_embeds_tasks_safely() -> None:
    task = {"id": "v:1", "order": "</script><b>", "video": "v", "crops": ["c.jpg"],
            "order_items": {"a": 1}, "suggested": "a"}  # fmt: skip
    page = render([task], ["a", "b"], "s")
    assert "</script><b>" not in page and "Download labels.json" in page and '"v:1"' in page


def test_labelled_crops_drops_unsure_and_excluded() -> None:
    tasks = [
        {"id": "1", "crops": ["x", "y"]},
        {"id": "2", "crops": ["z"]},
        {"id": "3", "crops": []},
    ]
    labels = {"labels": {"1": "a", "2": UNSURE}, "excluded": ["y"]}
    assert labelled_crops(tasks, labels) == {"1": ["x"]}


def v(x: float, y: float) -> np.ndarray:
    a = np.array([x, y], np.float32)
    return np.asarray(a / np.linalg.norm(a), np.float32)


def test_leave_one_order_out_uses_only_other_orders() -> None:
    tasks = [{"id": f"{o}_cam0:1", "order": o, "video": f"{o}_cam0"} for o in ("o1", "o2")]
    labels = {t["id"]: "b" for t in tasks}
    vectors = {t["id"]: [v(1, 1)] for t in tasks}
    base = (["a", "b"], [v(1, 0.2), v(0, 1)])  # packshots alone confuse these crops with "a"
    tracks = {t["id"]: {"first_seen_s": 0, "last_seen_s": 1} for t in tasks}
    orders = {o: Counter({"b": 1}) for o in ("o1", "o2")}
    args = (tasks, labels, vectors, base, tracks, orders)
    _, right_base, total = evaluate(*args, False, 0.0, 0.0)
    scores, right_real, _ = evaluate(*args, True, 0.0, 0.0)
    assert (right_base, right_real, total) == (0, 2, 2) and all(s.exact for s in scores)
    # with a single order there is nothing from elsewhere to learn from
    only = evaluate(tasks[:1], labels, vectors, base, tracks, {"o1": orders["o1"]}, True, 0.0, 0.0)
    assert only[1] == 0
    assert vote_none()


def vote_none() -> bool:
    from hello_vision.identify.index import GalleryIndex

    index = GalleryIndex(threshold=2.0)
    index.add("a", v(1, 0))
    return vote(index, [v(1, 0)]) is None
