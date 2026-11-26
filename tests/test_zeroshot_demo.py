"""Step 1 demo logic with fake models and tiny synthetic frames (no downloads, no real data)."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from hello_vision.detect.types import Detection
from hello_vision.detect.zeroshot import clean, drop_containers, label_of, nms
from hello_vision.identify.index import GalleryIndex, Match
from hello_vision.synth.session import write_video
from hello_vision.track.simple import ByteSimpleTracker
from sim.orders import OrderScore, count_items, merge_cameras, order_id, report
from sim.run_video import (
    Models,
    RotatedFaces,
    process_video,
    read_orders,
    speed_line,
    summarise_tracks,
)


def d(
    label: str, x: float, y: float, w: float = 20, h: float = 40, score: float = 0.9
) -> Detection:
    return Detection(label, score, x, y, x + w, y + h)


def test_nms_merges_same_label_only() -> None:
    kept = nms([d("product", 0, 0, score=0.5), d("product", 1, 1), d("hand", 0, 0)])
    assert [(k.label, k.score) for k in kept] == [("product", 0.9), ("hand", 0.9)]


def test_containers_and_implausible_boxes_removed() -> None:
    row = d("product", 0, 0, 100, 50)
    items = [d("product", 5, 5, 20, 40), d("product", 60, 5, 20, 40)]
    assert drop_containers([row, *items]) == items
    out = clean(
        [d("product", 0, 0, 190, 90), d("product", 3, 3, 4, 4), d("hand", 0, 0, 190, 90)], 200, 100
    )
    assert [o.label for o in out] == ["hand"]


def test_tracker_keeps_id_through_gaps_and_low_scores() -> None:
    tr = ByteSimpleTracker("product", high_score=0.4)
    path = [(10, 0.9), (20, 0.9), (30, 0.2), (None, 0), (50, 0.9)]
    ids = []
    for f, (x, s) in enumerate(path):
        dets = [] if x is None else [d("product", x, 10, score=s)]
        got = tr.update(f, dets)
        ids += [t.id for t in got.values()]
    assert ids == [1, 1, 1, 1] and len(tr.tracks) == 1


def test_low_score_detection_does_not_start_a_track() -> None:
    tr = ByteSimpleTracker("product", high_score=0.4)
    assert tr.update(0, [d("product", 0, 0, score=0.2)]) == {}


def test_sku_vote_is_confidence_weighted() -> None:
    tr = ByteSimpleTracker("product")
    seq = [("a", 0.6), ("b", 0.9), ("a", 0.5), ("a", 0.5)]
    for f, (sku, s) in enumerate(seq):
        tr.update(f, [d("product", 0, 0)], [Match(sku, s, 0.1, sku)])
    sku, conf, guess = tr.tracks[0].sku_vote()
    assert sku == "a" and guess == "a" and 0 < conf < 1
    tr2 = ByteSimpleTracker("product")
    tr2.update(0, [d("product", 0, 0)], [Match(None, 0.2, 0.0, "c")])
    assert tr2.tracks[0].sku_vote() == (None, 0.0, "c")


class ScriptedDetector:
    """A product that sits still, is grabbed by a hand, moves to the edge and disappears."""

    threshold = 0.2

    def __init__(self) -> None:
        self.i = -1

    def start_session(self, first_image: np.ndarray) -> None:
        self.i = -1

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.i += 1
        i = self.i
        still = d("product", 150, 20)  # never moves: not taken
        if i < 4:
            return [d("product", 40, 20), still]
        if i < 9:
            y = 20 + (i - 3) * 16
            return [d("product", 40, y), d("hand", 35, y + 10, 30, 30), still]
        return [still]


class FakeEmbedder:
    def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        return [np.array([1.0, 0.0], np.float32) for _ in crops]


def test_process_video_end_to_end(tmp_path: Path) -> None:
    frames = [np.full((120, 200, 3), 90, np.uint8) for _ in range(30)]
    write_video(frames, tmp_path / "v.mp4", fps=10)
    index = GalleryIndex(threshold=0.5)
    index.add("cola", np.array([1.0, 0.0], np.float32))
    models = Models(ScriptedDetector(), FakeEmbedder(), index, "fake")
    r = process_video(tmp_path / "v.mp4", tmp_path / "out" / "v.mp4", models, fps=5)
    assert r.processed_frames == 15
    assert r.cart == {"cola": 1}
    report = json.loads((tmp_path / "out" / "v.json").read_text())
    assert sum(t["taken_guess"] for t in report["tracks"]) == 1
    assert Path(r.annotated).stat().st_size > 0 and Path(r.sheet).stat().st_size > 0
    assert "processed fps" in speed_line(r)


def test_orders_and_compare(tmp_path: Path) -> None:
    (tmp_path / "o.csv").write_text("video_id,sku,qty\nv1.mp4,cola,2\nv1,chips,1\n")
    orders = read_orders(tmp_path / "o.csv")
    assert orders == {"v1": Counter({"cola": 2, "chips": 1})}
    score = OrderScore("v1", Counter({"cola": 1, "gum": 1}), orders["v1"], {"cola"})
    assert (score.correct, score.wrong, score.missed, score.exact) == (1, 1, 2, False)
    text = report([score])
    assert "0/1 = 0.0%" in text and "chips -> gum: 1" in text


def track(sku: str, first: float, last: float, taken: bool = True) -> dict:  # type: ignore[type-arg]
    return {"sku": sku, "first_seen_s": first, "last_seen_s": last, "taken_guess": taken}


def test_count_items_joins_fragments_but_keeps_simultaneous_items() -> None:
    tracks = [
        track("cola", 1, 3), track("cola", 4, 6),  # same item, lost and found again
        track("gum", 1, 5), track("gum", 2, 6),  # two at once
        track("chips", 1, 2), track("chips", 20, 22),  # taken far apart
        track("milk", 1, 2, taken=False),
    ]  # fmt: skip
    assert count_items(tracks) == Counter({"cola": 1, "gum": 2, "chips": 2})


def test_merge_cameras_and_order_ids() -> None:
    assert merge_cameras([Counter(a=1, b=2), Counter(a=3)]) == Counter(a=3, b=2)
    assert order_id("order-1_cam1") == "order-1" and order_id("demo-01") == "demo-01"


def test_distractors_suppress_products_and_labels_map() -> None:
    assert [label_of(n) for n in ("human head", "mobile phone", "hand", "drink can")] == [
        "head", "phone", "hand", "product"
    ]  # fmt: skip
    dets = [d("product", 10, 10), d("phone", 10, 10), d("product", 100, 10), d("head", 150, 10)]
    assert [x.label for x in clean(dets, 400, 200)] == ["product", "head"]


def test_rotated_face_boxes_map_back() -> None:
    class Marker:
        def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
            ys, xs = np.nonzero(image[..., 0])
            return [(int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)]

    img = np.zeros((60, 100, 3), np.uint8)
    img[10:20, 30:50] = 255
    assert RotatedFaces(Marker()).detect(img) == [(30, 10, 20, 10)] * 4


def test_shelf_line_rule_for_taken_items() -> None:
    products, hands = ByteSimpleTracker("product"), ByteSimpleTracker("hand")
    for f in range(4):
        dets = [d("product", 20, 10), d("product", 150, 150), d("hand", 140, 150, 40, 40)]
        products.update(f, dets)
        hands.update(f, dets)
    times = [0.0, 0.2, 0.4, 0.6]
    out = summarise_tracks(products.tracks, hands.tracks, times, (200, 200), shelf_line=0.45)
    assert [t["taken_guess"] for t in out] == [False, True]  # stock on top, held item below
