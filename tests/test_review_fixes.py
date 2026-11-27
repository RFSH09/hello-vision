"""Fixes from the first review: product filter, appearance tracking, blur protection, multiples."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hello_vision.datatools.label_sheet import NOT_PRODUCT, UNSURE
from hello_vision.datatools.real_crops import labelled_ids, load_or_build
from hello_vision.detect.types import Detection
from hello_vision.identify.index import GalleryIndex
from hello_vision.identify.product_filter import ProductFilter, leave_one_group_out
from hello_vision.privacy.blur import FaceBlurrer
from hello_vision.synth.session import write_video
from hello_vision.track.simple import ByteSimpleTracker
from sim.orders import OrderScore, report
from sim.run_video import Models, flag_multiple, process_video


def unit(*values: float) -> np.ndarray:
    a = np.array(values, np.float32)
    return np.asarray(a / np.linalg.norm(a), np.float32)


def blobs(n: int = 20) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(0)
    x = [unit(*(np.array([1, 0, 0]) + rng.normal(0, 0.1, 3))) for _ in range(n)]
    x += [unit(*(np.array([0, 1, 0]) + rng.normal(0, 0.1, 3))) for _ in range(n // 2)]
    return np.stack(x), [1] * n + [0] * (n // 2)


def test_product_filter_separates_and_needs_both_classes() -> None:
    x, y = blobs()
    p = ProductFilter().fit(x, y).predict(np.stack([unit(1, 0, 0), unit(0, 1, 0)]))
    assert p[0] > 0.8 and p[1] < 0.2
    with pytest.raises(ValueError):
        ProductFilter().fit(x[:3], [1, 1, 1])
    with pytest.raises(RuntimeError):
        ProductFilter().predict(x)
    groups = ["a" if i % 2 else "b" for i in range(len(y))]
    loo = leave_one_group_out(x, y, groups)
    assert np.mean((loo >= 0.5) == np.array(y, bool)) > 0.9


def det(x: float, score: float = 0.9) -> Detection:
    return Detection("product", score, x, 10, x + 20, 50)


def test_appearance_splits_a_drifting_track() -> None:
    tracker = ByteSimpleTracker("product", min_appearance=0.5)
    bottle, shoe = unit(1, 0), unit(0, 1)
    for f, v in enumerate([bottle, bottle, shoe, shoe]):
        tracker.update(f, [det(10 + f)], vectors=[v])  # same place, then another object
    assert [len(t.dets) for t in tracker.tracks] == [2, 2]
    plain = ByteSimpleTracker("product")
    for f in range(4):
        plain.update(f, [det(10 + f)])
    assert len(plain.tracks) == 1


def test_appearance_prefers_the_matching_track() -> None:
    tracker = ByteSimpleTracker("product", min_appearance=0.5)
    red, blue = unit(1, 0.3), unit(0.3, 1)
    tracker.update(0, [det(10), det(20)], vectors=[red, blue])
    got = tracker.update(1, [det(16), det(14)], vectors=[red, blue])  # boxes crossed over
    assert got[0].id == 1 and got[1].id == 2


class OneFace:
    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [(20, 20, 20, 20)]


def test_blur_leaves_a_false_face_on_a_product_sharp() -> None:
    img = np.random.default_rng(1).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    blurrer = FaceBlurrer(OneFace(), margin=0)
    assert not np.array_equal(blurrer.blur(img), img)
    assert np.array_equal(blurrer.blur(img, protect=[(10, 10, 60, 60)]), img)
    assert not np.array_equal(blurrer.blur(img, protect=[(0, 0, 25, 25)]), img)  # barely touches
    fixed = FaceBlurrer(None, [(0, 0, 0.5, 0.5)])
    assert not np.array_equal(fixed.blur(img, protect=[(0, 0, 80, 80)]), img)  # always blurred


def track(tid: int, x: int, taken: bool = True) -> dict[str, Any]:
    boxes = [[t, x, 100, x + 20, 140, 0.9, 1] for t in (0.0, 0.2, 0.4)]
    return {"track_id": tid, "taken_guess": taken, "boxes": boxes}


def test_items_held_together_are_flagged() -> None:
    tracks = [track(1, 10), track(2, 45), track(3, 500), track(4, 12, taken=False)]
    flag_multiple(tracks)
    assert [t["multiple"] for t in tracks] == [True, True, False, False]
    from collections import Counter

    text = report([OrderScore("o", Counter(), Counter(a=1), multiple=True)])
    assert "| REVIEW |" in text


class StillHeld:
    threshold = 0.2

    def start_session(self, first_image: np.ndarray) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[Detection]:
        hand = Detection("hand", 0.9, 30, 70, 60, 100)
        giant = Detection("hand", 0.9, 0, 0, 200, 110)
        return [Detection("product", 0.9, 40, 70, 60, 105), hand, giant]


class Embed:
    def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        return [unit(1, 0) for _ in crops]


def test_filter_rejects_before_identification_and_drops_giant_hands(tmp_path: Path) -> None:
    write_video([np.full((120, 200, 3), 90, np.uint8)] * 10, tmp_path / "o_cam0.mp4", fps=10)
    index = GalleryIndex(threshold=0.5)
    index.add("cola", unit(1, 0))
    keep = ProductFilter().fit(np.stack([unit(1, 0), unit(0, 1)]), [1, 0])
    drop = ProductFilter().fit(np.stack([unit(1, 0), unit(0, 1)]), [0, 1])
    for classifier, expected in ((keep, {"cola": 1}), (drop, {})):
        models = Models(StillHeld(), Embed(), index, "fake", product_filter=classifier)
        r = process_video(tmp_path / "o_cam0.mp4", tmp_path / "out.mp4", models, fps=5)
        assert r.cart == expected
    assert len({tuple(h[1:5]) for h in r.hands}) == 1  # the frame-sized "hand" was ignored
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["rejected_detections"] == 5


class ColourEmbed:
    calls = 0

    def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        ColourEmbed.calls += 1
        return [unit(float(c.mean()) + 1, 10) for c in crops]


def test_labelled_crops_are_leave_one_order_out(tmp_path: Path) -> None:
    videos, results = tmp_path / "videos", tmp_path / "results"
    videos.mkdir()
    results.mkdir()
    labels = {"o1_cam0:1": "a", "o2_cam0:1": NOT_PRODUCT, "o3_cam0:1": "b", "o3_cam0:2": UNSURE}
    assert "o3_cam0:2" not in labelled_ids(labels)
    for i, order in enumerate(("o1", "o2", "o3")):
        write_video(
            [np.full((100, 200, 3), 60 * i + 20, np.uint8)] * 8, videos / f"{order}_cam0.mp4", 10
        )
        boxes = [[t, 20, 60, 40, 90, 0.9, 1] for t in (0.1, 0.3)]
        tracks = [{"track_id": k, "boxes": boxes} for k in (1, 2)]
        (results / f"{order}_cam0.json").write_text(json.dumps({"tracks": tracks}))
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": labels}))
    crops = load_or_build(path, results, videos, ColourEmbed(), tmp_path / "cache")
    assert len(crops) == 6 and set(crops.orders) == {"o1", "o2", "o3"}
    again = load_or_build(path, results, videos, ColourEmbed(), tmp_path / "cache")
    assert ColourEmbed.calls == 1 and again.labels == crops.labels  # cached
    base = GalleryIndex(0.1)
    base.add("packshot", unit(1, 1))
    assert crops.gallery(base, "o1").skus == ["b", "packshot"]  # o1's own crops are left out
    assert crops.gallery(base, None).skus == ["a", "b", "packshot"]
    assert crops.product_filter("o1").weights is not None
