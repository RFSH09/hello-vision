import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from eval.offline_m2 import run as offline_run
from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.detect.baseline import BaselineDetector
from hello_vision.detect.metrics import match
from hello_vision.detect.onnx_detector import OnnxDetector
from hello_vision.detect.types import Detection, iou
from hello_vision.identify.embedders import ColorLayoutEmbedder, OnnxEmbedder
from hello_vision.identify.index import GalleryIndex, build_index, crop, identify_all
from hello_vision.identify.metrics import confusion, overall_accuracy, per_sku_accuracy
from hello_vision.synth.compose import generate_dataset
from training.build_embedding_dataset import build
from training.common import CsvLogger, set_seed

from .conftest import cached_clip


def d(label: str, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(label, 1.0, x1, y1, x2, y2)


def test_iou_and_properties() -> None:
    a, b = d("product", 0, 0, 10, 10), d("product", 5, 0, 15, 10)
    assert iou(a, b) == pytest.approx(1 / 3)
    assert iou(a, d("product", 20, 20, 30, 30)) == 0.0
    assert a.center == (5, 5) and a.area == 100


def test_match_counts() -> None:
    truth = [d("product", 0, 0, 10, 10), d("product", 50, 50, 60, 60), d("hand", 0, 0, 5, 5)]
    pred = [d("product", 1, 1, 10, 10), d("product", 100, 100, 110, 110)]
    res, pairs = match(truth, pred)
    assert (res["product"].tp, res["product"].fp, res["product"].fn) == (1, 1, 1)
    assert res["hand"].recall == 0.0 and res["product"].precision == 0.5
    assert pairs == {0: 0}


def test_baseline_detector_on_synthetic_clip() -> None:
    clip = cached_clip("take_one")
    det = BaselineDetector()
    det.start_session(clip.frames[0])
    tp = fn = 0
    for img, boxes in zip(clip.frames, clip.boxes, strict=True):
        gts = [d(b.label, b.x1, b.y1, b.x2, b.y2) for b in boxes]
        res, _ = match(gts, det.detect(img))
        tp += sum(r.tp for r in res.values())
        fn += sum(r.fn for r in res.values())
    assert tp / (tp + fn) > 0.9


def test_baseline_autocalibrates() -> None:
    clip = cached_clip("take_nothing")
    assert len([x for x in BaselineDetector().detect(clip.frames[0]) if x.label == "product"]) == 6


class FakeSession:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.feed: dict[str, Any] = {}

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        self.feed = input_feed
        return self.outputs


def test_onnx_detector_pre_and_post_processing() -> None:
    out = [
        np.array([[0, 1, 0]]),
        np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]]], dtype=np.float32),
        np.array([[0.9, 0.5, 0.1]], dtype=np.float32),
    ]
    sess = FakeSession(out)
    det = OnnxDetector(sess, input_size=32)
    dets = det.detect(np.zeros((48, 64, 3), np.uint8))
    assert [x.label for x in dets] == ["product", "hand"]  # third is below the threshold
    assert sess.feed["images"].shape == (1, 3, 32, 32)
    assert sess.feed["orig_target_sizes"].tolist() == [[64, 48]]
    det.start_session(np.zeros((2, 2, 3), np.uint8))


def test_color_embedder_normalised_and_brightness_robust() -> None:
    sp = placeholder_sprites()["demo_blue_can"]
    canvas = Image.new("RGB", sp.size, (175, 175, 170))
    canvas.paste(sp.convert("RGB"), (0, 0), sp.getchannel("A"))
    img = np.asarray(canvas, dtype=np.uint8)
    emb = ColorLayoutEmbedder()
    v, dark = emb.embed(img), emb.embed((img * 0.5).astype(np.uint8))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert float(v @ dark) > 0.97


def test_onnx_embedder_with_fake_session() -> None:
    sess = FakeSession([np.array([[3.0, 4.0]], dtype=np.float32)])
    v = OnnxEmbedder(sess, input_size=16).embed(np.zeros((20, 10, 3), np.uint8))
    assert v.tolist() == pytest.approx([0.6, 0.8]) and sess.feed["input"].shape == (1, 3, 16, 16)


def test_index_identifies_and_flags_unknown() -> None:
    sprites = placeholder_sprites()
    emb = ColorLayoutEmbedder()
    index = build_index({k: [v] for k, v in sprites.items() if k != "demo_gold_bar"}, emb, 0.9)
    assert len(index) == 8 and "demo_gold_bar" not in index.skus
    clip = cached_clip("take_nothing")
    boxes = [b for b in clip.boxes[0] if b.label == "product"]
    matches = identify_all(clip.frames[0], [(b.x1, b.y1, b.x2, b.y2) for b in boxes], emb, index)
    assert [m.sku for m in matches] == [b.sku for b in boxes]
    # a vector far from every gallery entry must come back as unknown, not a forced match
    far = np.zeros_like(emb.embed(clip.frames[0][:50, :50]))
    far[0] = 1.0
    m = index.query(far)
    assert m.sku is None and m.best_sku is not None


def test_index_margin_rejects_ambiguous_and_empty_raises() -> None:
    idx = GalleryIndex(threshold=0.5, min_margin=0.2)
    idx.add("a", np.array([1, 0], np.float32))
    idx.add("b", np.array([0.99, 0.14], np.float32) / np.float32(1.0))
    m = idx.query(np.array([1, 0], np.float32))
    assert m.sku is None and m.best_sku == "a"
    with pytest.raises(ValueError):
        GalleryIndex().query(np.array([1, 0], np.float32))


def test_crop_clips() -> None:
    img = np.zeros((10, 10, 3), np.uint8)
    assert crop(img, (-5, -5, 3, 3)).shape[:2] == (3, 3)
    assert crop(img, (9.9, 9.9, 9.9, 9.9)).shape[0] >= 1


def test_identify_metrics() -> None:
    conf = confusion(["a", "a", "b", "b"], ["a", None, "b", "a"])
    assert conf["a"] == Counter({"a": 1, "<unknown>": 1})
    assert per_sku_accuracy(conf) == {"a": 0.5, "b": 0.5} and overall_accuracy(conf) == 0.5
    assert overall_accuracy({}) == 0.0
    with pytest.raises(ValueError):
        confusion(["a"], [])


def test_training_helpers(tmp_path: Path) -> None:
    set_seed(1)
    log = CsvLogger(tmp_path / "run" / "m.csv")
    log.log(epoch=0, loss=1.0)
    log.log(epoch=1, loss=0.5)
    with pytest.raises(ValueError):
        log.log(epoch=2, acc=1)
    assert (tmp_path / "run" / "m.csv").read_text().splitlines()[0] == "epoch,loss"

    ann = generate_dataset(tmp_path / "ds", placeholder_sprites(), count=2, seed=0)
    counts = build(ann, tmp_path / "ds", tmp_path / "crops")
    assert sum(counts.values()) > 0
    assert all(any((tmp_path / "crops" / s).glob("*.png")) for s in counts)
    assert json.loads(ann.read_text())["annotations"]


def test_offline_report_runs() -> None:
    report = offline_run(scenarios=["take_nothing"])
    assert "| product |" in report and "Identification top-1" in report
