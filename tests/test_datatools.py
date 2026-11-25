import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.run import folder_evals
from hello_vision.catalog.packshots import placeholder_catalog
from hello_vision.datatools.labeling import (
    LabeledBox,
    LabeledFrame,
    SessionLabel,
    export_coco,
    import_coco,
    import_label_studio,
    preannotations,
)
from hello_vision.datatools.weak_labels import PrivacyError, build_weak_sessions
from hello_vision.detect.types import Detection
from hello_vision.session.types import AnnotatedDetection
from hello_vision.synth.session import write_video
from sim.pipeline import build_demo_analyzer
from tools.labeling_convert import main as convert_main
from tools.weak_labels import main as weak_main

from .conftest import cached_clip


def frame() -> LabeledFrame:
    return LabeledFrame(
        file_name="a.jpg",
        width=100,
        height=50,
        boxes=[
            LabeledBox(label="product", sku="demo_cola_bottle", x1=10, y1=5, x2=30, y2=45),
            LabeledBox(label="hand", x1=40, y1=10, x2=90, y2=50),
        ],
    )


def test_coco_roundtrip(tmp_path: Path) -> None:
    export_coco(
        [frame(), LabeledFrame(file_name="b.jpg", width=100, height=50, boxes=[])],
        tmp_path / "c.json",
    )
    back = import_coco(tmp_path / "c.json")
    assert back[0] == frame() and back[1].boxes == []
    raw = json.loads((tmp_path / "c.json").read_text())
    assert raw["annotations"][0]["attributes"] == {"sku": "demo_cola_bottle"}  # CVAT attribute


def test_coco_from_cvat_style_attributes_and_bad_category(tmp_path: Path) -> None:
    doc = {
        "images": [{"id": 1, "file_name": "x.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "product"}, {"id": 2, "name": "banana"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 4, 4],
                "attributes": {"sku": "s1"},
            }
        ],
    }
    (tmp_path / "c.json").write_text(json.dumps(doc))
    assert import_coco(tmp_path / "c.json")[0].boxes[0].sku == "s1"
    doc["annotations"][0]["category_id"] = 2
    (tmp_path / "c.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unknown category"):
        import_coco(tmp_path / "c.json")


def test_box_validation() -> None:
    with pytest.raises(ValidationError):
        LabeledBox(label="product", x1=5, y1=0, x2=5, y2=3)
    with pytest.raises(ValidationError):
        LabeledBox(label="hand", sku="a", x1=0, y1=0, x2=3, y2=3)


def test_label_studio_import(tmp_path: Path) -> None:
    def rect(i: str, label: str, x: float) -> dict:  # type: ignore[type-arg]
        return {"id": i, "type": "rectanglelabels", "original_width": 200, "original_height": 100,
                "value": {"x": x, "y": 10, "width": 20, "height": 50, "rectanglelabels": [label]}}  # fmt: skip

    tasks = [
        {
            "data": {"image": "/data/upload/frame1.jpg"},
            "annotations": [
                {
                    "result": [
                        rect("a", "product", 10),
                        {"id": "a", "from_name": "sku", "value": {"choices": ["demo_gold_bar"]}},
                        rect("b", "product:demo_blue_can", 50),
                        rect("c", "hand", 70),
                    ]
                }
            ],
        }
    ]
    (tmp_path / "ls.json").write_text(json.dumps(tasks))
    f = import_label_studio(tmp_path / "ls.json")[0]
    assert f.file_name == "frame1.jpg" and (f.width, f.height) == (200, 100)
    assert [(b.label, b.sku) for b in f.boxes] == [
        ("product", "demo_gold_bar"), ("product", "demo_blue_can"), ("hand", None)
    ]  # fmt: skip
    assert f.boxes[0].x1 == pytest.approx(20) and f.boxes[0].x2 == pytest.approx(60)
    tasks[0]["annotations"][0]["result"][0]["value"]["rectanglelabels"] = ["banana"]
    (tmp_path / "ls.json").write_text(json.dumps(tasks))
    with pytest.raises(ValueError):
        import_label_studio(tmp_path / "ls.json")


def test_session_label_consistency() -> None:
    ok = SessionLabel(
        session_id="s1",
        cart={"a": 1},
        actions=[
            {"kind": "take", "sku": "a", "frame": 3},
            {"kind": "take", "sku": "b", "frame": 9},
            {"kind": "return", "sku": "b", "frame": 20},
        ],  # type: ignore[list-item]
    )
    assert ok.cart == {"a": 1}
    with pytest.raises(ValidationError):
        SessionLabel(
            session_id="s1", cart={"a": 2}, actions=[{"kind": "take", "sku": "a", "frame": 1}]
        )  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        SessionLabel(session_id="s1", cart={}, actions=[{"kind": "return", "sku": "a", "frame": 1}])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        SessionLabel(session_id="../x", cart={})
    with pytest.raises(ValidationError):
        SessionLabel(session_id="s", cart={"a": 0})


def test_preannotations_from_pipeline_output(tmp_path: Path) -> None:
    dets = [[
        AnnotatedDetection(Detection("product", 1, 0, 0, 10, 10), 1, "demo_cola_bottle"),
        AnnotatedDetection(Detection("hand", 1, 20, 20, 40, 40), 2, None),
        AnnotatedDetection(Detection("product", 1, 5, 5, 5, 9), 3, None),  # degenerate: dropped
    ]]  # fmt: skip
    f = preannotations(["f0.jpg"], dets, 100, 100)[0]
    assert [(b.label, b.sku) for b in f.boxes] == [("product", "demo_cola_bottle"), ("hand", None)]


def test_convert_cli(tmp_path: Path) -> None:
    export_coco([frame()], tmp_path / "c.json")
    assert (
        convert_main(["cvat", "--in", str(tmp_path / "c.json"), "--out", str(tmp_path / "f.json")])
        == 0
    )
    assert json.loads((tmp_path / "f.json").read_text())[0]["file_name"] == "a.jpg"


def write_records(path: Path, rows: list[dict]) -> Path:  # type: ignore[type-arg]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_weak_labels_accept_reject_and_eval_roundtrip(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    clip = cached_clip("take_one")
    write_video(clip.frames, videos / "p1.mp4")
    (tmp_path / "secret.mp4").write_bytes(b"x")
    sku = clip.truth.cart[0].sku
    recs = write_records(tmp_path / "r.jsonl", [
        {"session_id": "p1", "video": "p1.mp4", "items": [{"sku": sku, "qty": 1}]},
        {"session_id": "p1", "video": "p1.mp4", "items": []},
        {"session_id": "p2", "video": "nope.mp4", "items": []},
        {"session_id": "p3", "video": "p1.mp4", "items": [{"sku": "mystery", "qty": 1}]},
        {"session_id": "p4", "video": "../secret.mp4", "items": []},
        {"session_id": "p5", "video": "p1.mp4", "items": [{"sku": sku, "qty": 0}]},
    ])  # fmt: skip
    out = tmp_path / "weak"
    rep = build_weak_sessions(recs, placeholder_catalog(), videos, out)
    assert rep.accepted == ["p1"]
    assert rep.rejected == {
        "duplicate_session_id": ["p1"], "missing_video": ["p2", "p4"],
        "unknown_sku": ["p3"], "bad_quantity": ["p5"],
    }  # fmt: skip
    truth = json.loads((out / "p1.truth.json").read_text())
    assert truth["label_quality"] == "weak" and truth["cart"] == [{"sku": sku, "qty": 1}]
    results = folder_evals(build_demo_analyzer(), out)  # weak labels feed the evaluation harness
    assert [r.session_id for r in results] == ["p1"] and results[0].pred == results[0].truth


def test_weak_labels_refuse_personal_fields(tmp_path: Path) -> None:
    recs = write_records(
        tmp_path / "r.jsonl", [{"session_id": "x", "video": "v", "items": [], "phone_number": "1"}]
    )
    with pytest.raises(PrivacyError, match="phone_number"):
        build_weak_sessions(recs, placeholder_catalog(), tmp_path, tmp_path / "o")
    cat = tmp_path / "cat.yaml"
    placeholder_catalog().dump(cat)
    assert (
        weak_main(
            [
                "--records",
                str(recs),
                "--catalog",
                str(cat),
                "--videos",
                str(tmp_path),
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )


def test_folder_evals_missing_video_is_loud(tmp_path: Path) -> None:
    (tmp_path / "x.truth.json").write_text(json.dumps({"cart": {}}))
    with pytest.raises(FileNotFoundError):
        folder_evals(build_demo_analyzer(), tmp_path)
