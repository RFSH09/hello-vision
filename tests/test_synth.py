import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.synth.compose import compose_scene, generate_dataset
from hello_vision.synth.session import SCENARIOS, render_session

from .conftest import cached_clip


@pytest.fixture(scope="module")
def sprites() -> dict[str, Image.Image]:
    return placeholder_sprites()


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_truth_matches_script(scenario: str, sprites: dict[str, Image.Image]) -> None:
    clip = cached_clip(scenario)
    t = clip.truth
    assert len(clip.frames) == len(clip.boxes) == t.num_frames
    assert t.door_close_frame == t.num_frames - 1
    net: dict[str, int] = {}
    for a in t.actions:
        net[a.sku] = net.get(a.sku, 0) + (1 if a.kind == "take" else -1)
    assert {c.sku: c.qty for c in t.cart} == {k: v for k, v in net.items() if v > 0}


def test_take_and_return_nets_to_empty_cart(sprites: dict[str, Image.Image]) -> None:
    clip = cached_clip("take_and_return")
    assert clip.truth.cart == [] and len(clip.truth.actions) == 2


def test_shelf_product_count_drops_after_take(sprites: dict[str, Image.Image]) -> None:
    clip = cached_clip("take_one")
    first = sum(b.label == "product" for b in clip.boxes[0])
    last = sum(b.label == "product" for b in clip.boxes[-1])
    assert first == 6 and last == 5
    assert not any(b.label == "hand" for b in clip.boxes[0])
    assert any(b.label == "hand" for b in clip.boxes[20])


def test_deterministic(sprites: dict[str, Image.Image]) -> None:
    a = cached_clip("take_two")
    b = render_session("take_two", sprites, seed=0)
    c = cached_clip("take_nothing", 1)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames, strict=True))
    assert not np.array_equal(a.frames[0], c.frames[0])


def test_low_light_is_darker(sprites: dict[str, Image.Image]) -> None:
    bright = cached_clip("take_one").frames[0].mean()
    dark = cached_clip("take_one_low_light").frames[0].mean()
    assert dark < bright * 0.8


def test_render_errors(sprites: dict[str, Image.Image]) -> None:
    with pytest.raises(KeyError):
        render_session("nope", sprites)
    with pytest.raises(ValueError):
        render_session("take_one", dict(list(sprites.items())[:3]))


def test_scene_boxes_inside_image(sprites: dict[str, Image.Image]) -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        img, boxes = compose_scene(sprites, rng)
        assert any(b.label == "product" for b in boxes)
        for b in boxes:
            assert 0 <= b.x1 < b.x2 <= img.width and 0 <= b.y1 < b.y2 <= img.height


def test_generate_dataset_coco(tmp_path: Path, sprites: dict[str, Image.Image]) -> None:
    ann = generate_dataset(tmp_path, sprites, count=3, seed=1)
    data = json.loads(ann.read_text())
    assert len(data["images"]) == 3 and data["annotations"]
    assert {c["name"] for c in data["categories"]} == {"product", "hand"}
    assert all((tmp_path / im["file_name"]).exists() for im in data["images"])
