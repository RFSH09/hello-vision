"""DINOv2 gallery logic with a fake embedder (no torch, no downloads)."""

from pathlib import Path

import numpy as np
from PIL import Image

from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.identify.gallery import DinoEmbedder, augment_view, build_gallery, load_packshots


class MeanColourEmbedder:
    def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for c in crops:
            v = c.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1.0
            out.append(v / np.linalg.norm(v))
        return out


def test_augmented_views_are_varied_rgb_and_seeded() -> None:
    sprite = placeholder_sprites()["demo_blue_can"]
    a = [augment_view(sprite, np.random.default_rng(i)) for i in range(25)]
    assert all(v.ndim == 3 and v.shape[2] == 3 and max(v.shape[:2]) <= 225 for v in a)
    assert len({v.shape for v in a}) > 5
    assert np.array_equal(augment_view(sprite, np.random.default_rng(3)), a[3])


def test_preprocess_pads_to_square_without_distortion() -> None:
    crop = np.full((100, 50, 3), 255, np.uint8)
    out = DinoEmbedder.preprocess(crop, 100)
    assert out.shape == (100, 100, 3)
    assert out[50, 50].tolist() == [255, 255, 255] and out[50, 5].tolist() == [124, 124, 124]


def test_load_packshots_and_build_gallery(tmp_path: Path) -> None:
    for name, colour in (("red", (200, 20, 20)), ("blue", (20, 40, 200))):
        img = Image.new("RGB", (80, 80), (255, 255, 255))
        img.paste(Image.new("RGB", (30, 60), colour), (25, 10))
        img.save(tmp_path / f"{name}.png")
    (tmp_path / "notes.txt").write_text("ignored")
    packs = load_packshots(tmp_path)
    assert sorted(packs) == ["blue", "red"] and packs["red"].size == (30, 60)
    index = build_gallery(packs, MeanColourEmbedder(), views=4, threshold=0.0)  # type: ignore[arg-type]
    assert len(index) == 2 * (1 + 4 + 1) and index.skus == ["blue", "red"]
    red = MeanColourEmbedder().embed_batch([np.full((8, 8, 3), (200, 20, 20), np.uint8)])[0]
    assert index.query(red).best_sku == "red"
