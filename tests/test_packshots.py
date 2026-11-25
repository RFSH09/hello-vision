from pathlib import Path

import numpy as np
from PIL import Image

from hello_vision.catalog.gallery import Gallery
from hello_vision.catalog.models import Catalog, Sku
from hello_vision.catalog.packshots import (
    augment_packshot,
    import_packshots,
    make_placeholder_packshot,
    placeholder_catalog,
    placeholder_sprites,
    remove_background,
)


def test_background_removed_but_label_kept() -> None:
    cat = placeholder_catalog()
    pack = make_placeholder_packshot(cat.get("demo_cola_bottle"), "bottle", (190, 30, 30))
    sprite = remove_background(pack)
    a = np.asarray(sprite.getchannel("A"))
    assert a[0, 0] == 0 and a[-1, -1] == 0  # corners are background
    assert a[a.shape[0] // 2, a.shape[1] // 2] == 255  # centre (label) kept
    assert sprite.height > sprite.width  # bottle aspect preserved (label did not split it)


def test_blank_image_returned_opaque() -> None:
    out = remove_background(Image.new("RGB", (20, 20), (255, 255, 255)))
    assert out.mode == "RGBA" and out.size == (20, 20)


def test_augment_is_seeded() -> None:
    sp = placeholder_sprites()["demo_blue_can"]
    a = augment_packshot(sp, np.random.default_rng(1))
    b = augment_packshot(sp, np.random.default_rng(1))
    assert a.tobytes() == b.tobytes() and a.mode == "RGBA"


def test_import_report(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    cat = Catalog(skus=[Sku(sku="a", name="A"), Sku(sku="b", name="B")])
    make_placeholder_packshot(cat.get("a"), "can", (10, 80, 160)).save(src / "a.png")
    Image.new("RGB", (8, 8)).save(src / "stray.jpg")
    (src / "notes.txt").write_text("ignored")
    gallery = Gallery(tmp_path / "gal")
    report = import_packshots(src, cat, gallery)
    assert report.imported == ["a"] and report.missing == ["b"] and report.unmatched == ["stray"]
    assert gallery.list_images("a")
