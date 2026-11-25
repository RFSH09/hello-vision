from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from hello_vision.catalog.gallery import Gallery
from hello_vision.catalog.models import Catalog, Sku
from hello_vision.catalog.packshots import placeholder_catalog
from hello_vision.catalog.sync import parse_catalog_payload


def test_roundtrip_yaml_and_json(tmp_path: Path) -> None:
    cat = placeholder_catalog()
    for name in ("c.yaml", "c.json"):
        cat.dump(tmp_path / name)
        assert Catalog.load(tmp_path / name) == cat


def test_duplicate_and_bad_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        Catalog(skus=[Sku(sku="a", name="A"), Sku(sku="a", name="B")])
    with pytest.raises(ValidationError):
        Sku(sku="../evil", name="x")


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        placeholder_catalog().get("nope")


def test_backend_payload() -> None:
    payload = {
        "skus": [{"id": "a", "name": "A"}, {"id": "b", "name": "B", "group": "g"}],
        "planograms": {"f1": {"shelf1": ["a", "b"]}},
    }
    cat, plan = parse_catalog_payload(payload)
    assert cat.ids() == ["a", "b"] and plan["f1"]["shelf1"] == ["a", "b"]
    payload["planograms"] = {"f1": {"shelf1": ["zzz"]}}
    with pytest.raises(ValueError, match="unknown SKUs"):
        parse_catalog_payload(payload)


def test_gallery(tmp_path: Path) -> None:
    g = Gallery(tmp_path)
    cat = placeholder_catalog()
    img = Image.new("RGBA", (4, 4), (1, 2, 3, 255))
    g.add_image("demo_cola_bottle", img, "a")
    g.add_image("demo_cola_bottle", img, "b")
    assert len(g.list_images("demo_cola_bottle")) == 2
    assert len(g.load_rgba("demo_cola_bottle")) == 2
    assert g.counts(cat)["demo_cola_bottle"] == 2 and g.counts(cat)["demo_gold_bar"] == 0
    with pytest.raises(ValueError):
        g.add_image("demo_cola_bottle", img, "../x")
