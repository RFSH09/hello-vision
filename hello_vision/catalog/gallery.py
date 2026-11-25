"""On-disk SKU gallery: ``<root>/<sku>/<name>.png`` (RGBA). Root is configuration, not code."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hello_vision.catalog.models import Catalog


class Gallery:
    """Reference images per SKU (packshots now, real crops later).

    Args:
        root: Directory holding one sub-folder per SKU. Must live outside the git repo for
            real data.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def sku_dir(self, sku: str) -> Path:
        """Return the folder for ``sku`` (not created)."""
        return self.root / sku

    def add_image(self, sku: str, image: Image.Image, name: str) -> Path:
        """Store ``image`` as ``<sku>/<name>.png`` and return its path."""
        if not name or any(c in name for c in "/\\"):
            raise ValueError(f"invalid gallery image name: {name!r}")
        folder = self.sku_dir(sku)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        image.convert("RGBA").save(path)
        return path

    def list_images(self, sku: str) -> list[Path]:
        """Return stored image paths for ``sku``, sorted by name."""
        folder = self.sku_dir(sku)
        return sorted(folder.glob("*.png")) if folder.is_dir() else []

    def load_rgba(self, sku: str) -> list[Image.Image]:
        """Load all images of ``sku`` as RGBA."""
        out: list[Image.Image] = []
        for p in self.list_images(sku):
            with Image.open(p) as im:
                out.append(im.convert("RGBA"))
        return out

    def counts(self, catalog: Catalog) -> dict[str, int]:
        """Return the number of gallery images per catalog SKU."""
        return {s: len(self.list_images(s)) for s in catalog.ids()}
