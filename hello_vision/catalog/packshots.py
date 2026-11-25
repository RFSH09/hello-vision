"""Packshot import: background removal, augmentation, and synthetic placeholder packshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

from hello_vision.catalog.gallery import Gallery
from hello_vision.catalog.models import Catalog, Sku

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def remove_background(img: Image.Image, tolerance: float = 12.0) -> Image.Image:
    """Cut a product out of a plain-background packshot.

    The background colour is the median of the image border. Background is the set of
    pixels within ``tolerance`` (RGB distance) of it that are connected to the border, so
    enclosed near-white areas such as labels stay part of the product. Only the largest
    foreground component is kept; the result is cropped to the product.

    Args:
        img: Packshot on a roughly uniform background.
        tolerance: Foreground distance threshold.

    Returns:
        RGBA image. If nothing is detected the input is returned fully opaque.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=2)
    # Background = near-background-coloured pixels connected to the image border. Anything
    # enclosed (e.g. a white label) stays foreground even if it matches the background.
    near_bg = (dist <= tolerance).astype(np.uint8)
    _, labels = cv2.connectedComponents(near_bg, connectivity=4)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    is_bg = np.isin(labels, edge_labels[edge_labels > 0]) & (near_bg > 0)
    mask = ((~is_bg) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img.convert("RGBA")
    biggest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [biggest], -1, 255, thickness=-1)
    alpha = cv2.GaussianBlur(filled, (3, 3), 0)
    rgba = Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")
    bbox = Image.fromarray(filled).getbbox()
    return rgba.crop(bbox) if bbox else rgba


def augment_packshot(rgba: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Return a randomly perturbed copy (rotation, scale, brightness, blur) of an RGBA sprite."""
    img = rgba.convert("RGBA")
    scale = float(rng.uniform(0.9, 1.1))
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    img = img.rotate(float(rng.uniform(-8, 8)), expand=True, resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * float(rng.uniform(0.8, 1.2)), 0, 255)
    out = Image.fromarray(arr.astype(np.uint8), "RGBA")
    radius = float(rng.uniform(0, 1.2))
    return out.filter(ImageFilter.GaussianBlur(radius)) if radius > 0.05 else out


@dataclass
class ImportReport:
    """Result of :func:`import_packshots`."""

    imported: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # catalog SKUs without a file
    unmatched: list[str] = field(default_factory=list)  # files that match no SKU


def import_packshots(src_dir: Path, catalog: Catalog, gallery: Gallery) -> ImportReport:
    """Import packshots named ``<sku>.<ext>`` into the gallery with the background removed.

    Args:
        src_dir: Folder with the packshot files (kept outside the repo).
        catalog: Catalog the file names are matched against.
        gallery: Destination gallery.

    Returns:
        Which SKUs were imported, which are missing a file, and which files matched nothing.
    """
    report = ImportReport()
    files = {p.stem: p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES}
    for sku in catalog.ids():
        path = files.pop(sku, None)
        if path is None:
            report.missing.append(sku)
            continue
        with Image.open(path) as im:
            gallery.add_image(sku, remove_background(im), "packshot")
        report.imported.append(sku)
    report.unmatched = sorted(files)
    return report


# ---------------------------------------------------------------- synthetic placeholders

_BODY = tuple[int, int, int]


def make_placeholder_packshot(sku: Sku, shape: str, color: _BODY, size: int = 800) -> Image.Image:
    """Draw a generic synthetic packshot (white background) for demos and tests.

    No real brand artwork is used. ``shape`` is ``bottle``, ``can`` or ``bar``.
    """
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    s = size
    dark = tuple(max(0, c - 70) for c in color)
    if shape == "bottle":
        d.rounded_rectangle((s * 0.34, s * 0.32, s * 0.66, s * 0.92), s * 0.06, fill=color)
        d.rounded_rectangle((s * 0.43, s * 0.10, s * 0.57, s * 0.34), s * 0.03, fill=color)
        d.rectangle((s * 0.43, s * 0.06, s * 0.57, s * 0.11), fill=dark)
        label = (s * 0.34, s * 0.5, s * 0.66, s * 0.72)
    elif shape == "can":
        d.rounded_rectangle((s * 0.32, s * 0.14, s * 0.68, s * 0.88), s * 0.04, fill=color)
        d.rectangle((s * 0.32, s * 0.14, s * 0.68, s * 0.19), fill=dark)
        label = (s * 0.32, s * 0.4, s * 0.68, s * 0.62)
    elif shape == "bar":
        d.rounded_rectangle((s * 0.12, s * 0.38, s * 0.88, s * 0.62), s * 0.03, fill=color)
        d.polygon([(s * 0.12, s * 0.38), (s * 0.06, s * 0.5), (s * 0.12, s * 0.62)], fill=dark)
        d.polygon([(s * 0.88, s * 0.38), (s * 0.94, s * 0.5), (s * 0.88, s * 0.62)], fill=dark)
        label = (s * 0.3, s * 0.42, s * 0.7, s * 0.58)
    else:
        raise ValueError(f"unknown shape {shape!r}")
    d.rectangle(label, fill=(236, 236, 236))
    d.text((label[0] + 8, label[1] + 8), sku.sku[:10], fill=dark)
    return img


# (sku, name, group, shape, colour)
_PLACEHOLDERS: list[tuple[str, str, str, str, _BODY]] = [
    ("demo_cola_bottle", "Demo Cola bottle", "bottles", "bottle", (190, 30, 30)),
    ("demo_orange_bottle", "Demo Orange bottle", "bottles", "bottle", (235, 130, 20)),
    ("demo_tea_bottle", "Demo Tea bottle", "bottles", "bottle", (170, 190, 40)),
    ("demo_black_can", "Demo Black can", "cans", "can", (40, 40, 45)),
    ("demo_blue_can", "Demo Blue can", "cans", "can", (30, 90, 200)),
    ("demo_maroon_can", "Demo Maroon can", "cans", "can", (110, 20, 50)),
    ("demo_brown_bar", "Demo Brown bar", "bars", "bar", (110, 60, 30)),
    ("demo_gold_bar", "Demo Gold bar", "bars", "bar", (210, 170, 40)),
    ("demo_navy_bar", "Demo Navy bar", "bars", "bar", (30, 50, 120)),
]


def placeholder_catalog() -> Catalog:
    """Return the synthetic demo catalog (generic names, look-alike groups)."""
    return Catalog(skus=[Sku(sku=i, name=n, group=g) for i, n, g, _, _ in _PLACEHOLDERS])


def placeholder_sprites() -> dict[str, Image.Image]:
    """Build RGBA cut-outs for the demo catalog via the real import pipeline."""
    catalog = placeholder_catalog()
    out: dict[str, Image.Image] = {}
    for sku_id, _, _, shape, color in _PLACEHOLDERS:
        pack = make_placeholder_packshot(catalog.get(sku_id), shape, color)
        out[sku_id] = remove_background(pack)
    return out


def sprite_alpha_bbox(rgba: Image.Image) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-transparent pixels, or ``None`` if empty."""
    alpha: NDArray[np.uint8] = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    return Image.fromarray((alpha > 8).astype(np.uint8) * 255).getbbox()
