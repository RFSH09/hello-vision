"""Synthetic scene composition: backgrounds, sprites, hand occluders, lighting, COCO export."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels. ``label`` is ``product`` or ``hand``; ``sku`` for products."""

    label: str
    sku: str | None
    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> list[float]:
        """Return ``[x1, y1, x2, y2]``."""
        return [self.x1, self.y1, self.x2, self.y2]


def make_background(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """Return a plain fridge-shelf-like backdrop: tinted gradient with shelf rails."""
    w, h = size
    tint = rng.uniform(0.85, 1.0, size=3)
    base = np.linspace(200, 150, h, dtype=np.float32)[:, None, None] * tint[None, None, :]
    arr = np.broadcast_to(base, (h, w, 3)).copy()
    for frac in (0.12, 0.74):  # rails above and below the product row
        y = int(h * frac)
        arr[y : y + max(3, h // 90)] *= 0.55
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def place_sprite(
    canvas: Image.Image,
    sprite: Image.Image,
    center: tuple[float, float],
    height: int,
    angle: float = 0.0,
    max_width: int | None = None,
) -> tuple[float, float, float, float] | None:
    """Paste ``sprite`` (RGBA) scaled to ``height`` px, rotated, centred at ``center``.

    If ``max_width`` is given the sprite is shrunk (aspect kept) to fit it.

    Returns:
        The visible bounding box ``(x1, y1, x2, y2)`` clipped to the canvas, or ``None`` if
        the sprite is entirely outside.
    """
    scale = height / sprite.height
    if max_width is not None:
        scale = min(scale, max_width / sprite.width)
    sp = sprite.resize(
        (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))),
        Image.Resampling.BILINEAR,
    )
    if angle:
        sp = sp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = round(center[0] - sp.width / 2), round(center[1] - sp.height / 2)
    alpha = sp.getchannel("A")
    canvas.paste(sp.convert("RGB"), (x, y), alpha)
    bb = alpha.point(lambda v: 255 if v > 8 else 0).getbbox()
    if bb is None:
        return None
    x1, y1 = max(0.0, x + bb[0]), max(0.0, y + bb[1])
    x2, y2 = min(float(canvas.width), x + bb[2]), min(float(canvas.height), y + bb[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def draw_hand(
    canvas: Image.Image,
    center: tuple[float, float],
    rng: np.random.Generator,
    scale: float = 1.0,
    skin: tuple[int, int, int] = (224, 172, 140),
) -> tuple[float, float, float, float] | None:
    """Draw a crude synthetic hand (palm, fingers, wrist) as an occluder.

    Returns:
        Visible bounding box clipped to the canvas, or ``None`` if fully outside.
    """
    s = scale
    layer = Image.new("RGBA", (int(260 * s), int(380 * s)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = (*skin, 255)
    d.rectangle((90 * s, 190 * s, 170 * s, 380 * s), fill=col)  # wrist/forearm
    d.ellipse((60 * s, 90 * s, 200 * s, 250 * s), fill=col)  # palm
    for i in range(4):  # fingers with a little random curl
        fx = (75 + i * 35) * s
        top = (20 + abs(i - 1.5) * 18 + float(rng.uniform(0, 14))) * s
        d.rounded_rectangle((fx - 14 * s, top, fx + 14 * s, 140 * s), 14 * s, fill=col)
    d.ellipse((20 * s, 130 * s, 80 * s, 200 * s), fill=col)  # thumb
    layer = layer.filter(ImageFilter.GaussianBlur(1.0))
    x, y = round(center[0] - layer.width / 2), round(center[1] - layer.height / 2)
    alpha = layer.getchannel("A")
    canvas.paste(layer.convert("RGB"), (x, y), alpha)
    bb = alpha.point(lambda v: 255 if v > 8 else 0).getbbox()
    if bb is None:
        return None
    x1, y1 = max(0.0, x + bb[0]), max(0.0, y + bb[1])
    x2, y2 = min(float(canvas.width), x + bb[2]), min(float(canvas.height), y + bb[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def apply_lighting(
    img: Image.Image,
    rng: np.random.Generator,
    gain: float = 1.0,
    gamma: float = 1.0,
    noise_sigma: float = 2.0,
) -> Image.Image:
    """Apply brightness gain, gamma and sensor noise (low light: gain<1, gamma>1)."""
    arr: NDArray[np.float32] = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr * gain, 0, 1), gamma) * 255.0
    if noise_sigma > 0:
        arr = arr + rng.normal(0, noise_sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def compose_scene(
    sprites: dict[str, Image.Image],
    rng: np.random.Generator,
    size: tuple[int, int] = (960, 540),
    with_hand: bool | None = None,
) -> tuple[Image.Image, list[Box]]:
    """Random shelf scene for detector pre-training: products in a row, optional hand.

    Args:
        sprites: SKU id -> RGBA cut-out.
        rng: Random generator (seeded by the caller for reproducibility).
        size: Output ``(width, height)``.
        with_hand: Force or forbid a hand; ``None`` picks at random (50%).

    Returns:
        The image and its product/hand boxes.
    """
    w, h = size
    img = make_background(size, rng)
    skus = sorted(sprites)
    n = int(rng.integers(3, 8))
    boxes: list[Box] = []
    slot_w = w / (n + 1)
    for i in range(n):
        sku = skus[int(rng.integers(len(skus)))]
        height = int(h * rng.uniform(0.28, 0.40))
        cx = slot_w * (i + 1) + rng.uniform(-0.2, 0.2) * slot_w
        cy = h * 0.45 + rng.uniform(-0.03, 0.03) * h
        bb = place_sprite(
            img, sprites[sku], (cx, cy), height, float(rng.uniform(-6, 6)), int(slot_w * 0.9)
        )
        if bb:
            boxes.append(Box("product", sku, *bb))
    if with_hand is None:
        with_hand = bool(rng.random() < 0.5)
    if with_hand:
        hb = draw_hand(img, (rng.uniform(0.1, 0.9) * w, h * rng.uniform(0.5, 0.9)), rng, 1.0)
        if hb:
            boxes.append(Box("hand", None, *hb))
    gain = float(rng.uniform(0.55, 1.15))
    img = apply_lighting(img, rng, gain, float(rng.uniform(0.9, 1.4)), float(rng.uniform(0, 5)))
    return img, boxes


def write_coco(
    path: Path,
    items: Sequence[tuple[str, int, int, Sequence[Box]]],
) -> None:
    """Write a class-agnostic COCO file (categories ``product`` and ``hand``).

    Args:
        path: Output JSON path.
        items: ``(file_name, width, height, boxes)`` per image. The SKU is kept in each
            annotation's ``sku`` field for identification training.
    """
    cats = [{"id": 1, "name": "product"}, {"id": 2, "name": "hand"}]
    images: list[dict[str, object]] = []
    anns: list[dict[str, object]] = []
    for img_id, (name, w, h, boxes) in enumerate(items, start=1):
        images.append({"id": img_id, "file_name": name, "width": w, "height": h})
        for b in boxes:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": 1 if b.label == "product" else 2,
                    "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],
                    "area": (b.x2 - b.x1) * (b.y2 - b.y1),
                    "iscrowd": 0,
                    "sku": b.sku,
                }
            )
    path.write_text(
        json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8"
    )


def generate_dataset(
    out_dir: Path, sprites: dict[str, Image.Image], count: int, seed: int = 0
) -> Path:
    """Render ``count`` random scenes to ``out_dir`` (JPEG + ``annotations.json``).

    Returns:
        Path of the COCO annotation file. ``out_dir`` should be outside the git repo.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, int, int, list[Box]]] = []
    for i in range(count):
        img, boxes = compose_scene(sprites, rng)
        name = f"synth_{i:05d}.jpg"
        img.save(out_dir / name, quality=92)
        items.append((name, img.width, img.height, boxes))
    ann = out_dir / "annotations.json"
    write_coco(ann, items)
    return ann
