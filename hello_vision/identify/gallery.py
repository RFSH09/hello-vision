"""DINOv2 SKU gallery: augmented packshot views -> embeddings -> cosine matching.

``facebook/dinov2-small`` (Apache-2.0). Needs ``uv sync --extra vision``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageEnhance, ImageFilter

from hello_vision.capture.types import Image
from hello_vision.catalog.packshots import IMAGE_SUFFIXES, remove_background
from hello_vision.identify.embedders import Vector
from hello_vision.identify.index import GalleryIndex

DINO_ID = "facebook/dinov2-small"


def augment_view(sprite: PILImage.Image, rng: np.random.Generator) -> Image:
    """Make one realistic-ish view of an RGBA packshot cut-out.

    Random rotation, perspective, scale, crop, background, brightness/contrast, partial
    occlusion, blur and down-sampling. Returns an RGB array.
    """
    img = sprite.convert("RGBA")
    angle = float(rng.uniform(-35, 35))
    if rng.random() < 0.2:
        angle += float(rng.choice([-90, 90, 180]))  # products also lie sideways
    img = img.rotate(angle, expand=True, resample=PILImage.Resampling.BICUBIC)
    # perspective: jitter the four corners
    w, h = img.size
    j = 0.14
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + (rng.uniform(-j, j, (4, 2)) * [w, h]).astype(np.float32)
    dst -= dst.min(axis=0)
    out_w, out_h = (int(v) + 1 for v in dst.max(axis=0))
    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(np.asarray(img), matrix, (out_w, out_h))
    # composite on a random, slightly textured background
    pad = int(0.08 * max(out_w, out_h))
    base = rng.uniform(40, 230, 3)
    noise = rng.normal(0, 8, (out_h + 2 * pad, out_w + 2 * pad, 3))
    canvas = PILImage.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    fg = PILImage.fromarray(warped)
    canvas.paste(fg.convert("RGB"), (pad, pad), fg.getchannel("A"))
    # partial occlusion: a skin- or random-coloured blob from one side
    if rng.random() < 0.35:
        d = ImageDraw.Draw(canvas)
        cw, ch = canvas.size
        colour = (
            (205, 150, 120) if rng.random() < 0.7 else tuple(int(v) for v in rng.uniform(0, 255, 3))
        )
        ow, oh = rng.uniform(0.25, 0.6) * cw, rng.uniform(0.2, 0.45) * ch
        ox = float(rng.choice([-ow * 0.3, cw - ow * 0.7]))
        oy = float(rng.uniform(0, ch - oh))
        d.ellipse((ox, oy, ox + ow, oy + oh), fill=colour)
    # random crop (keeps at least ~75 % of each side)
    cw, ch = canvas.size
    left, top, right, bottom = rng.uniform(0, 0.15, 4)
    canvas = canvas.crop(
        (int(left * cw), int(top * ch), int(cw - right * cw), int(ch - bottom * ch))
    )
    canvas = ImageEnhance.Brightness(canvas).enhance(float(rng.uniform(0.5, 1.35)))
    canvas = ImageEnhance.Contrast(canvas).enhance(float(rng.uniform(0.65, 1.3)))
    canvas = ImageEnhance.Color(canvas).enhance(float(rng.uniform(0.7, 1.2)))
    # camera resolution: products are small in the frame
    long_side = int(rng.uniform(60, 220))
    scale = long_side / max(canvas.size)
    canvas = canvas.resize((max(8, int(canvas.width * scale)), max(8, int(canvas.height * scale))))
    if rng.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.6))))
    return np.asarray(canvas, dtype=np.uint8)


class DinoEmbedder:
    """DINOv2-small image embedder (class token + mean patch token, L2-normalised).

    Args:
        device: ``cuda``, ``cpu`` or ``None`` for automatic.
        cache_dir: Where model weights are stored.
        batch_size: Images per forward pass.
    """

    def __init__(
        self,
        device: str | None = None,
        cache_dir: Path = Path("models"),
        batch_size: int = 64,
        model_id: str = DINO_ID,
    ) -> None:
        import torch
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model: Any = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    @staticmethod
    def preprocess(crop: Image, size: int = 224) -> Image:
        """Pad to a square (keeping the aspect ratio, which separates cans/bottles) and resize."""
        h, w = crop.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        square = np.full((side, side, 3), 124, dtype=np.uint8)
        square[top : top + h, left : left + w] = crop
        return np.asarray(
            cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )

    def embed_batch(self, crops: Sequence[Image]) -> list[Vector]:
        """Embed many RGB crops."""
        import torch

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        out: list[Vector] = []
        for i in range(0, len(crops), self.batch_size):
            batch = np.stack([self.preprocess(c) for c in crops[i : i + self.batch_size]])
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            with torch.inference_mode():
                tokens = self.model(pixel_values=(x - mean) / std).last_hidden_state
            cls = torch.nn.functional.normalize(tokens[:, 0], dim=1)
            patches = torch.nn.functional.normalize(tokens[:, 1:].mean(dim=1), dim=1)
            vec = torch.nn.functional.normalize(torch.cat([cls, patches], dim=1), dim=1)
            out.extend(vec.cpu().numpy().astype(np.float32))
        return out

    def embed(self, crop: Image) -> Vector:
        """Embed one crop."""
        return self.embed_batch([crop])[0]


def load_packshots(folder: Path) -> dict[str, PILImage.Image]:
    """Load ``<sku>.<ext>`` product photos and cut them out of their white background."""
    out: dict[str, PILImage.Image] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            with PILImage.open(p) as im:
                rgba = im.convert("RGBA")
                white = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
                white.alpha_composite(rgba)  # transparent photos get a white background
                out[p.stem] = remove_background(white.convert("RGB"))
    return out


def build_gallery(
    packshots: dict[str, PILImage.Image],
    embedder: DinoEmbedder,
    views: int = 50,
    threshold: float = 0.35,
    min_margin: float = 0.02,
    seed: int = 0,
) -> GalleryIndex:
    """Index each SKU with the clean cut-out, ``views`` augmented views and their mean."""
    rng = np.random.default_rng(seed)
    index = GalleryIndex(threshold, min_margin)
    for sku, sprite in packshots.items():
        clean = PILImage.new("RGB", sprite.size, (150, 150, 150))
        clean.paste(sprite.convert("RGB"), (0, 0), sprite.getchannel("A"))
        crops = [np.asarray(clean, dtype=np.uint8)]
        crops += [augment_view(sprite, rng) for _ in range(views)]
        vecs = embedder.embed_batch(crops)
        for v in vecs:
            index.add(sku, v)
        mean = np.mean(np.stack(vecs), axis=0)
        index.add(sku, (mean / np.linalg.norm(mean)).astype(np.float32))
    return index
