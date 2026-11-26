"""SKU gallery index: nearest-neighbour matching with an "unknown" outcome."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from hello_vision.capture.types import Image
from hello_vision.identify.embedders import Embedder, Vector


@dataclass(frozen=True)
class Match:
    """Identification result for one crop.

    Attributes:
        sku: Best SKU, or ``None`` when below the similarity threshold (unknown product).
        score: Cosine similarity to the best gallery vector.
        margin: Gap between the best SKU and the best *different* SKU (ambiguity signal).
        best_sku: Best SKU even when unknown (useful for reviewers).
    """

    sku: str | None
    score: float
    margin: float
    best_sku: str | None


class GalleryIndex:
    """Plain-numpy cosine index. Add reference vectors per SKU, then :meth:`query`.

    Args:
        threshold: Minimum cosine similarity to accept a match.
        min_margin: Minimum gap to the runner-up SKU; smaller gaps count as unknown because
            look-alike SKUs would otherwise be guessed.
    """

    def __init__(self, threshold: float = 0.85, min_margin: float = 0.0) -> None:
        self.threshold = threshold
        self.min_margin = min_margin
        self._vecs: list[Vector] = []
        self._skus: list[str] = []
        self._matrix: NDArray[np.float32] | None = None

    def add(self, sku: str, vector: Vector) -> None:
        """Add one reference vector for ``sku``."""
        self._vecs.append(vector)
        self._skus.append(sku)
        self._matrix = None

    def __len__(self) -> int:
        """Number of reference vectors."""
        return len(self._vecs)

    def export(self) -> tuple[list[str], list[Vector]]:
        """Return the SKU id of every reference vector and the vectors (for caching)."""
        return list(self._skus), list(self._vecs)

    @property
    def skus(self) -> list[str]:
        """Distinct SKU ids in the index, sorted."""
        return sorted(set(self._skus))

    def query(self, vector: Vector) -> Match:
        """Find the closest SKU for ``vector``.

        Raises:
            ValueError: If the index is empty.
        """
        if not self._vecs:
            raise ValueError("empty gallery index")
        if self._matrix is None:
            self._matrix = np.stack(self._vecs).astype(np.float32)
        sims = self._matrix @ vector
        per_sku: dict[str, float] = {}
        for sku, s in zip(self._skus, sims, strict=True):
            per_sku[sku] = max(per_sku.get(sku, -1.0), float(s))
        ranked = sorted(per_sku.items(), key=lambda kv: kv[1], reverse=True)
        best_sku, best = ranked[0]
        margin = best - ranked[1][1] if len(ranked) > 1 else best
        known = best >= self.threshold and margin >= self.min_margin
        return Match(best_sku if known else None, best, margin, best_sku)


def build_index(
    sprites: dict[str, list[PILImage.Image]],
    embedder: Embedder,
    threshold: float = 0.85,
    min_margin: float = 0.0,
    background: tuple[int, int, int] = (175, 175, 170),
) -> GalleryIndex:
    """Embed gallery sprites (RGBA cut-outs) over a neutral background and index them.

    Args:
        sprites: SKU id -> RGBA reference images (packshots, later real crops).
        embedder: Embedding model.
        threshold: See :class:`GalleryIndex`.
        min_margin: See :class:`GalleryIndex`.
        background: Colour composited behind each cut-out, matching shelf backgrounds.

    Returns:
        A populated index.
    """
    index = GalleryIndex(threshold, min_margin)
    for sku, images in sprites.items():
        for sp in images:
            canvas = PILImage.new("RGB", sp.size, background)
            canvas.paste(sp.convert("RGB"), (0, 0), sp.getchannel("A"))
            index.add(sku, embedder.embed(np.asarray(canvas, dtype=np.uint8)))
    return index


def crop(image: Image, box: tuple[float, float, float, float]) -> Image:
    """Crop ``image`` to an ``(x1, y1, x2, y2)`` box, clipped to the image (min 2x2 px)."""
    h, w = image.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, max(x1 + 2, int(box[2]))), min(h, max(y1 + 2, int(box[3])))
    return image[y1:y2, x1:x2]


def identify_all(
    image: Image,
    boxes: Iterable[tuple[float, float, float, float]],
    embedder: Embedder,
    index: GalleryIndex,
) -> list[Match]:
    """Identify every box in ``image``."""
    return [index.query(embedder.embed(crop(image, b))) for b in boxes]
