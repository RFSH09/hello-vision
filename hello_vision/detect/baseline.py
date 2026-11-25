"""Classical baseline detector for SYNTHETIC data only.

This is a stand-in so the whole pipeline can be exercised end to end before a learned
detector (RT-DETR / D-FINE / ...) is trained. It uses a per-row background colour taken from
the image margins of the first (hand-free) frame, and a skin-colour test to separate hands
from products. It is NOT expected to work on real footage.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from hello_vision.capture.types import Image
from hello_vision.detect.types import Detection


class BaselineDetector:
    """Background-difference + skin-colour detector.

    Args:
        diff_threshold: RGB distance from the row background to count as foreground.
        min_area: Minimum component area in pixels.
        margin: Width in pixels of the left/right strips assumed to be empty background.
    """

    def __init__(self, diff_threshold: float = 30.0, min_area: int = 700, margin: int = 24) -> None:
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.margin = margin
        self._bg: NDArray[np.float32] | None = None

    def _row_background(self, image: Image) -> NDArray[np.float32]:
        m = self.margin
        strips = np.concatenate([image[:, :m], image[:, -m:]], axis=1).astype(np.float32)
        return np.median(strips, axis=1, keepdims=True)  # (H, 1, 3)

    def start_session(self, first_image: Image) -> None:
        """Calibrate the background from the first frame of the session."""
        self._bg = self._row_background(first_image)

    def detect(self, image: Image) -> list[Detection]:
        """Detect hands (skin-coloured blobs) and products (other foreground blobs)."""
        if self._bg is None:
            self.start_session(image)
        assert self._bg is not None  # noqa: S101
        diff = np.linalg.norm(image.astype(np.float32) - self._bg, axis=2)
        fg = diff > self.diff_threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        skin = (h >= 5) & (h <= 25) & (s >= 50) & (s <= 150) & (v >= 50)
        hand_mask = (fg & skin).astype(np.uint8) * 255
        prod_mask = (fg & ~skin).astype(np.uint8) * 255
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        prod_mask = cv2.morphologyEx(prod_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # bridge fingers that cut a product in two
        prod_mask = cv2.morphologyEx(prod_mask, cv2.MORPH_CLOSE, np.ones((3, 31), np.uint8))
        out: list[Detection] = []
        for label, mask in (("hand", hand_mask), ("product", prod_mask)):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, w, hh, area = (int(t) for t in stats[i])
                if area >= self.min_area:
                    out.append(
                        Detection(label, 1.0, float(x), float(y), float(x + w), float(y + hh))
                    )
        return out
