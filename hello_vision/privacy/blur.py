"""Face blurring for every stored frame.

Two layers: (1) a face detector (YuNet via OpenCV, model file supplied separately) and
(2) optional fixed regions that are ALWAYS blurred (e.g. the area in front of the door where
a customer's face can appear), which does not rely on detection at all.

LIMITS: no face detector is perfect, and this one has NOT been checked on real fridge footage
(small, rotated, masked or partly hidden faces can be missed). Treat the fixed regions as the
dependable layer and validate the detector on consented test footage before storing real video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from hello_vision.capture.types import Image

Box = tuple[int, int, int, int]  # x, y, w, h in pixels


class FaceDetector(Protocol):
    """Finds faces in an RGB frame."""

    def detect(self, image: Image) -> list[Box]:
        """Return ``(x, y, w, h)`` boxes."""
        ...


class YuNetFaceDetector:
    """YuNet face detector (MIT, ``cv2.FaceDetectorYN``), loaded from an ONNX file.

    The model file is NOT in the repository: fetch it with ``tools/fetch_face_model.py``.

    Args:
        model_path: Path of ``face_detection_yunet_*.onnx``.
        score_threshold: Minimum face score; keep it low, a missed face is worse than a blur.
    """

    def __init__(self, model_path: Path, score_threshold: float = 0.5) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"face model not found: {model_path}")
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold)

    def detect(self, image: Image) -> list[Box]:
        """Return ``(x, y, w, h)`` boxes of the faces found in the RGB frame."""
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def _mostly_inside(
    face: Box, regions: Sequence[tuple[float, float, float, float]], share: float = 0.6
) -> bool:
    x, y, w, h = face
    for x1, y1, x2, y2 in regions:
        ix = max(0.0, min(x + w, x2) - max(x, x1))
        iy = max(0.0, min(y + h, y2) - max(y, y1))
        if w * h > 0 and ix * iy / (w * h) >= share:
            return True
    return False


@dataclass
class FaceBlurrer:
    """Blurs detected faces and fixed regions.

    Args:
        detector: Face detector, or ``None`` to rely on fixed regions only.
        fixed_regions: Normalised ``(x1, y1, x2, y2)`` rectangles that are always blurred.
        margin: Fraction by which face boxes are enlarged before blurring.
    """

    detector: FaceDetector | None = None
    fixed_regions: Sequence[tuple[float, float, float, float]] = field(default_factory=tuple)
    margin: float = 0.4

    def blur(
        self, image: Image, protect: Sequence[tuple[float, float, float, float]] = ()
    ) -> Image:
        """Return a copy of ``image`` with faces and fixed regions irreversibly blurred.

        Only ever apply this to media that is saved or displayed, never before detection.

        Args:
            image: RGB frame.
            protect: ``(x1, y1, x2, y2)`` product boxes. A detected "face" lying mostly inside
                one is treated as a false alarm on packaging and left sharp. Fixed regions
                are always blurred.
        """
        out = image.copy()
        h, w = out.shape[:2]
        boxes: list[Box] = [
            (int(x1 * w), int(y1 * h), int((x2 - x1) * w), int((y2 - y1) * h))
            for x1, y1, x2, y2 in self.fixed_regions
        ]
        if self.detector is not None:
            boxes += [b for b in self.detector.detect(image) if not _mostly_inside(b, protect)]
        for x, y, bw, bh in boxes:
            dx, dy = int(bw * self.margin), int(bh * self.margin)
            x1, y1 = max(0, x - dx), max(0, y - dy)
            x2, y2 = min(w, x + bw + dx), min(h, y + bh + dy)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            # pixelate first (not reversible by deblurring), then smooth
            small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)))
            pix = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            out[y1:y2, x1:x2] = np.asarray(cv2.GaussianBlur(pix, (0, 0), 6), dtype=np.uint8)
        return out
