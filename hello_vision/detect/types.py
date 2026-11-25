"""Detector interface and result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hello_vision.capture.types import Image


@dataclass(frozen=True)
class Detection:
    """One detected object. ``label`` is ``product`` or ``hand`` (class-agnostic products)."""

    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """Box centre ``(cx, cy)``."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two detections (0 if they do not overlap)."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class Detector(Protocol):
    """Finds products and hands in one frame."""

    def start_session(self, first_image: Image) -> None:
        """Prepare for a new session (e.g. calibrate a background); may be a no-op."""
        ...

    def detect(self, image: Image) -> list[Detection]:
        """Return detections for ``image`` (RGB ``uint8``)."""
        ...
