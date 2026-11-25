"""ONNX wrapper for a learned detector (RT-DETR style outputs).

UNTESTED against a real model: no detector has been trained yet. The wrapper expects a model
with input ``images`` (1x3xSxS float, RGB, 0..1) and ``orig_target_sizes`` (1x2 int64, w/h),
and outputs ``labels``, ``boxes`` (xyxy in original pixels) and ``scores``. The session is
injectable so the pre/post-processing is unit-tested with a fake.
"""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np

from hello_vision.capture.types import Image
from hello_vision.detect.types import Detection

LABELS = {0: "product", 1: "hand"}


class OrtLikeSession(Protocol):
    """The subset of ``onnxruntime.InferenceSession`` that is used."""

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:  # noqa: ANN401
        """Run inference."""
        ...


def load_session(path: str) -> OrtLikeSession:
    """Open an ONNX model with onnxruntime (optional dependency: ``uv sync --extra onnx``)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("onnxruntime missing: install with `uv sync --extra onnx`") from exc
    session: OrtLikeSession = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session


class OnnxDetector:
    """Detector backed by an ONNX session.

    Args:
        session: ONNX runtime session (or a compatible fake).
        input_size: Square network input size.
        score_threshold: Minimum score to keep a detection.
    """

    def __init__(
        self, session: OrtLikeSession, input_size: int = 640, score_threshold: float = 0.4
    ) -> None:
        self.session = session
        self.input_size = input_size
        self.score_threshold = score_threshold

    def start_session(self, first_image: Image) -> None:
        """No-op: learned detectors need no per-session calibration."""

    def preprocess(self, image: Image) -> np.ndarray[Any, Any]:
        """Resize to the network size and convert to ``1x3xSxS`` float32 in 0..1."""
        resized = cv2.resize(image, (self.input_size, self.input_size))
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        return chw[None]

    def detect(self, image: Image) -> list[Detection]:
        """Run the model and keep detections above the score threshold."""
        h, w = image.shape[:2]
        feed = {
            "images": self.preprocess(image),
            "orig_target_sizes": np.array([[w, h]], dtype=np.int64),
        }
        labels, boxes, scores = self.session.run(None, feed)
        out: list[Detection] = []
        for lab, box, sc in zip(labels[0], boxes[0], scores[0], strict=True):
            if float(sc) >= self.score_threshold:
                x1, y1, x2, y2 = (float(t) for t in box)
                out.append(Detection(LABELS.get(int(lab), "product"), float(sc), x1, y1, x2, y2))
        return out
