"""Crop embedders: a colour-layout baseline (synthetic only) and an ONNX wrapper."""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from hello_vision.capture.types import Image

Vector = NDArray[np.float32]


class Embedder(Protocol):
    """Maps an RGB crop to an L2-normalised vector."""

    def embed(self, crop: Image) -> Vector:
        """Return the embedding of ``crop``."""
        ...


def _normalise(v: Vector) -> Vector:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class ColorLayoutEmbedder:
    """Baseline embedding: hue/saturation histogram + coarse colour layout + aspect ratio.

    Works for the synthetic demo where SKUs differ in colour. It cannot separate real
    look-alike SKUs; the learned embedder (DINOv2 fine-tuned with metric learning) is meant to.
    The crop is brightness-normalised so low-light sessions stay comparable.
    """

    def embed(self, crop: Image) -> Vector:
        """Embed ``crop`` (``H x W x 3`` RGB ``uint8``)."""
        small = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        scale = 190.0 / max(1.0, float(np.percentile(small.max(axis=2), 95)))
        small = np.clip(small * scale, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 0, 256]).ravel()
        hist = hist / max(1.0, float(hist.sum()))
        bands = small.reshape(4, 16, 32, 3).mean(axis=(1, 2)).ravel() / 255.0  # 4 vertical bands
        aspect = np.array([crop.shape[1] / max(1, crop.shape[0])], dtype=np.float32)
        return _normalise(np.concatenate([hist * 2.0, bands, aspect]).astype(np.float32))


class OrtLikeSession(Protocol):
    """The subset of ``onnxruntime.InferenceSession`` that is used."""

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:  # noqa: ANN401
        """Run inference."""
        ...


class OnnxEmbedder:
    """Embedding model exported to ONNX (e.g. DINOv2-small), run through an injectable session.

    UNTESTED against a real model (none trained yet). Input: ``1x3xSxS`` float32, ImageNet
    normalisation; output: first output, flattened and L2-normalised.

    Args:
        session: ONNX runtime session (or a compatible fake).
        input_size: Square input size.
        input_name: Name of the model input.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self, session: OrtLikeSession, input_size: int = 224, input_name: str = "input"
    ) -> None:
        self.session = session
        self.input_size = input_size
        self.input_name = input_name

    def embed(self, crop: Image) -> Vector:
        """Embed ``crop`` with the ONNX model."""
        x = cv2.resize(crop, (self.input_size, self.input_size)).astype(np.float32) / 255.0
        x = ((x - self._MEAN) / self._STD).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x.astype(np.float32)})[0]
        return _normalise(np.asarray(out, dtype=np.float32).reshape(-1))
