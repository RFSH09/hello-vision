"""Zero-shot product/hand detection with open-vocabulary models (no training needed).

Backends (both Apache-2.0, run through ``transformers``):

* ``owlv2``: ``google/owlv2-base-patch16-ensemble``
* ``gdino``: ``IDEA-Research/grounding-dino-tiny``

Needs the optional extra: ``uv sync --extra vision``. Weights are cached under ``models/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from hello_vision.capture.types import Image
from hello_vision.detect.types import Detection, iou

PRODUCT_QUERIES = (
    "plastic bottle",
    "drink can",
    "chocolate bar",
    "bag of chips",
    "snack pack",
    "milk carton",
    "sandwich",
)
HAND_QUERY = "hand"
# queried so they are NOT mistaken for products; heads are also blurred in all outputs
HEAD_QUERY = "human head"
PHONE_QUERY = "mobile phone"
# measured on our fridge footage: real heads scored 0.24-0.45 (OWLv2) and 0.59-0.71 (Grounding
# DINO); false hits on cans and hands reached 0.16 and 0.49
HEAD_THRESHOLDS = {"owlv2": 0.2, "gdino": 0.55}
HEAD_MAX_AREA = 0.15  # fraction of the frame; larger "heads" were arms or whole scenes


def label_of(name: str) -> str:
    """Map a (possibly partial) query phrase to ``head``, ``phone``, ``hand`` or ``product``."""
    n = name.strip().lower()
    if "head" in n or "human" in n:
        return "head"
    if "phone" in n or "mobile" in n:
        return "phone"
    return "hand" if n == HAND_QUERY else "product"


def suppress_distractors(dets: Sequence[Detection], overlap: float = 0.45) -> list[Detection]:
    """Drop product boxes that coincide with a head or phone, and the phone boxes themselves."""
    blockers = [d for d in dets if d.label in ("head", "phone")]
    out: list[Detection] = []
    for d in dets:
        if d.label == "phone":
            continue
        if d.label == "product" and any(
            iou(d, b) > overlap and b.score >= 0.7 * d.score for b in blockers
        ):
            continue
        out.append(d)
    return out


MODEL_IDS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "gdino": "IDEA-Research/grounding-dino-tiny",
}
DEFAULT_THRESHOLDS = {"owlv2": 0.15, "gdino": 0.25}


def nms(dets: Sequence[Detection], threshold: float = 0.5) -> list[Detection]:
    """Greedy non-maximum suppression within each label, highest score first.

    All product queries map to the label ``product``, so one object matched by several text
    queries ("bottle" and "snack pack") is kept only once.
    """
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.score, reverse=True):
        if all(k.label != d.label or iou(k, d) <= threshold for k in kept):
            kept.append(d)
    return kept


def drop_containers(dets: Sequence[Detection], coverage: float = 0.8) -> list[Detection]:
    """Drop product boxes that mostly just enclose two or more other product boxes (shelf rows)."""
    products = [d for d in dets if d.label == "product"]
    bad: set[int] = set()
    for big in products:
        inside = 0
        for small in products:
            if small is big or small.area >= big.area * 0.7:
                continue
            ix = max(0.0, min(big.x2, small.x2) - max(big.x1, small.x1))
            iy = max(0.0, min(big.y2, small.y2) - max(big.y1, small.y1))
            if small.area > 0 and ix * iy / small.area >= coverage:
                inside += 1
        if inside >= 2:
            bad.add(id(big))
    return [d for d in dets if id(d) not in bad]


def clean(
    dets: Sequence[Detection],
    width: int,
    height: int,
    max_area: float = 0.25,
    min_side: int = 12,
    nms_iou: float = 0.5,
) -> list[Detection]:
    """Remove implausible boxes (tiny, or products covering a large part of the frame) + NMS."""
    frame = float(width * height)
    ok = [
        d
        for d in dets
        if d.x2 - d.x1 >= min_side
        and d.y2 - d.y1 >= min_side
        and (d.label != "product" or d.area <= max_area * frame)
        and (d.label != "head" or d.area <= HEAD_MAX_AREA * frame)
    ]
    return drop_containers(suppress_distractors(nms(ok, nms_iou)))


class ZeroShotDetector:
    """Open-vocabulary detector that outputs only ``product`` and ``hand``.

    Args:
        backend: ``owlv2`` or ``gdino``.
        threshold: Score threshold (default depends on the backend).
        device: ``cuda``, ``cpu`` or ``None`` for automatic.
        cache_dir: Where model weights are stored.
        max_area: Largest product box as a fraction of the frame.
    """

    def __init__(
        self,
        backend: str = "owlv2",
        threshold: float | None = None,
        device: str | None = None,
        cache_dir: Path = Path("models"),
        max_area: float = 0.25,
    ) -> None:
        if backend not in MODEL_IDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {sorted(MODEL_IDS)}")
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.backend = backend
        self.threshold = DEFAULT_THRESHOLDS[backend] if threshold is None else threshold
        self.max_area = max_area
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = MODEL_IDS[backend]
        auto_processor: Any = AutoProcessor
        self.processor: Any = auto_processor.from_pretrained(model_id, cache_dir=str(cache_dir))
        model: Any = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=str(cache_dir)
        )
        self.model = model.to(self.device).eval()
        self.queries = [*PRODUCT_QUERIES, HAND_QUERY, HEAD_QUERY, PHONE_QUERY]

    def start_session(self, first_image: Image) -> None:
        """No per-session state."""

    def detect(self, image: Image) -> list[Detection]:
        """Detect products and hands in one RGB frame."""
        import torch

        h, w = image.shape[:2]
        pil = PILImage.fromarray(image)
        if self.backend == "owlv2":
            text: Any = [self.queries]
        else:
            text = " ".join(f"{q}." for q in self.queries)
        inputs = self.processor(text=text, images=pil, return_tensors="pt").to(self.device)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"),
        ):
            outputs = self.model(**inputs)
        head_threshold = HEAD_THRESHOLDS[self.backend]
        low = min(self.threshold, head_threshold)
        if self.backend == "owlv2":
            res = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=low,
                target_sizes=[(h, w)],
                text_labels=[self.queries],
            )[0]
        else:
            res = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=low,
                text_threshold=low,
                target_sizes=[(h, w)],
            )[0]
        names = res.get("text_labels") or res.get("labels")
        dets: list[Detection] = []
        for box, score, name in zip(res["boxes"], res["scores"], names, strict=True):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            label = label_of(str(name))
            if float(score) < (head_threshold if label == "head" else self.threshold):
                continue
            dets.append(
                Detection(
                    label,
                    float(score),
                    max(0.0, x1),
                    max(0.0, y1),
                    min(float(w), x2),
                    min(float(h), y2),
                )
            )
        return clean(dets, w, h, self.max_area)
