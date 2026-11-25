"""Labeling exchange formats.

* Boxes (``product`` / ``hand``, plus a SKU per product box) travel as COCO 1.0, which CVAT
  imports and exports. The SKU is stored both as ``sku`` and as the CVAT attribute ``sku``.
* Label Studio rectangle exports are read too: label ``product`` (with the SKU in a ``choices``
  result named ``sku``) or the shorthand label ``product:<sku>``; ``hand`` for hands.
* Session-level labels (the true cart and TAKE/RETURN events) use :class:`SessionLabel` JSON.

Importing is strict: bad boxes and inconsistent session labels are rejected, not guessed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from hello_vision.session.types import AnnotatedDetection

LABELS = ("product", "hand")


class LabeledBox(BaseModel):
    """One labelled box in pixels."""

    label: Literal["product", "hand"]
    sku: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _valid(self) -> LabeledBox:
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError("box must have positive width and height")
        if self.label == "hand" and self.sku is not None:
            raise ValueError("a hand box cannot have a SKU")
        return self


class LabeledFrame(BaseModel):
    """Labelled boxes of one image."""

    file_name: str
    width: int
    height: int
    boxes: list[LabeledBox]


class ActionLabel(BaseModel):
    """A labelled TAKE or RETURN event."""

    kind: Literal["take", "return"]
    sku: str
    frame: int = Field(ge=0)


class SessionLabel(BaseModel):
    """Session-level truth: the cart and the actions that lead to it."""

    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    cart: dict[str, int]
    actions: list[ActionLabel] = []

    @model_validator(mode="after")
    def _consistent(self) -> SessionLabel:
        if any(q < 1 for q in self.cart.values()):
            raise ValueError("cart quantities must be >= 1")
        if self.actions:
            net: dict[str, int] = defaultdict(int)
            for a in self.actions:
                net[a.sku] += 1 if a.kind == "take" else -1
            expected = {s: n for s, n in net.items() if n > 0}
            if expected != self.cart or any(n < 0 for n in net.values()):
                raise ValueError(f"cart {self.cart} does not match actions (net {dict(net)})")
        return self


# ------------------------------------------------------------------------------- COCO / CVAT


def export_coco(frames: Sequence[LabeledFrame], path: Path) -> None:
    """Write ``frames`` as COCO 1.0 for CVAT (use it for pre-annotations that humans correct)."""
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    for i, f in enumerate(frames, start=1):
        images.append({"id": i, "file_name": f.file_name, "width": f.width, "height": f.height})
        for b in f.boxes:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": LABELS.index(b.label) + 1,
                    "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],
                    "area": (b.x2 - b.x1) * (b.y2 - b.y1),
                    "iscrowd": 0,
                    "sku": b.sku,
                    "attributes": {"sku": b.sku or ""},
                }
            )
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(LABELS)]
    path.write_text(
        json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8"
    )


def import_coco(path: Path) -> list[LabeledFrame]:
    """Read COCO written by :func:`export_coco` or exported from CVAT after correction.

    Raises:
        ValueError: Unknown category, or an invalid box.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    frames = {
        im["id"]: LabeledFrame(
            file_name=im["file_name"], width=im["width"], height=im["height"], boxes=[]
        )
        for im in data["images"]
    }
    for a in data["annotations"]:
        label = names.get(a["category_id"])
        if label not in LABELS:
            raise ValueError(f"unknown category {label!r}")
        x, y, w, h = a["bbox"]
        sku = a.get("sku") or (a.get("attributes") or {}).get("sku") or None
        frames[a["image_id"]].boxes.append(
            LabeledBox(
                label=label, sku=sku if label == "product" else None, x1=x, y1=y, x2=x + w, y2=y + h
            )
        )
    return list(frames.values())


def preannotations(
    file_names: Sequence[str], dets: Sequence[Sequence[AnnotatedDetection]], width: int, height: int
) -> list[LabeledFrame]:
    """Turn the pipeline's detections into labeled frames to correct in CVAT (active learning)."""
    out: list[LabeledFrame] = []
    for name, frame in zip(file_names, dets, strict=True):
        boxes = [
            LabeledBox(
                label="hand" if a.det.label == "hand" else "product",
                sku=None if a.det.label == "hand" else a.sku,
                x1=a.det.x1,
                y1=a.det.y1,
                x2=a.det.x2,
                y2=a.det.y2,
            )
            for a in frame
            if a.det.x2 > a.det.x1 and a.det.y2 > a.det.y1
        ]
        out.append(LabeledFrame(file_name=name, width=width, height=height, boxes=boxes))
    return out


# ------------------------------------------------------------------------------- Label Studio


def import_label_studio(path: Path) -> list[LabeledFrame]:
    """Read a Label Studio JSON export of rectangle annotations (percent coordinates)."""
    tasks = json.loads(path.read_text(encoding="utf-8"))
    out: list[LabeledFrame] = []
    for task in tasks:
        ann = (task.get("annotations") or [{}])[0]
        results = ann.get("result", [])
        skus = {r["id"]: r["value"]["choices"][0] for r in results if r.get("from_name") == "sku"}
        boxes: list[LabeledBox] = []
        width = height = 0
        for r in results:
            if r.get("type") != "rectanglelabels":
                continue
            width, height = r["original_width"], r["original_height"]
            v = r["value"]
            name = v["rectanglelabels"][0]
            label, _, inline_sku = name.partition(":")
            sku = inline_sku or skus.get(r["id"])
            x1, y1 = v["x"] / 100 * width, v["y"] / 100 * height
            x2, y2 = x1 + v["width"] / 100 * width, y1 + v["height"] / 100 * height
            if label not in LABELS:
                raise ValueError(f"unknown label {label!r}")
            boxes.append(
                LabeledBox(
                    label=label, sku=sku if label == "product" else None, x1=x1, y1=y1, x2=x2, y2=y2
                )
            )
        name = str(task["data"].get("image", task.get("id")))
        out.append(LabeledFrame(file_name=Path(name).name, width=width, height=height, boxes=boxes))
    return out
