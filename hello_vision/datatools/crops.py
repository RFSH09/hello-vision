"""Real product crops from processed videos, for labelling and for the SKU gallery.

Input is the per-video JSON written by ``sim.run_video`` (track boxes with a "held by a hand"
flag). Everything written here is private data and belongs in git-ignored folders.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hello_vision.capture.types import Image

Row = list[float]  # [time_s, x1, y1, x2, y2, score, held]


@dataclass
class Candidate:
    """One cut-out of a track."""

    time_s: float
    crop: Image
    quality: float


def held_outside(boxes: Sequence[Row], frame_height: int, shelf_line: float = 0.45) -> list[Row]:
    """Boxes held by a hand and below the shelf line (an item that has left the shelf)."""
    return [b for b in boxes if b[6] and (b[2] + b[4]) / 2 > shelf_line * frame_height]


def quality(crop: Image) -> float:
    """Prefer sharp and large crops: Laplacian variance (at 128 px) times sqrt of the area."""
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (128, 128))
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(sharp * float(crop.shape[0] * crop.shape[1]) ** 0.5)


def pick(candidates: Sequence[Candidate], n: int = 4, min_gap_s: float = 0.35) -> list[Candidate]:
    """The ``n`` best candidates that are at least ``min_gap_s`` apart, filling up if needed."""
    ranked = sorted(candidates, key=lambda c: c.quality, reverse=True)
    chosen: list[Candidate] = []
    for c in ranked:
        if len(chosen) < n and all(abs(c.time_s - o.time_s) >= min_gap_s for o in chosen):
            chosen.append(c)
    for c in ranked:
        if len(chosen) < n and all(c is not o for o in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.time_s)


def pad_crop(image: Image, box: Sequence[float], pad: float = 0.08) -> Image:
    """Crop ``(x1, y1, x2, y2)`` with a relative margin, clipped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    right, bottom = min(w, int(x2 + dx) + 1), min(h, int(y2 + dy) + 1)
    return image[top : max(bottom, top + 2), left : max(right, left + 2)]


def track_candidates(
    video: Path,
    tracks: Sequence[dict[str, Any]],
    min_boxes: int = 2,
    shelf_line: float = 0.45,
    prepare: Callable[[Image], Image] | None = None,
    max_area: float = 0.08,
    pad: float = 0.08,
) -> dict[int, list[Candidate]]:
    """Cut every held-outside box of the usable tracks out of ``video`` (one sequential pass).

    Args:
        video: Source video.
        tracks: ``tracks`` list of the video's JSON report.
        min_boxes: Tracks with fewer held-outside boxes are skipped.
        shelf_line: See :func:`held_outside`.
        prepare: Applied to each used frame before cropping (e.g. face blurring).
        max_area: Boxes larger than this fraction of the frame are ignored (a held product is
            small; the huge boxes cover arms or the whole area outside the door).

        pad: Relative margin added around each box.

    Returns:
        Track id -> candidates.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = max_area * height * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    wanted: dict[float, list[tuple[int, Row]]] = {}
    for t in tracks:
        rows = [
            r
            for r in held_outside(t["boxes"], height, shelf_line)
            if (r[3] - r[1]) * (r[4] - r[2]) <= limit
        ]
        if len(rows) >= min_boxes:
            for r in rows:
                wanted.setdefault(round(r[0], 2), []).append((int(t["track_id"]), r))
    out: dict[int, list[Candidate]] = {}
    index = -1
    try:
        while wanted and cap.grab():
            index += 1
            key = round(index / fps, 2)
            if key not in wanted:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            frame = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            if prepare:
                frame = prepare(frame)
            for track_id, r in wanted.pop(key):
                crop = pad_crop(frame, r[1:5], pad)
                out.setdefault(track_id, []).append(Candidate(r[0], crop, quality(crop)))
    finally:
        cap.release()
    return out
