"""Evidence clips (blurred, with overlays), result files, retention and review queues."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from hello_vision.capture.types import Image, Session
from hello_vision.privacy.blur import FaceBlurrer
from hello_vision.session.types import SessionResult

COLORS = {"product": (40, 200, 40), "hand": (230, 150, 30)}


def overlay(image: Image, dets: list, fps_text: str = "") -> Image:  # type: ignore[type-arg]
    """Draw boxes, SKU and track id for annotated detections on a copy of ``image``."""
    out = image.copy()
    for a in dets:
        d = a.det
        color = COLORS.get(d.label, (200, 200, 200))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{a.sku or ('?' if d.label == 'product' else d.label)}#{a.track_id}"
        cv2.putText(
            out, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
    if fps_text:
        cv2.putText(out, fps_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def session_dir(root: Path, session_id: str) -> Path:
    """Folder for one session's evidence (``session_id`` must be filename-safe)."""
    if not session_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"unsafe session id: {session_id!r}")
    return root / session_id


def write_evidence(
    session: Session,
    result: SessionResult,
    root: Path,
    blurrer: FaceBlurrer,
    fps: float = 15.0,
) -> str:
    """Write ``clip.mp4`` (blurred first, then overlaid) and ``result.json``; return its reference.

    Raw frames are never written. Only the first camera is written to the clip.
    """
    folder = session_dir(root, result.session_id)
    folder.mkdir(parents=True, exist_ok=True)
    cam = session.frames[0].camera_id if session.frames else ""
    frames = [f for f in session.frames if f.camera_id == cam]
    if frames:
        h, w = frames[0].image.shape[:2]
        writer = cv2.VideoWriter(
            str(folder / "clip.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h)
        )
        try:
            for i, f in enumerate(frames):
                dets = result.frames[i] if i < len(result.frames) else []
                shown = overlay(blurrer.blur(f.image), dets, f"frame {i}")
                writer.write(cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    summary = {
        "session_id": result.session_id,
        "items": [{"sku": i.sku, "qty": i.qty, "confidence": i.confidence} for i in result.items],
        "session_confidence": result.session_confidence,
        "needs_review": result.needs_review,
        "reasons": result.reasons,
        "shelf_diff": result.shelf_diff,
        "events": [
            {"action": e.action.value, "sku": e.sku, "confidence": e.confidence,
             "frames": [e.frame_start, e.frame_end], "hand": e.hand_associated}
            for e in result.events
        ],
        "model_version": result.model_version,
        "num_frames": len(frames),
    }  # fmt: skip
    (folder / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return f"{result.session_id}/clip.mp4"


def read_frame_jpeg(root: Path, session_id: str, index: int) -> bytes | None:
    """Return frame ``index`` of the evidence clip as JPEG bytes, or ``None`` if unavailable."""
    path = session_dir(root, session_id) / "clip.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", bgr)
        return bytes(np.asarray(buf)) if ok else None
    finally:
        cap.release()


def purge_expired(root: Path, retention_days: int, now: float | None = None) -> list[str]:
    """Delete evidence folders older than ``retention_days``; return the removed session ids."""
    t = time.time() if now is None else now
    removed: list[str] = []
    if not root.exists():
        return removed
    for folder in root.iterdir():
        expired = t - folder.stat().st_mtime > retention_days * 86400
        if folder.is_dir() and folder.name != "to_label" and expired:
            shutil.rmtree(folder)
            removed.append(folder.name)
    return removed


def enqueue_to_label(root: Path, session_id: str, reason: str) -> Path:
    """Add a session to the active-learning "to label" queue (low confidence or corrected)."""
    q = root / "to_label"
    q.mkdir(parents=True, exist_ok=True)
    path = q / f"{session_id}.json"
    path.write_text(json.dumps({"session_id": session_id, "reason": reason}), encoding="utf-8")
    return path


def list_to_label(root: Path) -> list[str]:
    """Session ids waiting to be labelled."""
    q = root / "to_label"
    return sorted(p.stem for p in q.glob("*.json")) if q.exists() else []
