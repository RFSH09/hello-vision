"""Replay a video as one shopping session.

M1 status: the capture side only. The video is wrapped in door_open/door_close events and
turned into a :class:`Session`; the cart stays ``null`` until detection and identification
(M2) and tracking/fusion (M3) exist.

Usage: ``uv run python -m sim.replay <video>``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from hello_vision.capture.recorder import SessionBuilder
from hello_vision.capture.sources import VideoFileSource
from hello_vision.capture.types import DoorEvent, DoorEventType, Session


class SessionSummary(BaseModel):
    """What the simulator prints for a replayed session."""

    session_id: str
    camera_ids: list[str]
    num_frames: int
    duration_s: float
    width: int
    height: int
    cart: list[dict[str, object]] | None
    note: str


def replay_video(path: Path, session_id: str = "replay", camera_id: str = "cam0") -> Session:
    """Wrap a whole video into one session (door opens at the first frame, closes at the last).

    Raises:
        ValueError: If the video has no frames.
    """
    builder = SessionBuilder()
    opened = False
    last_ts = 0.0
    for frame in VideoFileSource(path, camera_id):
        if not opened:
            builder.on_door(DoorEvent(DoorEventType.OPEN, frame.timestamp, session_id))
            opened = True
        builder.push_frame(frame)
        last_ts = frame.timestamp
    session = builder.on_door(DoorEvent(DoorEventType.CLOSE, last_ts)) if opened else None
    if session is None:
        raise ValueError(f"no frames in {path}")
    return session


def summarize(session: Session) -> SessionSummary:
    """Build the printable summary (cart is ``None`` until recognition exists)."""
    h, w = session.frames[0].image.shape[:2]
    return SessionSummary(
        session_id=session.session_id,
        camera_ids=session.camera_ids,
        num_frames=len(session.frames),
        duration_s=round(session.duration, 3),
        width=w,
        height=h,
        cart=None,
        note="recognition not implemented yet (detection M2, tracking/fusion M3)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--session-id", default="replay")
    args = parser.parse_args(argv)
    summary = summarize(replay_video(args.video, args.session_id))
    print(json.dumps(summary.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
