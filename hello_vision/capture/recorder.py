"""Ring buffer and session builder: frames + door events -> :class:`Session`."""

from __future__ import annotations

import uuid
from collections import deque

from hello_vision.capture.types import DoorEvent, DoorEventType, Frame, Session


class RingBuffer:
    """Keeps the most recent ``capacity`` frames (pre-roll before the door opens)."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._frames: deque[Frame] = deque(maxlen=capacity)

    def push(self, frame: Frame) -> None:
        """Append a frame, dropping the oldest if full."""
        if self._frames.maxlen:
            self._frames.append(frame)

    def snapshot(self) -> list[Frame]:
        """Return the buffered frames, oldest first."""
        return list(self._frames)

    def __len__(self) -> int:
        """Number of buffered frames."""
        return len(self._frames)


class SessionBuilder:
    """Collects frames between door_open and door_close.

    Feed frames with :meth:`push_frame` and door transitions with :meth:`on_door`; the
    completed :class:`Session` is returned by the closing :meth:`on_door` call.

    Args:
        pre_roll_frames: Frames kept from before the door opened (all cameras together).
    """

    def __init__(self, pre_roll_frames: int = 0) -> None:
        self._ring = RingBuffer(pre_roll_frames)
        self._session: Session | None = None

    @property
    def is_open(self) -> bool:
        """Whether a session is currently being recorded."""
        return self._session is not None

    def push_frame(self, frame: Frame) -> None:
        """Add a frame to the open session, or to the pre-roll buffer if the door is shut."""
        if self._session is None:
            self._ring.push(frame)
        else:
            self._session.frames.append(frame)

    def on_door(self, event: DoorEvent) -> Session | None:
        """Handle a door event.

        Returns:
            The finished session on ``CLOSE``; ``None`` otherwise. A ``CLOSE`` without a
            matching ``OPEN`` and a repeated ``OPEN`` are ignored.
        """
        if event.type is DoorEventType.OPEN:
            if self._session is None:
                self._session = Session(
                    session_id=event.session_id or uuid.uuid4().hex,
                    opened_at=event.timestamp,
                    closed_at=event.timestamp,
                    frames=self._ring.snapshot(),
                )
            return None
        done, self._session = self._session, None
        if done is not None:
            done.closed_at = event.timestamp
        return done
