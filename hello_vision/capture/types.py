"""Core capture types: frames, door events, sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class Frame:
    """One camera frame.

    Attributes:
        camera_id: Which camera produced it.
        index: Frame counter within its source.
        timestamp: Seconds (source clock; video files start at 0).
        image: ``H x W x 3`` RGB ``uint8`` array.
    """

    camera_id: str
    index: int
    timestamp: float
    image: Image


class DoorEventType(Enum):
    """Door state transitions that delimit a shopping session."""

    OPEN = "door_open"
    CLOSE = "door_close"


@dataclass(frozen=True)
class DoorEvent:
    """A door transition at ``timestamp`` (same clock as :class:`Frame`)."""

    type: DoorEventType
    timestamp: float
    session_id: str | None = None


@dataclass
class Session:
    """All frames captured between door open and door close."""

    session_id: str
    opened_at: float
    closed_at: float
    frames: list[Frame] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Seconds between door open and door close."""
        return self.closed_at - self.opened_at

    @property
    def camera_ids(self) -> list[str]:
        """Sorted ids of the cameras that contributed frames."""
        return sorted({f.camera_id for f in self.frames})
