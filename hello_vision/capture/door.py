"""Door sensor and lock abstractions with mocks.

Real implementations wait for the answers in docs/HARDWARE_QUESTIONS.md. No real lock control
exists in this repository until the pilot is approved.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from hello_vision.capture.types import DoorEvent


class DoorSensor(Protocol):
    """Source of door transitions."""

    def events(self) -> Iterator[DoorEvent]:
        """Yield door events in time order."""
        ...


class Lock(Protocol):
    """Electronic lock interface."""

    def unlock(self) -> None:
        """Release the door."""
        ...

    def lock(self) -> None:
        """Secure the door."""
        ...


class MockDoorSensor:
    """Replays a fixed list of door events (tests and simulation)."""

    def __init__(self, events: Sequence[DoorEvent]) -> None:
        self._events = list(events)

    def events(self) -> Iterator[DoorEvent]:
        """Yield the configured events."""
        yield from self._events


class MockLock:
    """Records calls instead of driving hardware."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def unlock(self) -> None:
        """Record an unlock request."""
        self.calls.append("unlock")

    def lock(self) -> None:
        """Record a lock request."""
        self.calls.append("lock")
