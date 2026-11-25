"""Durable offline queue for results that could not be delivered yet.

One JSON file per session id (so enqueueing the same session twice cannot duplicate it).
Files are written atomically. Permanently rejected results move to ``dead/``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class OutboxItem:
    """A queued payload with its retry state."""

    session_id: str
    payload: dict[str, object]
    attempts: int = 0
    next_attempt_at: float = 0.0


class Outbox:
    """File-backed queue under ``root`` (a configurable, local path)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dead = root / "dead"
        root.mkdir(parents=True, exist_ok=True)
        self.dead.mkdir(exist_ok=True)

    def _path(self, session_id: str, folder: Path | None = None) -> Path:
        if not _SAFE.match(session_id):
            raise ValueError(f"unsafe session id: {session_id!r}")
        return (folder or self.root) / f"{session_id}.json"

    def _write(self, path: Path, item: OutboxItem) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(item.__dict__), encoding="utf-8")
        os.replace(tmp, path)

    def put(self, session_id: str, payload: dict[str, object]) -> bool:
        """Queue a payload. Returns False (and keeps the existing entry) if already queued."""
        path = self._path(session_id)
        try:  # exclusive create: of two concurrent puts exactly one wins
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.close(fd)
        self._write(path, OutboxItem(session_id, payload))
        return True

    def update(self, item: OutboxItem) -> None:
        """Persist new retry state for a queued item."""
        self._write(self._path(item.session_id), item)

    def due(self, now: float | None = None) -> list[OutboxItem]:
        """Items whose next attempt time has come, oldest first."""
        t = time.time() if now is None else now
        items: list[OutboxItem] = []
        for p in self.root.glob("*.json"):
            try:
                items.append(OutboxItem(**json.loads(p.read_text("utf-8"))))
            except (json.JSONDecodeError, TypeError):  # being written right now; next flush
                continue
        return sorted((i for i in items if i.next_attempt_at <= t), key=lambda i: i.next_attempt_at)

    def done(self, session_id: str) -> None:
        """Remove a delivered item."""
        self._path(session_id).unlink(missing_ok=True)

    def kill(self, session_id: str) -> None:
        """Move an item the backend permanently rejected to ``dead/`` for inspection."""
        src = self._path(session_id)
        if src.exists():
            os.replace(src, self._path(session_id, self.dead))

    def __len__(self) -> int:
        """Number of queued (undelivered) items."""
        return sum(1 for _ in self.root.glob("*.json"))
