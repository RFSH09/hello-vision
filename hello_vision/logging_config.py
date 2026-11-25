"""Structured (JSON lines) logging for the service."""

from __future__ import annotations

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON (extra fields passed via ``extra=`` are included)."""
        skip = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
        out: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update({k: v for k, v in vars(record).items() if k not in skip})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure(level: int = logging.INFO) -> None:
    """Send all logs to stderr as JSON lines (systemd/journald collects them)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
