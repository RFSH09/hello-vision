"""Service settings read from environment variables (see ``.env.example``). No secrets in code."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _parse_regions(text: str) -> tuple[tuple[float, float, float, float], ...]:
    """Parse ``"x1,y1,x2,y2;x1,y1,x2,y2"`` (normalised) into rectangles."""
    out = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        x1, y1, x2, y2 = (float(v) for v in part.split(","))
        out.append((x1, y1, x2, y2))
    return tuple(out)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration of the service."""

    fridge_id: str
    backend_url: str
    hmac_secret: str
    sessions_root: Path
    evidence_root: Path
    queue_root: Path
    retention_days: int = 30
    review_token: str | None = None
    face_model: Path | None = None
    blur_regions: tuple[tuple[float, float, float, float], ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``VISION_*`` variables.

        Raises:
            KeyError: If a required variable is missing.
        """
        e = os.environ if env is None else env
        return cls(
            fridge_id=e["VISION_FRIDGE_ID"],
            backend_url=e["VISION_BACKEND_URL"],
            hmac_secret=e["VISION_HMAC_SECRET"],
            sessions_root=Path(e["VISION_SESSIONS_ROOT"]),
            evidence_root=Path(e["VISION_EVIDENCE_ROOT"]),
            queue_root=Path(e["VISION_QUEUE_ROOT"]),
            retention_days=int(e.get("VISION_RETENTION_DAYS", "30")),
            review_token=e.get("VISION_REVIEW_TOKEN") or None,
            face_model=Path(e["VISION_FACE_MODEL"]) if e.get("VISION_FACE_MODEL") else None,
            blur_regions=_parse_regions(e.get("VISION_BLUR_REGIONS", "")),
        )
