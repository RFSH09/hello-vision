"""Per-fridge configuration (YAML). Paths and endpoints are config, never hard-coded."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ZoneRect(BaseModel):
    """Axis-aligned zone in normalised image coordinates (0..1)."""

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class CameraConfig(BaseModel):
    """One camera and its zones. Mounting details are still unknown (hardware questions)."""

    id: str
    kind: Literal["file", "folder", "usb", "rtsp"]
    uri: str
    shelf_zone: ZoneRect
    door_zone: ZoneRect


class StorageConfig(BaseModel):
    """Where data lives. Must be placeable inside Uzbekistan; defaults are local folders."""

    gallery_root: Path
    evidence_root: Path
    queue_root: Path
    retention_days: int = Field(default=30, ge=1)


class FridgeConfig(BaseModel):
    """Everything the module needs to know about one fridge."""

    fridge_id: str
    cameras: list[CameraConfig]
    storage: StorageConfig
    pre_roll_frames: int = Field(default=15, ge=0)

    @classmethod
    def load(cls, path: Path) -> FridgeConfig:
        """Load and validate a YAML config file."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
