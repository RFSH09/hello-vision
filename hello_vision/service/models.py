"""Backend contract models (docs/API.md)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from hello_vision.session.types import SessionResult


class ItemPayload(BaseModel):
    """One cart line sent to the backend."""

    sku: str
    qty: int = Field(ge=1)
    confidence: float = Field(ge=0, le=1)


class SessionResultPayload(BaseModel):
    """Body of ``POST /api/vision/session-result``.

    When ``needs_review`` is true the backend must NOT charge the customer: ``items`` is only a
    proposal for the human reviewer.
    """

    fridge_id: str
    session_id: str
    started_at: datetime
    ended_at: datetime
    items: list[ItemPayload]
    session_confidence: float = Field(ge=0, le=1)
    needs_review: bool
    evidence_ref: str
    model_version: str


class CatalogSku(BaseModel):
    """SKU entry of ``GET /api/vision/catalog``."""

    id: str
    name: str
    group: str | None = None


class CatalogResponse(BaseModel):
    """Body of ``GET /api/vision/catalog``: SKUs and per-fridge planograms."""

    skus: list[CatalogSku]
    planograms: dict[str, dict[str, list[str]]] = {}


class SessionRequest(BaseModel):
    """Body of the local ``POST /session``: analyse a recorded session video."""

    fridge_id: str
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    started_at: datetime
    ended_at: datetime
    video: str = Field(description="Path relative to the configured sessions root")


def to_payload(
    result: SessionResult,
    fridge_id: str,
    started_at: datetime,
    ended_at: datetime,
    evidence_ref: str,
) -> SessionResultPayload:
    """Convert an analysis result to the backend payload."""
    return SessionResultPayload(
        fridge_id=fridge_id,
        session_id=result.session_id,
        started_at=started_at,
        ended_at=ended_at,
        items=[
            ItemPayload(sku=i.sku, qty=i.qty, confidence=min(1.0, max(0.0, i.confidence)))
            for i in result.items
        ],
        session_confidence=min(1.0, max(0.0, result.session_confidence)),
        needs_review=result.needs_review,
        evidence_ref=evidence_ref,
        model_version=result.model_version,
    )
