"""Session analysis result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hello_vision.detect.types import Detection


class Action(Enum):
    """What the customer did with one product."""

    TAKE = "take"
    RETURN = "return"


@dataclass(frozen=True)
class ActionEvent:
    """One TAKE or RETURN derived from a product track.

    Attributes:
        action: TAKE or RETURN.
        sku: Identified SKU, or ``None`` for an unknown product.
        confidence: Identification confidence of the track in ``[0, 1]``.
        track_id: Product track the event came from.
        frame_start: First frame of the track.
        frame_end: Last frame of the track.
        hand_associated: Whether a hand was near the product while it moved.
    """

    action: Action
    sku: str | None
    confidence: float
    track_id: int
    frame_start: int
    frame_end: int
    hand_associated: bool


@dataclass(frozen=True)
class CartLine:
    """One cart line for billing."""

    sku: str
    qty: int
    confidence: float


@dataclass(frozen=True)
class AnnotatedDetection:
    """A detection with its track and SKU, kept for evidence clips and the review UI."""

    det: Detection
    track_id: int
    sku: str | None


@dataclass
class SessionResult:
    """Final outcome for one shopping session.

    ``needs_review`` means a human must check the clip; the customer must not be charged a
    guessed item. ``reasons`` explains why.
    """

    session_id: str
    items: list[CartLine]
    session_confidence: float
    needs_review: bool
    reasons: list[str]
    events: list[ActionEvent]
    shelf_diff: dict[str, int]
    model_version: str
    frames: list[list[AnnotatedDetection]] = field(default_factory=list)

    def cart_dict(self) -> dict[str, int]:
        """Return ``{sku: qty}`` for comparison with ground truth."""
        return {line.sku: line.qty for line in self.items}
