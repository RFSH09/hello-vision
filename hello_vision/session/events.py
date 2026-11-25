"""TAKE/RETURN decisions from product tracks, using zones and hand association."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from hello_vision.session.types import Action, ActionEvent
from hello_vision.track.association import hand_associated
from hello_vision.track.tracker import Track


@dataclass(frozen=True)
class ZoneConfig:
    """Geometry used by the action logic (pixels).

    Attributes:
        width: Image width.
        height: Image height.
        shelf_zone_y_max: Fraction of the height; below it is the shelf, beyond it the door zone.
        min_move_px: Minimum track displacement to count as "moved".
        start_grace_frames: A track first seen within this many frames of the session start
            counts as already on the shelf when the door opened.
        min_hits: Tracks with fewer observations are treated as noise.
        hand_margin: Distance (px) for hand-product association.
    """

    width: int
    height: int
    shelf_zone_y_max: float = 0.7
    min_move_px: float = 80.0
    start_grace_frames: int = 2
    min_hits: int = 4
    hand_margin: float = 40.0

    @property
    def split_y(self) -> float:
        """Pixel row separating the shelf zone (above) from the door zone (below)."""
        return self.shelf_zone_y_max * self.height


def track_identity(track: Track) -> tuple[str | None, float]:
    """Return ``(sku, confidence)`` for a product track from its per-frame matches.

    Votes are weighted by similarity score. The SKU is ``None`` (unknown) when matches that
    fell below the similarity threshold carry at least as much weight as the best SKU.
    Confidence = vote share of the winning SKU x its mean similarity score.
    """
    known: dict[str, float] = defaultdict(float)
    votes: dict[str, int] = defaultdict(int)
    unknown = 0.0
    total = 0.0
    for o in track.obs:
        if o.match is None:
            continue
        total += o.match.score
        if o.match.sku is None:
            unknown += o.match.score
        else:
            known[o.match.sku] += o.match.score
            votes[o.match.sku] += 1
    if not known or total == 0:
        return None, 0.0
    sku, weight = max(known.items(), key=lambda kv: kv[1])
    if unknown >= weight:
        return None, 0.0
    return sku, (weight / total) * (weight / votes[sku])


def derive_events(
    products: list[Track], hands: list[Track], zones: ZoneConfig
) -> list[ActionEvent]:
    """Turn product tracks into TAKE / RETURN events.

    * origin shelf (present when the door opened, or first seen in the shelf zone) and
      destination door side (left the frame at the bottom or ended in the door zone): TAKE.
    * origin door side and destination shelf: RETURN.
    * everything else (stationary, shown-and-removed, shelf reshuffle) produces no event.
    """
    events: list[ActionEvent] = []
    for t in products:
        if t.hits < zones.min_hits or t.displacement < zones.min_move_px:
            continue
        sy, ey = t.start_center[1], t.end_center[1]
        last = t.obs[-1].det
        origin_shelf = t.first_frame <= zones.start_grace_frames or sy < zones.split_y
        dest_door = last.y2 >= zones.height - 2 or ey >= zones.split_y
        action: Action | None = None
        if origin_shelf and dest_door:
            action = Action.TAKE
        elif not origin_shelf and not dest_door:
            action = Action.RETURN
        if action is None:
            continue
        sku, conf = track_identity(t)
        events.append(
            ActionEvent(
                action,
                sku,
                conf,
                t.id,
                t.first_frame,
                t.last_frame,
                hand_associated(t, hands, zones.hand_margin),
            )
        )
    return sorted(events, key=lambda e: e.frame_start)
