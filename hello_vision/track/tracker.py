"""Multi-object tracker (greedy IoU + centre-distance association).

Deliberately simple and dependency-free. ByteTrack-style two-stage association and Kalman
prediction can replace it later without changing the :class:`Track` interface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hello_vision.detect.types import Detection, iou
from hello_vision.identify.index import Match


@dataclass(frozen=True)
class Observation:
    """A detection assigned to a track at ``frame``; ``match`` is its SKU identification."""

    frame: int
    det: Detection
    match: Match | None = None


@dataclass
class Track:
    """A tracked object over time."""

    id: int
    label: str
    obs: list[Observation] = field(default_factory=list)
    missed: int = 0

    @property
    def first_frame(self) -> int:
        """Frame index of the first observation."""
        return self.obs[0].frame

    @property
    def last_frame(self) -> int:
        """Frame index of the last observation."""
        return self.obs[-1].frame

    @property
    def hits(self) -> int:
        """Number of observations."""
        return len(self.obs)

    @property
    def start_center(self) -> tuple[float, float]:
        """Centre of the first box."""
        return self.obs[0].det.center

    @property
    def end_center(self) -> tuple[float, float]:
        """Centre of the last box."""
        return self.obs[-1].det.center

    @property
    def displacement(self) -> float:
        """Straight-line distance between first and last centre."""
        (x1, y1), (x2, y2) = self.start_center, self.end_center
        return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


MatchFn = Callable[[Detection, "Track | None"], "Match | None"]
"""Called once per assigned detection, AFTER association, with the track it joined (``None``
for a brand-new track). It lets the caller identify a crop only when needed (e.g. every few
frames per track) and reuse the track's previous identification otherwise."""


class Tracker:
    """Associates detections of one label across frames.

    Args:
        label: Label this tracker handles (``product`` or ``hand``).
        iou_threshold: Minimum IoU for a primary match.
        max_distance: Fallback centre-distance (px) for matching when boxes jump or change shape.
        max_missed: Frames a track may go unmatched before it is closed.
    """

    def __init__(
        self,
        label: str,
        iou_threshold: float = 0.2,
        max_distance: float = 150.0,
        max_missed: int = 3,
    ) -> None:
        self.label = label
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._active: list[Track] = []
        self._next_id = 0

    def _score(self, track: Track, det: Detection) -> float:
        last = track.obs[-1].det
        overlap = iou(last, det)
        if overlap >= self.iou_threshold:
            return 1.0 + overlap
        (x1, y1), (x2, y2) = last.center, det.center
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return 1.0 - dist / self.max_distance if dist < self.max_distance else 0.0

    def update(
        self, frame: int, dets: Sequence[Detection], match_fn: MatchFn | None = None
    ) -> dict[int, Track]:
        """Feed the detections of one frame (only this tracker's label is used).

        Returns:
            For each detection of this tracker's label, its index in ``dets`` -> the track it
            was assigned to (an existing one or a new one).
        """
        idx = [i for i, d in enumerate(dets) if d.label == self.label]
        mine = [dets[i] for i in idx]
        pairs = sorted(
            (
                (self._score(t, d), ti, di)
                for ti, t in enumerate(self._active)
                for di, d in enumerate(mine)
            ),
            reverse=True,
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        assigned: dict[int, Track] = {}
        for score, ti, di in pairs:
            if score <= 0:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            track = self._active[ti]
            match = match_fn(mine[di], track) if match_fn else None
            track.obs.append(Observation(frame, mine[di], match))
            track.missed = 0
            assigned[idx[di]] = track
        still: list[Track] = []
        for ti, t in enumerate(self._active):
            if ti not in used_t:
                t.missed += 1
            if t.missed <= self.max_missed:
                still.append(t)
        for di, d in enumerate(mine):
            if di not in used_d:
                match = match_fn(d, None) if match_fn else None
                t = Track(self._next_id, self.label, [Observation(frame, d, match)])
                self._next_id += 1
                self.tracks.append(t)
                still.append(t)
                assigned[idx[di]] = t
        self._active = still
        return assigned
