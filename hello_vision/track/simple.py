"""ByteTrack-style tracker for low-frame-rate video: two-stage IoU association + SKU voting.

Stage 1 matches confident detections to tracks (using a constant-velocity prediction), stage 2
tries the leftover tracks against low-score detections (a half-hidden product often scores
low). Only confident detections start new tracks. Implemented from the idea of ByteTrack
(MIT); no code was copied.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from hello_vision.detect.types import Detection, iou
from hello_vision.identify.index import Match


@dataclass
class SimpleTrack:
    """One physical item followed over time."""

    id: int
    label: str
    frames: list[int] = field(default_factory=list)
    dets: list[Detection] = field(default_factory=list)
    matches: list[Match | None] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self) -> Detection:
        """Latest detection."""
        return self.dets[-1]

    def predicted(self) -> Detection:
        """Last box shifted by the latest per-step velocity."""
        d = self.dets[-1]
        if len(self.dets) < 2:
            return d
        prev = self.dets[-2]
        steps = max(1, self.frames[-1] - self.frames[-2])
        dx = (d.center[0] - prev.center[0]) / steps * (self.missed + 1)
        dy = (d.center[1] - prev.center[1]) / steps * (self.missed + 1)
        return Detection(d.label, d.score, d.x1 + dx, d.y1 + dy, d.x2 + dx, d.y2 + dy)

    def sku_vote(self) -> tuple[str | None, float, str | None]:
        """Confidence-weighted SKU vote.

        Returns:
            ``(sku or None if unknown, confidence, best guess)``. The weight of a frame is
            detection score x similarity. The track is unknown if "below threshold" frames
            outweigh the winning SKU.
        """
        known: dict[str, float] = defaultdict(float)
        sims: dict[str, list[float]] = defaultdict(list)
        guesses: dict[str, float] = defaultdict(float)
        unknown = 0.0
        total = 0.0
        for det, m in zip(self.dets, self.matches, strict=True):
            if m is None:
                continue
            w = det.score * max(m.score, 0.0)
            total += w
            if m.best_sku is not None:
                guesses[m.best_sku] += w
            if m.sku is None:
                unknown += w
            else:
                known[m.sku] += w
                sims[m.sku].append(m.score)
        guess = max(guesses, key=guesses.__getitem__) if guesses else None
        if not known:
            return None, 0.0, guess
        sku = max(known, key=known.__getitem__)
        if unknown > known[sku]:
            return None, 0.0, guess
        share = known[sku] / total
        return sku, share * sum(sims[sku]) / len(sims[sku]), guess


class ByteSimpleTracker:
    """Tracks one label.

    Args:
        label: ``product`` or ``hand``.
        high_score: Detections at or above this are used first and may start tracks.
        iou_threshold: Minimum IoU against the predicted box.
        max_distance: Fallback centre distance, as a multiple of the box diagonal.
        max_missed: Processed frames a track survives without a detection.
    """

    def __init__(
        self,
        label: str,
        high_score: float = 0.25,
        iou_threshold: float = 0.2,
        max_distance: float = 1.0,
        max_missed: int = 5,
    ) -> None:
        self.label = label
        self.high_score = high_score
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: list[SimpleTrack] = []
        self._active: list[SimpleTrack] = []

    def _affinity(self, track: SimpleTrack, det: Detection) -> float:
        pred = track.predicted()
        overlap = max(iou(pred, det), iou(track.last, det))
        if overlap >= self.iou_threshold:
            return 1.0 + overlap
        diag = ((pred.x2 - pred.x1) ** 2 + (pred.y2 - pred.y1) ** 2) ** 0.5
        (ax, ay), (bx, by) = pred.center, det.center
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        limit = self.max_distance * max(diag, 1.0)
        return 1.0 - dist / limit if dist < limit else 0.0

    def _match(
        self, tracks: list[SimpleTrack], dets: list[tuple[int, Detection]]
    ) -> dict[int, SimpleTrack]:
        pairs = sorted(
            (
                (self._affinity(t, d), ti, di)
                for ti, t in enumerate(tracks)
                for di, (_, d) in enumerate(dets)
            ),
            reverse=True,
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        out: dict[int, SimpleTrack] = {}
        for score, ti, di in pairs:
            if score <= 0:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            out[dets[di][0]] = tracks[ti]
        return out

    def update(
        self, frame: int, dets: Sequence[Detection], matches: Sequence[Match | None] | None = None
    ) -> dict[int, SimpleTrack]:
        """Add one processed frame.

        Args:
            frame: Processed-frame counter.
            dets: All detections of the frame (other labels are ignored).
            matches: SKU identification per detection (same order), or ``None``.

        Returns:
            Detection index -> track for every detection that belongs to a track.
        """
        mine = [(i, d) for i, d in enumerate(dets) if d.label == self.label]
        high = [(i, d) for i, d in mine if d.score >= self.high_score]
        low = [(i, d) for i, d in mine if d.score < self.high_score]
        assigned = self._match(self._active, high)
        taken = {id(t) for t in assigned.values()}
        assigned.update(self._match([t for t in self._active if id(t) not in taken], low))
        taken = {id(t) for t in assigned.values()}
        for i, track in assigned.items():
            track.frames.append(frame)
            track.dets.append(dets[i])
            track.matches.append(matches[i] if matches else None)
            track.missed = 0
        still: list[SimpleTrack] = []
        for t in self._active:
            if id(t) not in taken:
                t.missed += 1
            if t.missed <= self.max_missed:
                still.append(t)
        for i, d in high:
            if i not in assigned:
                t = SimpleTrack(len(self.tracks) + 1, self.label)
                t.frames.append(frame)
                t.dets.append(d)
                t.matches.append(matches[i] if matches else None)
                self.tracks.append(t)
                still.append(t)
                assigned[i] = t
        self._active = still
        return assigned
