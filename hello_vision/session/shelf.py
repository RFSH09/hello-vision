"""Shelf-state cross-check: SKU counts before vs after the session."""

from __future__ import annotations

import statistics
from collections import Counter

from hello_vision.session.events import ZoneConfig
from hello_vision.session.types import AnnotatedDetection

UNKNOWN_KEY = "<unknown>"


def frame_counts(dets: list[AnnotatedDetection], zones: ZoneConfig) -> Counter[str]:
    """Count products per SKU on the shelf zone of one frame (unknown products separately)."""
    c: Counter[str] = Counter()
    for a in dets:
        if a.det.label == "product" and a.det.center[1] < zones.split_y:
            c[a.sku or UNKNOWN_KEY] += 1
    return c


def _snapshot(
    frames: list[list[AnnotatedDetection]], idxs: list[int], zones: ZoneConfig
) -> Counter[str] | None:
    """Median per-SKU count over the given frames; ``None`` if there are no frames."""
    if not idxs:
        return None
    per = [frame_counts(frames[i], zones) for i in idxs]
    skus = {s for c in per for s in c}
    out: Counter[str] = Counter()
    for s in skus:
        n = int(statistics.median(c.get(s, 0) for c in per))
        if n:
            out[s] = n
    return out


def shelf_diff(
    frames: list[list[AnnotatedDetection]], zones: ZoneConfig, k: int = 5
) -> dict[str, int] | None:
    """Return ``before - after`` per SKU (positive = removed from the shelf).

    Snapshots use up to ``k`` hand-free frames at the start and at the end of the session.

    Returns:
        The difference, or ``None`` if either snapshot has no hand-free frame (evidence
        unavailable).
    """
    clear = [i for i, f in enumerate(frames) if not any(a.det.label == "hand" for a in f)]
    before = _snapshot(frames, clear[:k], zones)
    after = _snapshot(frames, clear[-k:], zones)
    if before is None or after is None:
        return None
    # the two snapshots must come from different parts of the session
    if len(clear) < 2 or clear[0] == clear[-1]:
        return None
    diff = {s: before.get(s, 0) - after.get(s, 0) for s in set(before) | set(after)}
    return {s: n for s, n in sorted(diff.items()) if n != 0}
