"""Unit tests for tracking, TAKE/RETURN logic, shelf diff and fusion (no images needed)."""

import pytest

from hello_vision.detect.types import Detection
from hello_vision.identify.index import Match
from hello_vision.session.events import ZoneConfig, derive_events, track_identity
from hello_vision.session.fusion import FusionConfig, fuse
from hello_vision.session.shelf import UNKNOWN_KEY, frame_counts, shelf_diff
from hello_vision.session.types import Action, ActionEvent, AnnotatedDetection
from hello_vision.track.association import hand_associated
from hello_vision.track.tracker import Observation, Track, Tracker

ZONES = ZoneConfig(960, 540)


def box(cx: float, cy: float, w: float = 60, h: float = 100, label: str = "product") -> Detection:
    return Detection(label, 1.0, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def product_track(
    tid: int,
    path: list[tuple[float, float]],
    sku: str | None = "a",
    start: int = 0,
    score: float = 0.95,
) -> Track:
    m = Match(sku, score, 0.1, sku or "a")
    t = Track(tid, "product")
    for i, (x, y) in enumerate(path):
        t.obs.append(Observation(start + i, box(x, y), m))
    return t


def hand_track(tid: int, path: list[tuple[float, float]], start: int = 0) -> Track:
    t = Track(tid, "hand")
    for i, (x, y) in enumerate(path):
        t.obs.append(Observation(start + i, box(x, y, 140, 200, "hand")))
    return t


def line(x0: float, y0: float, x1: float, y1: float, n: int = 10) -> list[tuple[float, float]]:
    return [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)) for i in range(n)]


TAKE_PATH = line(300, 250, 300, 560)  # shelf -> out through the bottom
RETURN_PATH = line(300, 600, 300, 250)  # door side -> shelf
HAND_DOWN = line(300, 300, 300, 600)


def test_tracker_follows_and_splits_objects() -> None:
    tr = Tracker("product")
    for f in range(6):
        tr.update(f, [(box(100 + 5 * f, 100), None), (box(400, 300 + 5 * f), None)])
    assert len(tr.tracks) == 2 and all(t.hits == 6 for t in tr.tracks)
    tr.update(6, [(box(500, 50), None)])  # a far-away new object
    assert len(tr.tracks) == 3


def test_tracker_closes_lost_tracks_and_ignores_other_labels() -> None:
    tr = Tracker("product", max_missed=1)
    tr.update(0, [(box(100, 100), None), (box(300, 300, label="hand"), None)])
    for f in (1, 2, 3):
        tr.update(f, [])
    tr.update(4, [(box(100, 100), None)])  # same place but the old track is closed
    assert len(tr.tracks) == 2 and tr.tracks[0].hits == 1


def test_take() -> None:
    ev = derive_events([product_track(0, TAKE_PATH)], [hand_track(1, HAND_DOWN)], ZONES)
    assert [(e.action, e.sku, e.hand_associated) for e in ev] == [(Action.TAKE, "a", True)]


def test_return() -> None:
    ev = derive_events(
        [product_track(0, RETURN_PATH, start=20)], [hand_track(1, RETURN_PATH, 20)], ZONES
    )
    assert [(e.action, e.sku) for e in ev] == [(Action.RETURN, "a")]


def test_nothing_taken_stationary_and_noise() -> None:
    still = product_track(0, [(300, 250)] * 10)
    blip = product_track(1, TAKE_PATH[:2])  # too short: noise
    assert derive_events([still, blip], [], ZONES) == []


def test_shown_and_removed_is_not_a_take() -> None:
    path = line(300, 600, 300, 500) + line(300, 500, 300, 600)  # door side only
    assert derive_events([product_track(0, path, start=10)], [hand_track(1, path, 10)], ZONES) == []


def test_two_items() -> None:
    tracks = [
        product_track(0, TAKE_PATH, "a"),
        product_track(1, line(500, 250, 500, 560), "b", start=20),
    ]
    ev = derive_events(
        tracks, [hand_track(2, HAND_DOWN), hand_track(3, line(500, 300, 500, 600), 20)], ZONES
    )
    lines, conf, review, reasons = fuse(ev, {"a": 1, "b": 1})
    assert [(ln.sku, ln.qty) for ln in lines] == [("a", 1), ("b", 1)] and not review and not reasons
    assert conf == pytest.approx(0.95)


def test_swap_take_a_return_it_take_b() -> None:
    tracks = [
        product_track(0, TAKE_PATH, "a"),
        product_track(1, RETURN_PATH, "a", start=15),
        product_track(2, line(500, 250, 500, 560), "b", start=40),
    ]
    hands = [
        hand_track(9, HAND_DOWN),
        hand_track(8, RETURN_PATH, 15),
        hand_track(7, line(500, 300, 500, 600), 40),
    ]
    lines, _, review, _ = fuse(derive_events(tracks, hands, ZONES), {"b": 1})
    assert [(ln.sku, ln.qty) for ln in lines] == [("b", 1)] and not review


def test_returning_an_unseen_item_needs_review() -> None:
    ev = derive_events(
        [product_track(0, RETURN_PATH, start=5)], [hand_track(1, RETURN_PATH, 5)], ZONES
    )
    lines, _, review, reasons = fuse(ev, {})
    assert lines == [] and review and any("not taken" in r for r in reasons)


def test_unknown_product_needs_review() -> None:
    ev = derive_events([product_track(0, TAKE_PATH, None)], [hand_track(1, HAND_DOWN)], ZONES)
    assert ev[0].sku is None
    _, _, review, reasons = fuse(ev, {UNKNOWN_KEY: 1})
    assert review and any("unknown product" in r for r in reasons)


def test_motion_without_hand_needs_review() -> None:
    ev = derive_events([product_track(0, TAKE_PATH)], [], ZONES)
    assert not ev[0].hand_associated
    _, _, review, reasons = fuse(ev, {"a": 1})
    assert review and any("without a hand" in r for r in reasons)


def test_shelf_disagreement_and_missing_snapshot_need_review() -> None:
    ev = derive_events([product_track(0, TAKE_PATH)], [hand_track(1, HAND_DOWN)], ZONES)
    _, _, review, reasons = fuse(ev, {})
    assert review and any("disagrees" in r for r in reasons)
    _, conf, review, reasons = fuse(ev, None, FusionConfig(min_confidence=0.75))
    assert review and conf < 0.75 and any("unavailable" in r for r in reasons)


def test_low_confidence_needs_review_and_clean_empty_session_passes() -> None:
    ev = derive_events([product_track(0, TAKE_PATH, score=0.5)], [hand_track(1, HAND_DOWN)], ZONES)
    ev = [
        ActionEvent(e.action, e.sku, 0.5, e.track_id, e.frame_start, e.frame_end, True) for e in ev
    ]
    _, _, review, reasons = fuse(ev, {"a": 1})
    assert review and any("low identification" in r for r in reasons)
    lines, conf, review, reasons = fuse([], {})
    assert lines == [] and not review and conf == FusionConfig().empty_confidence


def test_unidentified_shelf_item_needs_review() -> None:
    _, _, review, reasons = fuse([], {UNKNOWN_KEY: 1})
    assert review and any("unidentified" in r for r in reasons)


def test_track_identity_votes() -> None:
    t = Track(0, "product")
    for sku, score in [("a", 0.9), ("a", 0.9), ("b", 0.8), (None, 0.5)]:
        t.obs.append(Observation(len(t.obs), box(1, 1), Match(sku, score, 0.1, sku or "a")))
    assert track_identity(t) == ("a", pytest.approx(1.8 / 3.1 * 0.9))
    t2 = Track(1, "product")
    t2.obs.append(Observation(0, box(1, 1), Match(None, 0.5, 0.1, "a")))
    t2.obs.append(Observation(1, box(1, 1), None))
    assert track_identity(t2) == (None, 0.0)
    assert track_identity(Track(2, "product")) == (None, 0.0)


def test_hand_association() -> None:
    p = product_track(0, TAKE_PATH)
    assert hand_associated(p, [hand_track(1, HAND_DOWN)])
    assert not hand_associated(p, [hand_track(1, line(800, 100, 800, 120))])
    assert not hand_associated(p, [hand_track(1, HAND_DOWN, start=100)])  # never at the same time


def ann(label: str, sku: str | None, y: float) -> AnnotatedDetection:
    return AnnotatedDetection(box(100, y, label=label), 0, sku)


def test_frame_counts_only_shelf_products() -> None:
    f = [
        ann("product", "a", 200),
        ann("product", "a", 250),
        ann("product", None, 200),
        ann("product", "b", 500),
        ann("hand", None, 200),
    ]
    assert dict(frame_counts(f, ZONES)) == {"a": 2, UNKNOWN_KEY: 1}


def test_shelf_diff_median_ignores_single_frame_misses() -> None:
    full = [ann("product", "a", 200), ann("product", "b", 200)]
    after_missing_b = [ann("product", "a", 200)]
    frames = [
        full,
        full,
        full,
        [ann("hand", None, 300)],
        after_missing_b,
        after_missing_b,
        after_missing_b + full[1:],
    ]
    assert shelf_diff(frames, ZONES, k=3) == {"b": 1}
    assert shelf_diff([full, full, full, full], ZONES) == {}


def test_shelf_diff_unavailable_without_clear_frames() -> None:
    hand = [ann("hand", None, 300)]
    assert shelf_diff([hand, hand, hand], ZONES) is None
    assert shelf_diff([[ann("product", "a", 200)]], ZONES) is None
