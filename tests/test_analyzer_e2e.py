"""End-to-end: synthetic session -> baseline pipeline -> exact cart."""

import numpy as np
import pytest

from hello_vision.capture.types import Frame, Session
from hello_vision.session.types import SessionResult
from hello_vision.synth.session import SCENARIOS
from sim.pipeline import build_demo_analyzer

from .conftest import cached_clip


def run(scenario: str, seed: int = 0, cam: str = "cam0") -> SessionResult:
    clip = cached_clip(scenario, seed)
    frames = [Frame(cam, i, i / 15, im) for i, im in enumerate(clip.frames)]
    return build_demo_analyzer().analyze(Session(clip.truth.session_id, 0.0, 1.0, frames))


HARD = {"occluded"}  # known to be too hard for the baseline: it must flag review, never mis-charge


@pytest.mark.parametrize("scenario", sorted(set(SCENARIOS) - HARD))
def test_exact_cart_without_review(scenario: str) -> None:
    result = run(scenario)
    truth = {c.sku: c.qty for c in cached_clip(scenario).truth.cart}
    assert result.cart_dict() == truth, result.reasons
    assert not result.needs_review, result.reasons
    assert len(result.frames) == cached_clip(scenario).truth.num_frames
    assert all(0 <= ln.confidence <= 1 for ln in result.items)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_safety_invariant_unflagged_sessions_are_exact(scenario: str, seed: int = 0) -> None:
    """The property that protects customers: if the module does not ask for review, it is right."""
    result = run(scenario, seed)
    truth = {c.sku: c.qty for c in cached_clip(scenario, seed).truth.cart}
    assert result.needs_review or result.cart_dict() == truth, (scenario, seed, result.reasons)


def test_occluded_scenario_is_flagged_not_guessed() -> None:
    flagged = [run("occluded", s).needs_review for s in (0, 1)]
    assert all(flagged)


def test_events_and_annotations() -> None:
    result = run("take_and_return")
    assert [e.action.value for e in result.events] == ["take", "return"]
    assert result.shelf_diff == {}
    tracked = [a for f in result.frames for a in f if a.det.label == "product"]
    assert tracked and all(a.track_id >= 0 for a in tracked)


def test_empty_session_needs_review() -> None:
    result = build_demo_analyzer().analyze(Session("s", 0.0, 1.0, []))
    assert result.needs_review and result.items == []


def test_unknown_gallery_forces_review() -> None:
    clip = cached_clip("take_one")
    analyzer = build_demo_analyzer(threshold=1.01)  # nothing can match
    frames = [Frame("c", i, i / 15, im) for i, im in enumerate(clip.frames)]
    result = analyzer.analyze(Session("s", 0.0, 1.0, frames))
    assert result.needs_review and result.items == []


def test_two_cameras_that_disagree_need_review() -> None:
    a, b = cached_clip("take_one"), cached_clip("take_nothing")
    frames = [Frame("cam0", i, i / 15, im) for i, im in enumerate(a.frames)]
    frames += [Frame("cam1", i, i / 15, im) for i, im in enumerate(b.frames)]
    result = build_demo_analyzer().analyze(Session("s", 0.0, 1.0, frames))
    assert result.needs_review and "cameras disagree on the cart" in result.reasons


def test_two_cameras_that_agree_pass() -> None:
    a = cached_clip("take_one")
    frames = [
        Frame(c, i, i / 15, np.array(im)) for c in ("cam0", "cam1") for i, im in enumerate(a.frames)
    ]
    result = build_demo_analyzer().analyze(Session("s", 0.0, 1.0, frames))
    assert not result.needs_review and result.cart_dict() == {c.sku: c.qty for c in a.truth.cart}
