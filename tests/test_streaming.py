"""Streaming analysis: incremental pushes, embedding cache, interleaved cameras."""

import numpy as np
import pytest

from eval.run import evaluate_session
from hello_vision.capture.types import Frame, Session
from hello_vision.session.analyzer import AnalyzerConfig, SessionAnalyzer
from sim.pipeline import build_demo_analyzer

from .conftest import cached_clip


class CountingEmbedder:
    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self.inner = inner
        self.calls = 0

    def embed(self, crop):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.inner.embed(crop)


def frames_of(name: str, cam: str = "cam0") -> list[Frame]:
    return [Frame(cam, i, i / 15, im) for i, im in enumerate(cached_clip(name).frames)]


def analyzer_with(identify_every: int) -> tuple[SessionAnalyzer, CountingEmbedder]:
    base = build_demo_analyzer()
    emb = CountingEmbedder(base.embedder)
    cfg = AnalyzerConfig(identify_every=identify_every)
    return SessionAnalyzer(base.detector, emb, base.index, cfg), emb


def test_live_push_then_finish_equals_batch_analyze() -> None:
    a = build_demo_analyzer()
    frames = frames_of("take_two")
    live = a.begin("s")
    for f in frames:
        live.push(f)
    streamed = live.finish()
    batch = a.analyze(Session("s", 0, 1, frames))
    assert streamed.cart_dict() == batch.cart_dict() and streamed.needs_review == batch.needs_review
    assert [e.action for e in streamed.events] == [e.action for e in batch.events]


def test_identification_cache_cuts_embedding_calls_without_changing_the_cart() -> None:
    every_frame, emb1 = analyzer_with(1)
    cached, emb3 = analyzer_with(3)
    frames = frames_of("take_two")
    r1 = every_frame.analyze(Session("s", 0, 1, frames))
    r3 = cached.analyze(Session("s", 0, 1, frames))
    assert r1.cart_dict() == r3.cart_dict() and not r3.needs_review
    assert emb3.calls < emb1.calls * 0.6


def test_interleaved_cameras_keep_separate_state() -> None:
    a = build_demo_analyzer()
    one, none = frames_of("take_one", "cam0"), frames_of("take_one", "cam1")
    live = a.begin("s")
    for f0, f1 in zip(one, none, strict=True):  # alternate frames of the two cameras
        live.push(f0)
        live.push(f1)
    result = live.finish()
    truth = {c.sku: c.qty for c in cached_clip("take_one").truth.cart}
    assert result.cart_dict() == truth and not result.needs_review


def test_finish_without_frames_flags_review() -> None:
    r = build_demo_analyzer().begin("empty").finish()
    assert r.needs_review and r.items == []


def test_evaluate_session_reports_split_timing() -> None:
    frames = frames_of("take_nothing")
    e = evaluate_session(build_demo_analyzer(), Session("s", 0, len(frames) / 15, frames), {}, [])
    assert e.processing_s > 0 and e.latency_s >= 0
    assert e.latency_s < e.processing_s  # after door close only the final decision remains
    assert e.realtime_factor == pytest.approx(e.processing_s / (len(frames) / 15))
    assert isinstance(frames[0].image, np.ndarray)
