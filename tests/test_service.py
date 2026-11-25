import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hello_vision.capture.types import Frame, Session
from hello_vision.privacy.blur import FaceBlurrer
from hello_vision.review.ui import list_page, session_page
from hello_vision.service.app import create_app
from hello_vision.service.client import BackendClient, Dispatcher, backoff_delay
from hello_vision.service.evidence import (
    enqueue_to_label,
    list_to_label,
    purge_expired,
    read_frame_jpeg,
    session_dir,
    write_evidence,
)
from hello_vision.service.models import SessionResultPayload, to_payload
from hello_vision.service.outbox import Outbox
from hello_vision.service.settings import Settings
from hello_vision.service.signing import SIGNATURE_HEADER, TIMESTAMP_HEADER, sign, verify
from hello_vision.session.types import CartLine, SessionResult
from hello_vision.synth.session import write_video
from sim.pipeline import build_demo_analyzer

from .conftest import cached_clip

SECRET = "test-secret"
T0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 1, 10, 1, tzinfo=UTC)


def make_payload(sid: str = "s1", review: bool = False) -> SessionResultPayload:
    res = SessionResult(sid, [CartLine("a", 1, 1.4)], -0.2, review, [], [], {}, "m1")
    return to_payload(res, "f1", T0, T1, f"{sid}/clip.mp4")


# ------------------------------------------------------------------ contract + signing


def test_payload_contract_fields_and_clamping() -> None:
    p = make_payload()
    assert set(p.model_dump()) == {
        "fridge_id", "session_id", "started_at", "ended_at", "items", "session_confidence",
        "needs_review", "evidence_ref", "model_version",
    }  # fmt: skip
    assert p.session_confidence == 0.0 and p.items[0].confidence == 1.0
    with pytest.raises(ValueError):
        SessionResultPayload.model_validate(
            {**p.model_dump(), "items": [{"sku": "a", "qty": 0, "confidence": 1}]}
        )


def test_signing_roundtrip_and_tamper_and_skew() -> None:
    body = b'{"x":1}'
    sig = sign(SECRET, body, 1000)
    assert verify(SECRET, body, "1000", sig, now=1100)
    assert not verify(SECRET, b'{"x":2}', "1000", sig, now=1100)
    assert not verify("other", body, "1000", sig, now=1100)
    assert not verify(SECRET, body, "1000", sig, now=5000)  # stale
    assert not verify(SECRET, body, "abc", sig, now=1100)


# ------------------------------------------------------------------ outbox + client


def test_outbox_idempotent_and_safe(tmp_path: Path) -> None:
    box = Outbox(tmp_path)
    assert box.put("s1", {"a": 1}) and not box.put("s1", {"a": 2}) and len(box) == 1
    assert box.due()[0].payload == {"a": 1}
    with pytest.raises(ValueError):
        box.put("../x", {})
    box.kill("s1")
    assert len(box) == 0 and (tmp_path / "dead" / "s1.json").exists()
    box.put("s2", {})
    box.done("s2")
    assert len(box) == 0


def backend(handler):  # type: ignore[no-untyped-def]
    return BackendClient(
        "https://backend.test", SECRET, httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_client_outcomes_and_headers() -> None:
    seen: list[httpx.Request] = []
    codes = iter([200, 409, 500, 429, 400])

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(next(codes))

    c = backend(handler)
    p = make_payload()
    assert [c.post_result(p) for _ in range(5)] == ["sent", "sent", "retry", "retry", "rejected"]
    r = seen[0]
    assert r.headers["Idempotency-Key"] == "s1" and r.url.path == "/api/vision/session-result"
    assert verify(SECRET, r.content, r.headers[TIMESTAMP_HEADER], r.headers[SIGNATURE_HEADER])
    assert json.loads(r.content)["session_id"] == "s1"


def test_client_network_error_is_retry_and_catalog() -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    assert backend(boom).post_result(make_payload()) == "retry"

    def ok(req: httpx.Request) -> httpx.Response:
        assert SIGNATURE_HEADER in req.headers
        return httpx.Response(
            200, json={"skus": [{"id": "a", "name": "A"}], "planograms": {"f": {"s": ["a"]}}}
        )

    cat = backend(ok).fetch_catalog()
    assert cat.skus[0].id == "a" and cat.planograms["f"]["s"] == ["a"]


def test_dispatcher_queues_offline_then_delivers(tmp_path: Path) -> None:
    online = {"up": False}
    sent: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if not online["up"]:
            raise httpx.ConnectError("offline")
        sent.append(json.loads(req.content)["session_id"])
        return httpx.Response(200)

    now = {"t": 1000.0}
    d = Dispatcher(backend(handler), Outbox(tmp_path), clock=lambda: now["t"])
    r1 = d.submit(make_payload("s1"))
    r2 = d.submit(make_payload("s1"))  # duplicate submit must not duplicate the queue entry
    assert (r1.retry_later, len(d.outbox)) == (1, 1) and r2.sent == 0
    assert d.flush().sent == 0  # backoff not elapsed: nothing is attempted
    online["up"], now["t"] = True, 1e9
    assert d.flush().sent == 1 and sent == ["s1"] and len(d.outbox) == 0


def test_dispatcher_rejects_and_gives_up(tmp_path: Path) -> None:
    d = Dispatcher(backend(lambda r: httpx.Response(422)), Outbox(tmp_path / "a"))
    assert d.submit(make_payload("bad")).rejected == 1
    assert (tmp_path / "a" / "dead" / "bad.json").exists()
    d2 = Dispatcher(backend(lambda r: httpx.Response(503)), Outbox(tmp_path / "b"), max_attempts=1)
    assert d2.submit(make_payload("x")).rejected == 1


def test_backoff_grows_and_caps() -> None:
    assert backoff_delay(1, jitter=0) == 4 and backoff_delay(3, jitter=0) == 16
    assert backoff_delay(20, cap=60, jitter=0) == 60


# ------------------------------------------------------------------ privacy + evidence


class FakeFaces:
    def detect(self, image):  # type: ignore[no-untyped-def]
        return [(40, 30, 20, 20)]


def test_blur_changes_only_face_and_fixed_regions() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    out = FaceBlurrer(FakeFaces(), fixed_regions=[(0.75, 0.0, 1.0, 0.2)]).blur(img)
    assert out.shape == img.shape and img is not out
    assert not np.array_equal(out[30:50, 40:60], img[30:50, 40:60])  # face blurred
    assert not np.array_equal(out[0:20, 150:200], img[0:20, 150:200])  # fixed region blurred
    assert np.array_equal(out[70:, :100], img[70:, :100])  # untouched area unchanged
    assert np.array_equal(FaceBlurrer().blur(img), img)  # nothing configured: no change


def test_blur_edge_boxes_do_not_crash() -> None:
    class Edge:
        def detect(self, image):  # type: ignore[no-untyped-def]
            return [(-30, -30, 40, 40), (190, 90, 50, 50), (500, 500, 10, 10)]

    FaceBlurrer(Edge()).blur(np.zeros((100, 200, 3), np.uint8))


YUNET = os.environ.get("HV_TEST_YUNET")


@pytest.mark.skipif(not YUNET, reason="set HV_TEST_YUNET to the YuNet .onnx path to run")
def test_yunet_detector_runs_on_blank_image() -> None:
    from hello_vision.privacy.blur import YuNetFaceDetector

    assert YuNetFaceDetector(Path(str(YUNET))).detect(np.full((120, 160, 3), 128, np.uint8)) == []


def test_yunet_missing_model_fails_closed(tmp_path: Path) -> None:
    from hello_vision.privacy.blur import YuNetFaceDetector

    with pytest.raises(FileNotFoundError):
        YuNetFaceDetector(tmp_path / "missing.onnx")


def test_evidence_clip_frames_purge_and_queue(tmp_path: Path) -> None:
    clip = cached_clip("take_one")
    frames = [Frame("cam0", i, i / 15, im) for i, im in enumerate(clip.frames[:20])]
    session = Session("s-1", 0, 1, frames)
    result = build_demo_analyzer().analyze(session)
    ref = write_evidence(session, result, tmp_path, FaceBlurrer())
    assert ref == "s-1/clip.mp4"
    summary = json.loads((tmp_path / "s-1" / "result.json").read_text())
    assert summary["num_frames"] == 20 and not any(
        p.suffix in {".png", ".jpg"} for p in (tmp_path / "s-1").iterdir()
    )
    jpeg = read_frame_jpeg(tmp_path, "s-1", 3)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    assert (
        read_frame_jpeg(tmp_path, "s-1", 999) is None
        and read_frame_jpeg(tmp_path, "nope", 0) is None
    )
    enqueue_to_label(tmp_path, "s-1", "needs_review")
    assert list_to_label(tmp_path) == ["s-1"] and list_to_label(tmp_path / "missing") == []
    assert purge_expired(tmp_path, 30) == []
    old = time.time() - 40 * 86400
    os.utime(tmp_path / "s-1", (old, old))
    assert purge_expired(tmp_path, 30) == ["s-1"] and not (tmp_path / "s-1").exists()
    assert (tmp_path / "to_label").exists()  # the labelling queue survives retention purges
    with pytest.raises(ValueError):
        session_dir(tmp_path, "../evil")


# ------------------------------------------------------------------ settings + app


def test_settings_from_env() -> None:
    env = {
        "VISION_FRIDGE_ID": "f", "VISION_BACKEND_URL": "u", "VISION_HMAC_SECRET": "s",
        "VISION_SESSIONS_ROOT": "/a", "VISION_EVIDENCE_ROOT": "/b", "VISION_QUEUE_ROOT": "/c",
        "VISION_REVIEW_TOKEN": "",
    }  # fmt: skip
    s = Settings.from_env(env)
    assert s.retention_days == 30 and s.review_token is None and s.evidence_root == Path("/b")
    assert s.face_model is None and s.blur_regions == ()
    regions = "0,0,1,0.2;0.5,0.5,1,1"
    s2 = Settings.from_env({**env, "VISION_FACE_MODEL": "/m.onnx", "VISION_BLUR_REGIONS": regions})
    assert s2.face_model == Path("/m.onnx") and s2.blur_regions == (
        (0, 0, 1, 0.2),
        (0.5, 0.5, 1, 1),
    )
    with pytest.raises(KeyError):
        Settings.from_env({})


@pytest.fixture
def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    sessions, evidence, queue = tmp_path / "sessions", tmp_path / "evidence", tmp_path / "queue"
    sessions.mkdir()
    write_video(cached_clip("take_two").frames, sessions / "take_two.mp4")
    write_video(cached_clip("take_nothing").frames, sessions / "nothing.mp4")
    settings = Settings("f1", "https://backend.test", SECRET, sessions, evidence, queue, 30, "tok")
    received: list[dict] = []  # type: ignore[type-arg]
    up = {"ok": True}

    def handler(req: httpx.Request) -> httpx.Response:
        if not up["ok"]:
            raise httpx.ConnectError("offline")
        assert verify(
            SECRET, req.content, req.headers[TIMESTAMP_HEADER], req.headers[SIGNATURE_HEADER]
        )
        received.append(json.loads(req.content))
        return httpx.Response(200)

    disp = Dispatcher(backend(handler), Outbox(queue))
    app = create_app(settings, build_demo_analyzer(), disp, FaceBlurrer())
    return TestClient(app), settings, received, up, disp


BODY = {"fridge_id": "f1", "started_at": T0.isoformat(), "ended_at": T1.isoformat()}


def test_app_session_flow_idempotency_and_metrics(env) -> None:  # type: ignore[no-untyped-def]
    client, settings, received, up, disp = env
    r = client.post("/session", json={**BODY, "session_id": "s-two", "video": "take_two.mp4"})
    assert r.status_code == 200, r.text
    payload = r.json()["payload"]
    truth = {c.sku: c.qty for c in cached_clip("take_two").truth.cart}
    assert {i["sku"]: i["qty"] for i in payload["items"]} == truth and not payload["needs_review"]
    assert payload["evidence_ref"] == "s-two/clip.mp4" and len(received) == 1
    assert received[0]["session_id"] == "s-two"
    r2 = client.post("/session", json={**BODY, "session_id": "s-two", "video": "take_two.mp4"})
    assert r2.json()["duplicate"] is True and len(received) == 1  # no double processing/sending
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["outbox_depth"] == 0
    m = client.get("/metrics").text
    assert "hello_vision_sessions_total 1\n" in m and "hello_vision_outbox_depth 0\n" in m


def test_app_offline_queue_then_flush(env) -> None:  # type: ignore[no-untyped-def]
    client, _, received, up, disp = env
    up["ok"] = False
    r = client.post("/session", json={**BODY, "session_id": "s-off", "video": "nothing.mp4"})
    assert (
        r.json()["delivery"]["retry_later"] == 1
        and client.get("/health").json()["outbox_depth"] == 1
    )
    up["ok"] = True
    item = disp.outbox.due(1e12)[0]
    item.next_attempt_at = 0
    disp.outbox.update(item)
    assert disp.flush().sent == 1 and received[0]["session_id"] == "s-off"


def test_app_rejects_bad_input(env) -> None:  # type: ignore[no-untyped-def]
    client = env[0]
    for video in ("../../etc/passwd", "missing.mp4", "/etc/passwd"):
        r = client.post("/session", json={**BODY, "session_id": "x", "video": video})
        assert r.status_code == 400
    assert (
        client.post("/session", json={**BODY, "session_id": "../x", "video": "a"}).status_code
        == 422
    )


def test_review_ui_auth_pages_and_correction(env) -> None:  # type: ignore[no-untyped-def]
    client, settings, *_ = env
    client.post("/session", json={**BODY, "session_id": "s-two", "video": "take_two.mp4"})
    assert client.get("/review").status_code == 401
    assert client.get("/review?token=wrong").status_code == 401
    ok = client.get("/review?token=tok")
    assert ok.status_code == 200 and "s-two" in ok.text
    page = client.get("/review/s-two?token=tok")
    assert page.status_code == 200 and "Correct the cart" in page.text
    assert client.get("/review/nope?token=tok").status_code == 404
    jpg = client.get("/review/s-two/frame/5.jpg", headers={"Authorization": "Bearer tok"})
    assert jpg.status_code == 200 and jpg.headers["content-type"] == "image/jpeg"
    assert client.get("/review/s-two/frame/9999.jpg?token=tok").status_code == 404
    c = client.post(
        "/review/s-two/correction?token=tok",
        json={"items": [{"sku": "demo_blue_can", "qty": 2}], "note": "n"},
    )
    assert c.status_code == 200 and list_to_label(settings.evidence_root) == ["s-two"]
    assert "demo_blue_can" in client.get("/review/s-two?token=tok").text
    bad = client.post(
        "/review/s-two/correction?token=tok", json={"items": [{"sku": "a b", "qty": 1}]}
    )
    assert bad.status_code == 422
    assert client.post("/review/s-two/correction", json={"items": []}).status_code == 401
    assert "hello_vision_corrections_total 1\n" in client.get("/metrics").text


def test_review_ui_escapes_html() -> None:
    evil = "<script>alert(1)</script>"
    summary = {
        "session_id": "s1", "num_frames": 3, "reasons": [evil], "model_version": evil,
        "session_confidence": 0.5, "needs_review": True,
        "items": [{"sku": evil, "qty": 1}],
        "events": [{"action": evil, "sku": evil, "frames": [0, 1], "confidence": 0.1}],
    }  # fmt: skip
    page = session_page(summary, {"items": [{"sku": evil, "qty": 1}]}, token="</script><b>")  # noqa: S106
    assert "<script>alert" not in page and "</script><b>" not in page
    lst = list_page(
        [{"session_id": "s1", "needs_review": False, "items": [{"sku": evil, "qty": 1}]}]
    )
    assert "<script>alert" not in lst


def test_open_review_when_no_token_configured(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, settings, _, _, disp = env
    open_settings = Settings(
        "f", "u", "s", settings.sessions_root, settings.evidence_root, settings.queue_root, 30, None
    )
    c = TestClient(create_app(open_settings, build_demo_analyzer(), disp, FaceBlurrer()))
    assert c.get("/review").status_code == 200
