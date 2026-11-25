"""FastAPI service: ``/session``, ``/health``, ``/metrics`` and the review UI."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from hello_vision.privacy.blur import FaceBlurrer
from hello_vision.review.ui import list_page, login_page, session_page
from hello_vision.service.client import Dispatcher
from hello_vision.service.evidence import (
    enqueue_to_label,
    list_to_label,
    purge_expired,
    read_frame_jpeg,
    session_dir,
    write_evidence,
)
from hello_vision.service.models import SessionRequest, to_payload
from hello_vision.service.settings import Settings
from hello_vision.service.signing import SIGNATURE_HEADER, TIMESTAMP_HEADER, verify
from hello_vision.session.analyzer import SessionAnalyzer


class CorrectionItem(BaseModel):
    """One corrected cart line."""

    sku: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    qty: int = Field(ge=1, le=100)


class Correction(BaseModel):
    """Body of ``POST /review/{id}/correction``."""

    items: list[CorrectionItem]
    note: str = Field(default="", max_length=500)


class Login(BaseModel):
    """Body of ``POST /review/login``."""

    token: str


def create_app(
    settings: Settings,
    analyzer: SessionAnalyzer,
    dispatcher: Dispatcher,
    blurrer: FaceBlurrer,
) -> FastAPI:
    """Build the service. Everything with side effects is injected (tests use fakes)."""
    app = FastAPI(title="hello-vision", version="0.0.1")
    counters: dict[str, float] = {
        "sessions_total": 0,
        "sessions_needs_review_total": 0,
        "analysis_seconds_sum": 0.0,
        "analysis_seconds_count": 0,
        "corrections_total": 0,
    }

    cookie_name = "hv_review"

    def cookie_value(token: str) -> str:
        """Session cookie = HMAC of the token, so the token itself is never stored in a cookie."""
        msg = b"review-session:" + token.encode()
        return hmac.new(settings.hmac_secret.encode(), msg, hashlib.sha256).hexdigest()

    def auth(request: Request) -> None:
        token = settings.review_token
        if token is None:
            return
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        cookie = request.cookies.get(cookie_name, "")
        if hmac.compare_digest(bearer, token) or hmac.compare_digest(cookie, cookie_value(token)):
            return
        raise HTTPException(401, "not logged in: open /review/login")

    async def signed(request: Request) -> None:
        """Require the HMAC signature (same scheme as the backend contract) on /session."""
        body = await request.body()
        ok = verify(
            settings.hmac_secret,
            body,
            request.headers.get(TIMESTAMP_HEADER, ""),
            request.headers.get(SIGNATURE_HEADER, ""),
        )
        if not ok:
            raise HTTPException(401, "missing or invalid request signature")

    locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
    locks_guard = threading.Lock()

    def session_lock(session_id: str) -> threading.Lock:
        with locks_guard:
            return locks[session_id]

    def folder(session_id: str) -> Path:
        try:
            return session_dir(settings.evidence_root, session_id)
        except ValueError as exc:
            raise HTTPException(400, "invalid session id") from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": analyzer.config.model_version,
            "fridge_id": settings.fridge_id,
            "outbox_depth": len(dispatcher.outbox),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [f"hello_vision_{k} {v}" for k, v in counters.items()]
        lines.append(f"hello_vision_outbox_depth {len(dispatcher.outbox)}")
        lines.append(f"hello_vision_to_label_depth {len(list_to_label(settings.evidence_root))}")
        return "\n".join(lines) + "\n"

    @app.post("/session", dependencies=[Depends(signed)])
    def post_session(req: SessionRequest) -> dict[str, Any]:
        with session_lock(req.session_id):  # two requests for one session run one after the other
            return process_session(req)

    def process_session(req: SessionRequest) -> dict[str, Any]:
        from sim.replay import replay_video  # local import: the simulator is optional

        out = folder(req.session_id)
        stored = out / "payload.json"
        if stored.exists():  # idempotent: same session id -> same stored result
            payload = json.loads(stored.read_text(encoding="utf-8"))
            dispatcher.flush()
            return {"payload": payload, "duplicate": True}
        video = (settings.sessions_root / req.video).resolve()
        if not video.is_relative_to(settings.sessions_root.resolve()) or not video.is_file():
            raise HTTPException(400, "video must be an existing file under the sessions root")
        session = replay_video(video, session_id=req.session_id)
        t0 = time.perf_counter()
        result = analyzer.analyze(session)
        counters["analysis_seconds_sum"] += time.perf_counter() - t0
        counters["analysis_seconds_count"] += 1
        ref = write_evidence(session, result, settings.evidence_root, blurrer)
        payload = to_payload(result, req.fridge_id, req.started_at, req.ended_at, ref)
        stored.write_text(payload.model_dump_json(), encoding="utf-8")
        counters["sessions_total"] += 1
        if result.needs_review:
            counters["sessions_needs_review_total"] += 1
            enqueue_to_label(settings.evidence_root, req.session_id, "needs_review")
        delivery = dispatcher.submit(payload)
        purge_expired(settings.evidence_root, settings.retention_days)
        return {"payload": payload.model_dump(mode="json"), "delivery": delivery.__dict__}

    def summaries() -> list[dict[str, Any]]:
        root = settings.evidence_root
        files = sorted(root.glob("*/result.json")) if root.exists() else []
        return [json.loads(p.read_text(encoding="utf-8")) for p in files]

    @app.get("/review/login", response_class=HTMLResponse)
    def review_login_page() -> str:
        return login_page()

    @app.post("/review/login")
    def review_login(body: Login) -> Response:
        token = settings.review_token
        if token is None or not hmac.compare_digest(body.token, token):
            raise HTTPException(401, "wrong token")
        resp = JSONResponse({"status": "ok"})
        resp.set_cookie(
            cookie_name, cookie_value(token), httponly=True, samesite="strict", max_age=8 * 3600
        )
        return resp

    @app.get("/review", response_class=HTMLResponse, dependencies=[Depends(auth)])
    def review_list() -> str:
        return list_page(summaries())

    @app.get("/review/{session_id}", response_class=HTMLResponse, dependencies=[Depends(auth)])
    def review_session(session_id: str) -> str:
        d = folder(session_id)
        if not (d / "result.json").exists():
            raise HTTPException(404, "unknown session")
        summary = json.loads((d / "result.json").read_text(encoding="utf-8"))
        corr_path = d / "correction.json"
        corr = json.loads(corr_path.read_text(encoding="utf-8")) if corr_path.exists() else None
        return session_page(summary, corr)

    @app.get("/review/{session_id}/frame/{index}.jpg", dependencies=[Depends(auth)])
    def review_frame(session_id: str, index: int) -> Response:
        folder(session_id)
        jpeg = read_frame_jpeg(settings.evidence_root, session_id, index)
        if jpeg is None:
            raise HTTPException(404, "frame not found")
        return Response(jpeg, media_type="image/jpeg")

    @app.post("/review/{session_id}/correction", dependencies=[Depends(auth)])
    def review_correct(session_id: str, body: Correction) -> dict[str, str]:
        d = folder(session_id)
        if not (d / "result.json").exists():
            raise HTTPException(404, "unknown session")
        (d / "correction.json").write_text(body.model_dump_json(), encoding="utf-8")
        enqueue_to_label(settings.evidence_root, session_id, "human_correction")
        counters["corrections_total"] += 1
        return {"status": "saved"}

    return app
