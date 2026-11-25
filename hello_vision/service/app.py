"""FastAPI service: ``/session``, ``/health``, ``/metrics`` and the review UI."""

from __future__ import annotations

import hmac
import json
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from hello_vision.privacy.blur import FaceBlurrer
from hello_vision.review.ui import list_page, session_page
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
from hello_vision.session.analyzer import SessionAnalyzer


class CorrectionItem(BaseModel):
    """One corrected cart line."""

    sku: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    qty: int = Field(ge=1, le=100)


class Correction(BaseModel):
    """Body of ``POST /review/{id}/correction``."""

    items: list[CorrectionItem]
    note: str = Field(default="", max_length=500)


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

    def auth(request: Request) -> None:
        token = settings.review_token
        if token is None:
            return
        given = request.query_params.get("token") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        if not hmac.compare_digest(given, token):
            raise HTTPException(401, "invalid or missing review token")

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

    @app.post("/session")
    def post_session(req: SessionRequest) -> dict[str, Any]:
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

    @app.get("/review", response_class=HTMLResponse, dependencies=[Depends(auth)])
    def review_list(request: Request) -> str:
        t = request.query_params.get("token")
        return list_page(summaries(), f"?token={t}" if t else "")

    @app.get("/review/{session_id}", response_class=HTMLResponse, dependencies=[Depends(auth)])
    def review_session(session_id: str, request: Request) -> str:
        d = folder(session_id)
        if not (d / "result.json").exists():
            raise HTTPException(404, "unknown session")
        summary = json.loads((d / "result.json").read_text(encoding="utf-8"))
        corr_path = d / "correction.json"
        corr = json.loads(corr_path.read_text(encoding="utf-8")) if corr_path.exists() else None
        return session_page(summary, corr, request.query_params.get("token", ""))

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
