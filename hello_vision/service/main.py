"""Service entry point: ``uv run python -m hello_vision.service.main`` (reads VISION_* env vars)."""

from __future__ import annotations

import importlib
import logging
import os
import threading

import uvicorn
from fastapi import FastAPI

from hello_vision.logging_config import configure
from hello_vision.privacy.blur import FaceBlurrer, YuNetFaceDetector
from hello_vision.service.app import create_app
from hello_vision.service.client import BackendClient, Dispatcher
from hello_vision.service.outbox import Outbox
from hello_vision.service.settings import Settings
from hello_vision.service.watchdog import notify, start_watchdog
from hello_vision.session.analyzer import SessionAnalyzer

log = logging.getLogger("hello_vision")


def build_app() -> FastAPI:
    """Assemble the production app from the environment."""
    settings = Settings.from_env()
    spec = os.environ.get("VISION_ANALYZER", "sim.pipeline:build_demo_analyzer")
    module, _, func = spec.partition(":")
    analyzer: SessionAnalyzer = getattr(importlib.import_module(module), func)()
    dispatcher = Dispatcher(
        BackendClient(settings.backend_url, settings.hmac_secret), Outbox(settings.queue_root)
    )
    if settings.face_model is None:
        raise RuntimeError("VISION_FACE_MODEL is required: refusing to store unblurred evidence")
    blurrer = FaceBlurrer(YuNetFaceDetector(settings.face_model), settings.blur_regions)
    log.info("service built", extra={"model_version": analyzer.config.model_version})
    return create_app(settings, analyzer, dispatcher, blurrer)


def main() -> int:
    """Run the service with uvicorn on localhost (put a reverse proxy in front if needed)."""
    configure()
    app = build_app()
    stop = threading.Event()
    start_watchdog(lambda: True, stop)
    notify("READY=1")  # sent just before serving; uvicorn binds immediately after
    try:
        uvicorn.run(
            app, host="127.0.0.1", port=int(os.environ.get("VISION_PORT", "8080")), log_config=None
        )
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
