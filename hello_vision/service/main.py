"""Service entry point: ``uv run python -m hello_vision.service.main`` (reads VISION_* env vars)."""

from __future__ import annotations

import importlib
import os

import uvicorn

from hello_vision.privacy.blur import FaceBlurrer, YuNetFaceDetector
from hello_vision.service.app import create_app
from hello_vision.service.client import BackendClient, Dispatcher
from hello_vision.service.outbox import Outbox
from hello_vision.service.settings import Settings
from hello_vision.session.analyzer import SessionAnalyzer


def build_app() -> object:
    """Assemble the production app from the environment."""
    settings = Settings.from_env()
    module, _, func = os.environ.get(
        "VISION_ANALYZER", "sim.pipeline:build_demo_analyzer"
    ).partition(":")
    analyzer: SessionAnalyzer = getattr(importlib.import_module(module), func)()
    dispatcher = Dispatcher(
        BackendClient(settings.backend_url, settings.hmac_secret), Outbox(settings.queue_root)
    )
    if settings.face_model is None:
        raise RuntimeError("VISION_FACE_MODEL is required: refusing to store unblurred evidence")
    blurrer = FaceBlurrer(YuNetFaceDetector(settings.face_model), settings.blur_regions)
    return create_app(settings, analyzer, dispatcher, blurrer)


def main() -> int:
    """Run the service with uvicorn on localhost (put a reverse proxy in front if needed)."""
    uvicorn.run(build_app(), host="127.0.0.1", port=int(os.environ.get("VISION_PORT", "8080")))  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
