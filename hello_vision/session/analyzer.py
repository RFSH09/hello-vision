"""Session analysis: frames -> detections -> identity -> tracks -> events -> cart.

Two ways to use it:

* ``analyzer.analyze(session)`` for a finished recording (replay, evaluation);
* ``live = analyzer.begin(session_id)``, ``live.push(frame)`` for every frame while the door is
  open, then ``live.finish()`` at door close. Almost all the work happens in ``push``, so the
  time from door close to result is only the final decision (events, shelf diff, fusion).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from hello_vision.capture.types import Frame, Image, Session
from hello_vision.detect.types import Detection, Detector
from hello_vision.identify.embedders import Embedder
from hello_vision.identify.index import GalleryIndex, Match, crop
from hello_vision.session.events import ZoneConfig, derive_events
from hello_vision.session.fusion import FusionConfig, fuse
from hello_vision.session.shelf import shelf_diff
from hello_vision.session.types import AnnotatedDetection, SessionResult
from hello_vision.track.tracker import Track, Tracker


@dataclass(frozen=True)
class AnalyzerConfig:
    """Tunable parameters of :class:`SessionAnalyzer`.

    ``identify_every``: a product track is re-identified only every N-th observation (the
    previous identification is reused in between), which cuts embedding cost by about N.
    """

    shelf_zone_y_max: float = 0.7
    min_hits: int = 4
    min_move_px: float = 80.0
    snapshot_frames: int = 5
    identify_every: int = 3
    fusion: FusionConfig = field(default_factory=FusionConfig)
    model_version: str = "baseline-0"


class _CameraRun:
    """Incremental analysis of one camera's frames."""

    def __init__(self, analyzer: SessionAnalyzer, first_image: Image) -> None:
        self.a = analyzer
        h, w = first_image.shape[:2]
        c = analyzer.config
        self.zones = ZoneConfig(w, h, c.shelf_zone_y_max, c.min_move_px, min_hits=c.min_hits)
        # a private (shallow) copy so per-session detector state, e.g. a calibrated background,
        # cannot leak between cameras that are analysed at the same time
        self.detector: Detector = copy.copy(analyzer.detector)
        self.detector.start_session(first_image)
        self.products = Tracker("product")
        self.hands = Tracker("hand")
        self.frames: list[list[AnnotatedDetection]] = []

    def push(self, image: Image) -> None:
        index = len(self.frames)
        dets = self.detector.detect(image)
        cfg = self.a.config

        def identify(det: Detection, track: Track | None) -> Match | None:
            if det.label != "product":
                return None
            if track is not None and track.hits % cfg.identify_every != 0:
                last = track.obs[-1].match
                if last is not None:
                    return last  # reuse the previous identification of this track
            crop_img = crop(image, (det.x1, det.y1, det.x2, det.y2))
            return self.a.index.query(self.a.embedder.embed(crop_img))

        prod = self.products.update(index, dets, identify)
        hand = self.hands.update(index, dets)
        annotated: list[AnnotatedDetection] = []
        for i, d in enumerate(dets):
            track = prod.get(i) or hand.get(i)
            match = track.obs[-1].match if track is not None else None
            annotated.append(
                AnnotatedDetection(d, track.id if track else -1, match.sku if match else None)
            )
        self.frames.append(annotated)

    def finish(self, session_id: str) -> SessionResult:
        cfg = self.a.config
        events = derive_events(self.products.tracks, self.hands.tracks, self.zones)
        diff = shelf_diff(self.frames, self.zones, cfg.snapshot_frames)
        lines, conf, review, reasons = fuse(events, diff, cfg.fusion)
        return SessionResult(
            session_id, lines, conf, review, reasons, events, diff or {}, cfg.model_version,
            self.frames,
        )  # fmt: skip


class LiveSession:
    """Feed frames as they arrive; call :meth:`finish` when the door closes."""

    def __init__(self, analyzer: SessionAnalyzer, session_id: str) -> None:
        self._a = analyzer
        self.session_id = session_id
        self._runs: dict[str, _CameraRun] = {}

    def push(self, frame: Frame) -> None:
        """Process one frame of any camera (cameras may be interleaved)."""
        run = self._runs.get(frame.camera_id)
        if run is None:
            run = self._runs[frame.camera_id] = _CameraRun(self._a, frame.image)
        run.push(frame.image)

    def finish(self) -> SessionResult:
        """Make the final decision. Cameras must agree on the cart, otherwise it is flagged."""
        cfg = self._a.config
        if not self._runs:
            return SessionResult(
                self.session_id, [], 0.0, True, ["session has no frames"], [], {}, cfg.model_version
            )
        results = [r.finish(self.session_id) for _, r in sorted(self._runs.items())]
        first = results[0]
        if len(results) > 1:
            carts = {tuple(sorted(r.cart_dict().items())) for r in results}
            if len(carts) > 1:
                first.reasons.append("cameras disagree on the cart")
            first.needs_review = first.needs_review or any(r.needs_review for r in results)
            first.needs_review = first.needs_review or len(carts) > 1
            first.session_confidence = min(r.session_confidence for r in results)
        return first


class SessionAnalyzer:
    """Runs the whole recognition pipeline.

    With several cameras their carts must agree exactly; otherwise the session is flagged for
    review (proper multi-camera fusion needs the camera geometry, still an open hardware
    question).

    Args:
        detector: Product/hand detector (copied per camera for its per-session state).
        embedder: Crop embedder.
        index: SKU gallery index.
        config: Parameters.
    """

    def __init__(
        self,
        detector: Detector,
        embedder: Embedder,
        index: GalleryIndex,
        config: AnalyzerConfig | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.index = index
        self.config = config or AnalyzerConfig()

    def begin(self, session_id: str) -> LiveSession:
        """Start an incremental session (see :class:`LiveSession`)."""
        return LiveSession(self, session_id)

    def analyze(self, session: Session) -> SessionResult:
        """Analyse a finished recording (frames are pushed in recorded order)."""
        live = self.begin(session.session_id)
        for f in session.frames:
            live.push(f)
        return live.finish()
