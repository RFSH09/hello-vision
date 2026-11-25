"""End-to-end session analysis: frames -> detections -> identity -> tracks -> events -> cart."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from hello_vision.capture.types import Frame, Session
from hello_vision.detect.types import Detector
from hello_vision.identify.embedders import Embedder
from hello_vision.identify.index import GalleryIndex, Match, crop
from hello_vision.session.events import ZoneConfig, derive_events
from hello_vision.session.fusion import FusionConfig, fuse
from hello_vision.session.shelf import shelf_diff
from hello_vision.session.types import AnnotatedDetection, SessionResult
from hello_vision.track.tracker import Tracker


@dataclass(frozen=True)
class AnalyzerConfig:
    """Tunable parameters of :class:`SessionAnalyzer`."""

    shelf_zone_y_max: float = 0.7
    min_hits: int = 4
    min_move_px: float = 80.0
    snapshot_frames: int = 5
    fusion: FusionConfig = field(default_factory=FusionConfig)
    model_version: str = "baseline-0"


class SessionAnalyzer:
    """Runs the whole recognition pipeline on one :class:`Session`.

    Cameras are analysed independently. With several cameras, their carts must agree exactly;
    otherwise the session is flagged for review (proper multi-camera fusion needs the camera
    geometry, which is still an open hardware question).

    Args:
        detector: Product/hand detector.
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

    def analyze(self, session: Session) -> SessionResult:
        """Analyse ``session`` and return the cart and review decision."""
        by_cam: dict[str, list[Frame]] = defaultdict(list)
        for f in session.frames:
            by_cam[f.camera_id].append(f)
        if not by_cam:
            return SessionResult(
                session.session_id, [], 0.0, True, ["session has no frames"], [], {},
                self.config.model_version,
            )  # fmt: skip
        results = [self._analyze_camera(session.session_id, fr) for _, fr in sorted(by_cam.items())]
        first = results[0]
        if len(results) > 1:
            carts = {tuple(sorted(r.cart_dict().items())) for r in results}
            if len(carts) > 1:
                first.reasons.append("cameras disagree on the cart")
            first.needs_review = first.needs_review or any(r.needs_review for r in results)
            first.needs_review = first.needs_review or len(carts) > 1
            first.session_confidence = min(r.session_confidence for r in results)
        return first

    def _analyze_camera(self, session_id: str, frames: list[Frame]) -> SessionResult:
        cfg = self.config
        h, w = frames[0].image.shape[:2]
        zones = ZoneConfig(w, h, cfg.shelf_zone_y_max, cfg.min_move_px, min_hits=cfg.min_hits)
        self.detector.start_session(frames[0].image)
        products = Tracker("product")
        hands = Tracker("hand")
        per_frame: list[list[AnnotatedDetection]] = []
        for i, frame in enumerate(frames):
            dets = self.detector.detect(frame.image)
            matches: list[Match | None] = [
                self.index.query(self.embedder.embed(crop(frame.image, (d.x1, d.y1, d.x2, d.y2))))
                if d.label == "product"
                else None
                for d in dets
            ]
            pairs = list(zip(dets, matches, strict=True))
            products.update(i, pairs)
            hands.update(i, pairs)
            per_frame.append([AnnotatedDetection(d, -1, m.sku if m else None) for d, m in pairs])
        self._label_track_ids(per_frame, products, hands)

        events = derive_events(products.tracks, hands.tracks, zones)
        diff = shelf_diff(per_frame, zones, cfg.snapshot_frames)
        lines, conf, review, reasons = fuse(events, diff, cfg.fusion)
        return SessionResult(
            session_id, lines, conf, review, reasons, events, diff or {}, cfg.model_version,
            per_frame,
        )  # fmt: skip

    @staticmethod
    def _label_track_ids(
        per_frame: list[list[AnnotatedDetection]], products: Tracker, hands: Tracker
    ) -> None:
        """Fill in ``track_id`` of every annotated detection from the finished tracks."""
        for tr in [*products.tracks, *hands.tracks]:
            for o in tr.obs:
                for k, a in enumerate(per_frame[o.frame]):
                    if a.det is o.det:
                        per_frame[o.frame][k] = AnnotatedDetection(a.det, tr.id, a.sku)
