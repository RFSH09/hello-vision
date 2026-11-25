"""Wiring of the demo pipeline (synthetic baselines). Real deployments build their own."""

from __future__ import annotations

from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.detect.baseline import BaselineDetector
from hello_vision.identify.embedders import ColorLayoutEmbedder
from hello_vision.identify.index import build_index
from hello_vision.session.analyzer import AnalyzerConfig, SessionAnalyzer


def build_demo_analyzer(threshold: float = 0.9) -> SessionAnalyzer:
    """Analyzer using the synthetic baseline detector/embedder and the placeholder gallery."""
    embedder = ColorLayoutEmbedder()
    index = build_index({k: [v] for k, v in placeholder_sprites().items()}, embedder, threshold)
    return SessionAnalyzer(
        BaselineDetector(), embedder, index, AnalyzerConfig(model_version="demo-baseline-0")
    )
