import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "hooks"))

from hello_vision.catalog.packshots import placeholder_sprites
from hello_vision.synth.session import SessionClip, render_session


@cache
def cached_clip(scenario: str, seed: int = 0) -> SessionClip:
    """Render a demo session once per test run (rendering is the slow part)."""
    return render_session(scenario, placeholder_sprites(), seed=seed)
