"""Hand-product association: was a hand near the product while it moved?"""

from __future__ import annotations

from hello_vision.detect.types import Detection
from hello_vision.track.tracker import Track


def _near(a: Detection, b: Detection, margin: float) -> bool:
    return not (
        a.x2 + margin < b.x1 or b.x2 + margin < a.x1 or a.y2 + margin < b.y1 or b.y2 + margin < a.y1
    )


def hand_associated(product: Track, hands: list[Track], margin: float = 40.0) -> bool:
    """Return True if any hand box came within ``margin`` px of the product during its life.

    A product that changes place with no hand nearby is "unexplained motion" (it fell, the
    detector glitched, ...), which the session logic sends to review.
    """
    frames = {o.frame: o.det for o in product.obs}
    for hand in hands:
        for o in hand.obs:
            det = frames.get(o.frame)
            if det is not None and _near(det, o.det, margin):
                return True
    return False
