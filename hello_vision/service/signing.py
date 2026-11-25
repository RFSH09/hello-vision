"""HMAC request signing. The secret comes from the environment, never from the repository.

Signature = hex(HMAC-SHA256(secret, f"{timestamp}." + body)). Headers:
``X-Vision-Timestamp`` (unix seconds) and ``X-Vision-Signature``. The receiving backend must
use :func:`verify` (copy it into the Django project) and reject stale timestamps.
"""

from __future__ import annotations

import hashlib
import hmac
import time

TIMESTAMP_HEADER = "X-Vision-Timestamp"
SIGNATURE_HEADER = "X-Vision-Signature"


def sign(secret: str, body: bytes, timestamp: int) -> str:
    """Return the hex signature of ``body`` at ``timestamp``."""
    msg = f"{timestamp}.".encode() + body
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def signed_headers(secret: str, body: bytes, now: int | None = None) -> dict[str, str]:
    """Return the two signature headers for ``body``."""
    ts = int(time.time()) if now is None else now
    return {TIMESTAMP_HEADER: str(ts), SIGNATURE_HEADER: sign(secret, body, ts)}


def verify(
    secret: str,
    body: bytes,
    timestamp: str,
    signature: str,
    max_skew_s: int = 300,
    now: int | None = None,
) -> bool:
    """Check a signature in constant time and reject timestamps older/newer than ``max_skew_s``."""
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - ts) > max_skew_s:
        return False
    return hmac.compare_digest(sign(secret, body, ts), signature)
