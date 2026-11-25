"""Backend client: signed, idempotent, retried, offline-tolerant delivery of session results."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from hello_vision.service.models import CatalogResponse, SessionResultPayload
from hello_vision.service.outbox import Outbox
from hello_vision.service.signing import signed_headers

RESULT_PATH = "/api/vision/session-result"
CATALOG_PATH = "/api/vision/catalog"


@dataclass
class FlushReport:
    """What one :meth:`Dispatcher.flush` did."""

    sent: int = 0
    retry_later: int = 0
    rejected: int = 0


def backoff_delay(
    attempt: int, base: float = 2.0, cap: float = 300.0, jitter: float = 0.1
) -> float:
    """Exponential backoff with jitter: ``base * 2**attempt`` capped at ``cap``."""
    d = float(min(cap, base * (2**attempt)))
    return d * (1 + random.uniform(-jitter, jitter))  # noqa: S311 - jitter, not security


class BackendClient:
    """Thin HTTP client for the Django backend.

    Args:
        base_url: Backend base URL (from configuration).
        secret: HMAC secret (from the environment).
        http: Injected ``httpx.Client`` (tests use a mock transport).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        secret: str,
        http: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.secret = secret
        self.http = http or httpx.Client(timeout=timeout)

    def post_result(self, payload: SessionResultPayload) -> str:
        """Send one result. Returns ``"sent"``, ``"retry"`` (transient) or ``"rejected"``.

        2xx and 409 (the backend already has this ``session_id``) count as sent. Other 4xx are
        permanent rejections; 5xx, 429 and network errors are retried later.
        """
        body = payload.model_dump_json().encode()
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": payload.session_id,
            **signed_headers(self.secret, body),
        }
        try:
            r = self.http.post(self.base_url + RESULT_PATH, content=body, headers=headers)
        except httpx.HTTPError:
            return "retry"
        if r.status_code < 300 or r.status_code == 409:
            return "sent"
        if r.status_code == 429 or r.status_code >= 500:
            return "retry"
        return "rejected"

    def fetch_catalog(self) -> CatalogResponse:
        """Fetch the SKU catalog and planograms (signed GET)."""
        headers = signed_headers(self.secret, b"")
        r = self.http.get(self.base_url + CATALOG_PATH, headers=headers)
        r.raise_for_status()
        return CatalogResponse.model_validate(r.json())


class Dispatcher:
    """Queue-first delivery: every result is stored, then sent; failures retry with backoff."""

    def __init__(
        self,
        client: BackendClient,
        outbox: Outbox,
        clock: Callable[[], float] = time.time,
        max_attempts: int = 50,
    ) -> None:
        self.client = client
        self.outbox = outbox
        self.clock = clock
        self.max_attempts = max_attempts

    def submit(self, payload: SessionResultPayload) -> FlushReport:
        """Queue ``payload`` (idempotent by session id) and try to deliver everything due."""
        self.outbox.put(payload.session_id, payload.model_dump(mode="json"))
        return self.flush()

    def flush(self) -> FlushReport:
        """Try every queued item that is due; reschedule failures with exponential backoff."""
        report = FlushReport()
        for item in self.outbox.due(self.clock()):
            outcome = self.client.post_result(SessionResultPayload.model_validate(item.payload))
            if outcome == "sent":
                self.outbox.done(item.session_id)
                report.sent += 1
            elif outcome == "rejected" or item.attempts + 1 >= self.max_attempts:
                self.outbox.kill(item.session_id)
                report.rejected += 1
            else:
                item.attempts += 1
                item.next_attempt_at = self.clock() + backoff_delay(item.attempts)
                self.outbox.update(item)
                report.retry_later += 1
        return report
