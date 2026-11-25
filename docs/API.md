# Backend contract

Implemented in `hello_vision/service/models.py` (pydantic), `signing.py`, `client.py`, `outbox.py`; contract tests in
`tests/test_service.py`.

## POST /api/vision/session-result  (vision module -> Django)

```json
{"fridge_id": "string", "session_id": "string",
 "started_at": "ISO-8601", "ended_at": "ISO-8601",
 "items": [{"sku": "string", "qty": 1, "confidence": 0.0}],
 "session_confidence": 0.0, "needs_review": false,
 "evidence_ref": "string", "model_version": "string"}
```

- **`needs_review: true` means DO NOT CHARGE.** `items` is then only a proposal for the human reviewer.
- **Idempotent by `session_id`.** Header `Idempotency-Key: <session_id>`. The backend must answer 2xx or 409 for a session it
  already has; the module treats both as delivered, so retries can never double-charge.
- **Signed with HMAC-SHA256.** Headers `X-Vision-Timestamp` (unix seconds) and `X-Vision-Signature` =
  `hex(HMAC_SHA256(secret, f"{timestamp}." + raw_body))`. The secret comes from the environment on both sides.
  The backend must reject timestamps more than 300 s old. Reference verifier (copy into Django):

```python
import hashlib, hmac, time


def verify(secret: str, body: bytes, timestamp: str, signature: str, max_skew_s: int = 300) -> bool:
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > max_skew_s:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

- **Retries and offline:** every result is written to a local queue first, then sent. 5xx, 429 and network errors are
  retried with exponential backoff (2 s doubling, capped at 300 s, +-10% jitter); other 4xx responses are permanent
  rejections moved to `queue/dead/` for inspection. Results survive restarts and internet outages.

## GET /api/vision/catalog  (Django -> vision module, signed GET with an empty body)

```json
{"skus": [{"id": "string", "name": "string", "group": null}],
 "planograms": {"<fridge_id>": {"<shelf_id>": ["<sku id>", "..."]}}}
```

## Local service (runs on the fridge computer, bound to 127.0.0.1)

| Endpoint | Purpose |
| --- | --- |
| `POST /session` | analyse a recorded session video under the sessions root; idempotent by `session_id`; **requires the same HMAC signature headers as the backend contract** (stale timestamps rejected); requests for one session id are serialised |
| `GET /health` | status, model version, queue depth |
| `GET /metrics` | Prometheus text format |
| `GET /review/login`, `POST /review/login` | exchange the review token for an HttpOnly, SameSite=Strict cookie (derived from the token, never the token itself); `Authorization: Bearer` also works; tokens in URLs are NOT accepted |
| `GET /review`, `/review/{id}` | review UI (needs login when `VISION_REVIEW_TOKEN` is set) |
| `POST /review/{id}/correction` | store a human-corrected cart and queue the session for labelling |

Settings are environment variables, see `.env.example`.

## Not decided yet

How the backend should treat `needs_review` sessions operationally (hold, notify, charge after review) is the business
and Django team's decision; this module only guarantees it never marks an uncertain session as billable.
