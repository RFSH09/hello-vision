# Backend contract (draft; pydantic models and contract tests land in M5)

## POST /api/vision/session-result  (to the Django backend)

```json
{"fridge_id": "string", "session_id": "string",
 "started_at": "ISO-8601", "ended_at": "ISO-8601",
 "items": [{"sku": "string", "qty": 1, "confidence": 0.0}],
 "session_confidence": 0.0, "needs_review": false,
 "evidence_ref": "string", "model_version": "string"}
```

- Idempotent by `session_id` (a repeated POST returns the same result, no double charge).
- Signed with HMAC (secret from the environment, never in the repo); signature in a header.
- Retries with exponential backoff; results queue locally and flush when the internet returns.

## GET /api/vision/catalog

SKUs (ids, names) and the current planogram per fridge.
