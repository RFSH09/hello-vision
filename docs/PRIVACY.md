# Privacy design

> **NEEDS LAWYER REVIEW.** The personal-data law of Uzbekistan is understood to require that citizens' personal data be
> stored locally. This document is engineering intent, not legal advice.

## What the code does today

- **No real data in the repository.** `.gitignore` plus the pre-commit hooks block media, model weights, `.env`, keys
  and anything under `data/`. Real footage, phone numbers and purchase data live in private storage referenced by config.
- **No customer identity.** The module only sees `session_id` and `fridge_id`; it never receives phone numbers or card data.
- **Raw frames are never written.** Evidence clips are written only after faces are blurred (`privacy/blur.py`) and are
  overlaid with detections afterwards. Blur is pixelation followed by smoothing, so it is not reversible.
- **Two blur layers.** A face detector (YuNet, supplied as a separate model file) and optional fixed regions that are
  always blurred (e.g. the area in front of the door), which do not depend on detection.
- **Fail closed.** The service refuses to start without a face model (`VISION_FACE_MODEL`).
- **Configurable retention.** Evidence older than `VISION_RETENTION_DAYS` (default 30) is deleted; the labelling queue
  survives purges but only holds session ids and reasons.
- **Local storage.** Every storage path and endpoint is configuration (`.env.example`); nothing is sent anywhere except
  the signed result to the configured backend.
- **Review UI** is bound to localhost by default and protected by a token when `VISION_REVIEW_TOKEN` is set; it shows
  blurred frames only.

## Known gaps (to close before storing real footage)

- The face detector and the blur have **not** been validated on real fridge footage. Validate on consented test footage
  and measure the miss rate before relying on it.
- The 24/7 security camera stream is not processed by this module; its handling (and its own retention) is outside this code.
- Training data from team recordings requires written consent and its own retention rules.
- Backups, access control to the evidence folder and audit logging are operational tasks not yet defined.
