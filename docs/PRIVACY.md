# Privacy design

> **NEEDS LAWYER REVIEW.** The personal-data law of Uzbekistan is understood to require that
> citizens' personal data be stored locally. This document is engineering intent, not legal
> advice.

- Real footage, phone numbers and purchase data never enter this repository (enforced by
  `.gitignore` and pre-commit hooks); they live in private storage referenced by config.
- Faces are blurred in every stored frame or clip before it is written (M5).
- Keep only what is needed; retention is configurable and expired evidence is deleted.
- All storage locations and endpoints are configuration, so the system can run fully inside
  Uzbekistan. Team-recorded training data requires consent.
- The module needs no customer identity; only `session_id` and fridge id are used.
