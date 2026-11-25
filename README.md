# hello-vision

> **Status: research prototype. Not in production.** No real customer is charged by this
> module. Anything described below as "planned" does not exist yet.

An open, in-house product-recognition module for smart vending fridges: it watches one
shopping session (door open to door close), works out which products were taken or put back,
and produces a cart for billing. Items it cannot decide on are flagged `needs_review`
instead of being charged as a guess.

Built by Osonvend. Licensed under Apache-2.0. Real data (footage, customers, sales) is never
stored in this repository; see [docs/PRIVACY.md](docs/PRIVACY.md).

## Current state (milestone M0)

Repository skeleton, CI, pre-commit guards and design docs only. No recognition code yet.

| Doc | Purpose |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline design |
| [docs/adr/](docs/adr/) | Architecture decision records |
| [docs/API.md](docs/API.md) | Backend contract (draft) |
| [docs/LICENSES.md](docs/LICENSES.md) | Model / dataset / dependency licenses |
| [docs/PRIVACY.md](docs/PRIVACY.md) | Privacy design (needs lawyer review) |
| [docs/HARDWARE_QUESTIONS.md](docs/HARDWARE_QUESTIONS.md) | Open hardware questions |
| [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) | Capture protocol (draft) |
| [docs/EDGE.md](docs/EDGE.md) | Edge hardware plan |

## Developer quickstart

```
uv sync --group dev
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run mypy
```

The 5-minute simulator quickstart on a synthetic demo video arrives with M1/M3.

## Roadmap

M0 skeleton, M1 catalog + synthetic data + capture + replay, M2 detection + identification,
M3 tracking + TAKE/RETURN + fusion, M4 evaluation + shadow mode, M5 service + review UI +
face blur, M6 edge export + benchmarks, M7 single-fridge shadow pilot plan.
