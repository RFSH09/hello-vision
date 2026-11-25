# hello-vision

> **Status: research prototype. Not in production.** No real customer is charged by this
> module. Anything described below as "planned" does not exist yet.

An open, in-house product-recognition module for smart vending fridges: it watches one
shopping session (door open to door close), works out which products were taken or put back,
and produces a cart for billing. Items it cannot decide on are flagged `needs_review`
instead of being charged as a guess.

Built by Osonvend. Licensed under Apache-2.0. Real data (footage, customers, sales) is never
stored in this repository; see [docs/PRIVACY.md](docs/PRIVACY.md).

## Current state (milestone M3)

An end-to-end pipeline runs on **synthetic** sessions: capture -> detection -> identification ->
tracking -> TAKE/RETURN logic -> shelf-diff cross-check -> fusion -> cart with a `needs_review`
decision. It gets the exact cart on all six synthetic scenarios (take one, take two,
take-and-return, take nothing, swap, low light).

**Important limits.** The detector and embedder used there are classical stand-ins that only work
on the drawn placeholder products. No learned model has been trained, nothing has been tested on
real footage, and the synthetic results say nothing about real-world accuracy. Live camera
capture and multi-camera fusion are untested on hardware.

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

## 5-minute quickstart (synthetic demo, no real footage)

```
uv sync --group dev
uv run python -m sim.make_demo --scenario take_two   # writes demo_out/take_two.mp4 + .truth.json
uv run python -m sim.replay demo_out/take_two.mp4    # prints the cart + review decision
uv run python -m sim.make_demo --help                # scenarios: take_one, take_two, ...
```

`demo_out/take_two.truth.json` is the ground truth; the replay output should match its cart.
Other tools (all write outside the repo when pointed at private folders):

```
uv run python tools/gen_synthetic.py --out /private/synth --count 500   # COCO detection set
uv run python tools/import_packshots.py --catalog catalog.yaml --src packshots/ --gallery /private/gallery
```

## Roadmap

M0 skeleton, M1 catalog + synthetic data + capture + replay, M2 detection + identification,
M3 tracking + TAKE/RETURN + fusion, M4 evaluation + shadow mode, M5 service + review UI +
face blur, M6 edge export + benchmarks, M7 single-fridge shadow pilot plan.
