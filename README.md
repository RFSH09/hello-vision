# hello-vision

> **Status: research prototype. Not in production.** No real customer is charged by this
> module. Anything described below as "planned" does not exist yet.

An open, in-house product-recognition module for smart vending fridges: it watches one
shopping session (door open to door close), works out which products were taken or put back,
and produces a cart for billing. Items it cannot decide on are flagged `needs_review`
instead of being charged as a guess.

Built by Osonvend. Licensed under Apache-2.0. Real data (footage, customers, sales) is never
stored in this repository; see [docs/PRIVACY.md](docs/PRIVACY.md).

## Current state (milestones M0-M7 built; research prototype)

Built and tested **on synthetic data only**:

- capture adapters, SKU catalog and gallery, packshot import, synthetic scene/session generator
- detection and identification interfaces; classical stand-in models for the synthetic demo
- tracking, TAKE/RETURN logic, shelf-diff cross-check, fusion into a cart with a `needs_review` decision
- evaluation harness (reports, proposed go/no-go gate), shadow-mode comparison, weak labels, CVAT/Label Studio exchange
- FastAPI service, signed idempotent backend client with an offline queue, face blurring, review UI
- edge tooling: benchmark, export command plans, model store with rollback, Docker and systemd files

**Not true yet:** no learned model has been trained, nothing was tested on real footage or on any edge board, the live camera and
door-sensor path is not wired, and the Docker image and systemd unit were never run. The synthetic results (exact cart on every
scripted scenario) only show that the plumbing works. See [docs/KNOWN_FLAWS.md](docs/KNOWN_FLAWS.md) for the honest list.

| Doc | Purpose |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline design |
| [docs/KNOWN_FLAWS.md](docs/KNOWN_FLAWS.md) | What is missing or weak |
| [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md), [docs/GO_NO_GO.md](docs/GO_NO_GO.md) | One-fridge shadow pilot and go/no-go checklist |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metrics, reports, shadow mode, gate |
| [docs/API.md](docs/API.md) | Backend contract, signing, offline behaviour |
| [docs/PRIVACY.md](docs/PRIVACY.md) | Privacy design (needs lawyer review) |
| [docs/EDGE.md](docs/EDGE.md) | Edge hardware, benchmarks, operations |
| [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md), [docs/DVC.md](docs/DVC.md) | Capture protocol and data versioning |
| [docs/LICENSES.md](docs/LICENSES.md) | Model, dataset and dependency licenses |
| [docs/HARDWARE_QUESTIONS.md](docs/HARDWARE_QUESTIONS.md) | Open hardware questions |
| [docs/adr/](docs/adr/) | Architecture decision records |

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
uv run python -m sim.make_demo --help                # scenarios: take_one, take_two, swap, ...
uv run python -m eval.run --synthetic --out demo_out/eval   # evaluation report on synthetic sessions
```

`demo_out/take_two.truth.json` is the ground truth; the replay output should match its cart.
Other tools (all write outside the repo when pointed at private folders):

```
uv run python tools/gen_synthetic.py --out /private/synth --count 500   # COCO detection set
uv run python tools/import_packshots.py --catalog catalog.yaml --src packshots/ --gallery /private/gallery
```

## Roadmap

M0-M7 are built (see the `docs/M*_summary.md` files). Next: fix the flaws listed in docs/KNOWN_FLAWS.md, then collect
real data, train the learned models, and run the one-fridge shadow pilot.
