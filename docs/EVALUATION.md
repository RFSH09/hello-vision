# Evaluation and shadow mode

## Gate before any customer is charged by our model (PROPOSED, awaiting owner confirmation)

On shadow data: exact cart accuracy >= 99% for 2 weeks, false-charge rate < 0.2%, review rate < 3%,
result < 5 s after door close on the target edge device. `eval/gate.py` encodes these; the latency row can
only pass if the measurement was taken on the target device, and the duration row needs 14 days of data.

## Metrics (`eval/metrics.py`)

- **Exact cart accuracy**, reported twice: over all sessions (proposal vs truth) and over auto-charged sessions
  only (those not flagged `needs_review`), since only those would be billed.
- **False-charge rate**: share of auto-charged sessions where something was charged beyond what was taken.
- **Review rate**, item precision/recall, missed items, TAKE/RETURN action errors, per-SKU confusion, latency p50/p95.

## Running

```
uv run python -m eval.run --synthetic --out demo_out/eval          # harness check on synthetic data
uv run python -m eval.run --sessions /private/sessions --out /private/reports \
    --analyzer mypackage.module:build_analyzer                      # <name>.mp4 + <name>.truth.json pairs
```

Outputs `report.md`, `report.html`, `results.json`. Failures list every non-exact session with its clip.

## Shadow mode (`eval/shadow.py`)

Run our module next to the current system on the same sessions, export both as JSON Lines
(`session_id`, `items[{sku,qty}]`, optional `needs_review`, `started_at`, `clip`), map the other system's SKU ids
with `--sku-map`, then:

```
uv run python -m eval.shadow --ours ours.jsonl --theirs theirs.jsonl --sku-map map.json --out report/
```

The report gives the agreement rate (over sessions we did not flag) and lists every disagreement with its kind
(`we_missed`, `we_added`, `swapped`, `quantity`) and clip reference. Note that disagreement is not the same as
error: the other system can be wrong too, so disagreements need human labels before they count as ours.

## Caveat

All numbers produced so far come from synthetic data with stand-in models. They test the tooling only.
