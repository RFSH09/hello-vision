# M4 summary

Evaluation harness (`eval/run.py`), metrics, report (Markdown + HTML + JSON), proposed gate checker, and shadow-mode comparison. See [EVALUATION.md](EVALUATION.md).

- Synthetic run: 18 sessions, exact 1.000, review 0.000, false charge 0.000. The gate shows NOT PASSED because latency was not measured on the target device and there are no 14 days of shadow data (as designed).
- Open: latency numbers on this PC are noisy (same scenario 22 s then 3.7 s on repeats); needs profiling on a quiet machine. Shadow format is my proposal; the supplier export format is still unknown.
