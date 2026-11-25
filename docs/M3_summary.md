# M3 summary

Pipeline (see ARCHITECTURE.md): `SessionAnalyzer` runs detection, identification, tracking, TAKE/RETURN, shelf diff and fusion.

- **Tracking:** greedy IoU + centre-distance tracker (`track/tracker.py`); hand-product association (`track/association.py`).
- **Actions:** zone logic (`session/events.py`): shelf-origin to door-side = TAKE, door-origin to shelf = RETURN; stationary, noise and shown-and-removed give no event.
- **Shelf diff:** median per-SKU counts over hand-free frames before/after (`session/shelf.py`).
- **Fusion:** cart from events; `needs_review` on unknown product, motion without hand, returned-but-not-taken, shelf/event disagreement, missing snapshots, low confidence, or camera disagreement (`session/fusion.py`).
- **Result on synthetic scenarios:** exact cart, no review, 12/12 (6 scenarios x 2 seeds); coverage of session/ and identify/ is above 90% (CI gate 80%).
- **A rendering issue found while building:** when the held product was drawn *under* the hand it was almost invisible and the tracker missed every take. The pipeline correctly failed safe (all flagged for review), which is the intended behaviour. Real hands will occlude products, so this is a known risk for real data.
- **Open risks:** thresholds are tuned on synthetic data only; the tracker is simple; multi-camera fusion only checks agreement; real occlusion, fast hands, two hands and similar-looking SKUs are untested.
