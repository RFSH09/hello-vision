# Known flaws and open work (to fix after the milestones)

Collected while building M0-M7. Ordered roughly by how much they block a real pilot.

## Blockers for any real use

1. **No learned models.** The detector and embedder used in the pipeline are classical stand-ins that only work on the
   drawn synthetic products. `OnnxDetector` / `OnnxEmbedder` are written but never run against a real model; the training
   scripts have never been executed; there is no detector trainer yet.
2. **No real data has been used anywhere.** All accuracy numbers (100% on synthetic scenarios) only show that the plumbing works.
3. **The colour embedder cannot separate look-alikes** (a gold bar scored 0.93 against a brown bar). Real look-alike SKUs are
   the main accuracy risk and need the fine-tuned embedding model plus hard-negative evaluation.
4. **No live pipeline.** Capture, door sensor, lock and the analyzer are not wired into a running daemon; `/session` analyses a
   recorded video. Door sensor and camera access depend on [HARDWARE_QUESTIONS.md](HARDWARE_QUESTIONS.md).
5. **Latency.** PARTLY FIXED: analysis is now streaming (frames are processed as they arrive, only the final decision runs after door close, about 0 s) and tracks are re-identified every 3rd frame. The new risk is keeping up in real time: the baseline uses 0.62x of the session time on a fast desktop, and a real model on a cheap board may exceed 1.0 (the gate has a row for it). Needs frame subsampling / a fast detector and measurement on the target board.

## Recognition logic

6. Thresholds (identification 0.9, min confidence 0.75, move distance, zone split) were tuned on synthetic data only.
7. The tracker is greedy IoU/distance matching. Synthetic stress scenarios now exist: **fast hand and two hands pass** (exact cart, no review); **heavy occlusion** (held product behind the hand) is NOT solved: the pipeline flags it for review instead of guessing. A safety-invariant test asserts that no scenario is ever auto-accepted with a wrong cart. Real occlusion, products crossing and real hand motion are untested. Hand-product association is a simple "near" test.
8. Shelf diff needs hand-free frames at both ends of the session; busy real sessions may not have them (they then go to review).
9. Multi-camera "fusion" only checks that per-camera carts agree; real fusion needs the camera geometry.
10. Zones are a single horizontal split (shelf above, door below); the real geometry depends on camera mounting.
11. Evidence clips use only the first camera.

## Service, privacy, operations

12. Face blur was never run on real faces; no fixed blur regions are configured by default.
13. ~~Review token in the URL~~ FIXED (cookie login, no tokens in URLs; `/session` now signed). Still open: the review page JavaScript was never tested in a browser.
14. ~~Outbox/`/session` races~~ FIXED within one process (exclusive file create, per-session lock, tested with threads). Still open: several service processes sharing one folder are not coordinated.
15. Docker image never built (no Docker here); systemd unit never run; the watchdog socket test is skipped on Windows;
    exports (TensorRT/OpenVINO/RKNN) never run; no board measured.
16. ~~Transitive licenses unaudited~~ DONE (see LICENSES.md): all permissive except `certifi` (MPL-2.0, runtime) - owner decision needed. Model/dataset licenses are still unverified.
17. OTA: only the local model store exists; how models are delivered and signed is not designed.

## Repo and process

18. Only M0 has a pull request; GitHub had an outage when merging it. M1-M7 branches are local and unpushed, and CI has never run
    on them (only locally: ruff, mypy, pytest).
19. ~~No `.gitattributes`~~ FIXED (`* text=auto eol=lf`).
20. The default `uv` interpreter here is Python 3.14; CI tests 3.11 and 3.12 but they were not run locally.
21. Weak labels from the current system's carts are not ground truth; they must not be used as the sole judge.
