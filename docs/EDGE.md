# Edge hardware

Budget (draft, owner to confirm): up to about 2.5 mln so'm (~$200) per fridge for the compute box.

## What has been MEASURED

Only one machine, and only the synthetic classical baselines (no learned model exists yet):

| Machine | Stage | Mean | p95 | Notes |
| --- | --- | --- | --- | --- |
| Windows 11 desktop, Intel 16 logical CPUs | baseline detector, 1 frame (960x540) | 43 ms | 56 ms | NumPy/OpenCV, single process |
| same | baseline embedder, 1 crop | 0.7 ms | 1.8 ms | |
| same | whole session, 44 frames | 1.85 s | 1.92 s | detector dominates; about 42 ms per frame |

Reproduce with `uv run python -m hello_vision.edge.benchmark --out bench.json`. These numbers say nothing about the
real models or about the edge boards: a learned detector at 640x640 will cost far more on a CPU, and a typical
session has more frames than this 44-frame demo. **Nothing has been measured on Jetson, Intel mini-PC, or RK3588,
and no power or cost has been measured.**

## Export paths (commands built, not yet run)

`hello_vision/edge/export.py` produces the conversion commands from an ONNX model: ONNX Runtime (CPU), TensorRT
(`trtexec`, on the Jetson), OpenVINO (`ovc`), RKNN (`rknn-toolkit2` script). Each must be run on its own toolchain
with a real model before it counts as supported.

## Candidate boards: PROVISIONAL, unverified

This is a shortlist from general knowledge, to be replaced by measurements. Prices are rough and must be re-checked.

| Option | Runtime | Why consider | Risks |
| --- | --- | --- | --- |
| Rockchip RK3588 board (Orange Pi 5 / Radxa class) | RKNN (NPU, ~6 TOPS) | fits the budget, low power, built-in NPU | RKNN conversion of transformer-style detectors/embedders can be painful; operator support must be tested; toolchain is vendor-specific |
| Intel N100/N305 mini PC | OpenVINO / ONNX Runtime CPU+iGPU | easy x86 software stack, Docker friendly, often within budget | no NPU; fan/thermals inside a fridge need checking |
| NVIDIA Jetson Orin Nano | TensorRT | best software ecosystem for vision models | likely above the $200 budget; higher power |
| CPU only on the fridge's existing computer | ONNX Runtime | zero extra hardware | the current computer runs the supplier's software and may be closed |

**Provisional recommendation (to be confirmed by benchmarks):** benchmark the RK3588 and an Intel N100 box first, because
they are the two that fit the budget. Pick by measured latency at the target model, with the 5 s door-close budget
as the hard limit. Do not buy in volume before that.

## How to benchmark a board

1. Export the trained detector and embedder to ONNX; convert with the commands above.
2. Run `hello_vision.edge.benchmark` on the board with the real models (extend `run_benchmark` to load them).
3. Record: p50/p95 per stage and for a full 5 s session, sustained FPS under thermal load, and power at the wall.
4. Fill in the table in this file.

## Operations

- **Docker:** `docker/Dockerfile` (not built or tested yet; Docker is not installed on the dev machine).
- **systemd:** `deploy/hello-vision.service` with watchdog (`WatchdogSec=60`), restart on failure and sandboxing. Not tested
  on a device. The watchdog pings only while the service reports healthy.
- **Health and metrics:** `GET /health`, `GET /metrics` (Prometheus text).
- **Logs:** JSON lines on stderr (journald).
- **Model updates:** `hello_vision/edge/ota.py` installs a new model version with checksum verification, activates it
  atomically, restores the previous version automatically if the health check fails, and supports manual rollback.
  The analyzer factory should report `ModelStore.current` as `model_version`, so every result names the model that
  produced it. The download/transport side of OTA (who pushes models, signing of model files) is not designed yet.
