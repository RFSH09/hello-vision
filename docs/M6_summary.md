# M6 summary

Edge and operations tooling: benchmark harness, export command plans (ONNX CPU / TensorRT / OpenVINO / RKNN), model store with checksum, atomic activation and rollback, JSON logging, systemd watchdog, Dockerfile, systemd unit. See [EDGE.md](EDGE.md).

- **Measured:** one desktop CPU, synthetic baselines only (about 42 ms per frame for the classical detector).
- **Not done / not verified:** no Jetson, Intel or RK3588 measurement, no power or cost data, Docker image never built, systemd unit never run, the watchdog socket test is skipped on Windows, exports never run on a real model. The board recommendation is provisional.
