# Licenses of models, datasets and dependencies

Our code: Apache-2.0. Rule: Apache/MIT/BSD only; no AGPL/GPL.

**Nothing below is adopted yet. "Candidate" entries must be re-verified (code, weights and
training data) before use.**

| Component | Role | License (to verify) | Status |
| --- | --- | --- | --- |
| RT-DETR / D-FINE | detector | Apache-2.0 | candidate |
| RTMDet (MMDetection) | detector | Apache-2.0 | candidate |
| YOLOX | detector | Apache-2.0 | candidate |
| DINOv2 | embeddings | Apache-2.0 (check weights) | candidate |
| CLIP-style (OpenCLIP) | embeddings | MIT (check weights) | candidate |
| ByteTrack | tracking | MIT | candidate |
| ONNX Runtime | inference | MIT | candidate |
| FAISS | index | MIT | candidate |
| Pretrained weights / COCO | pretraining | check per file | open |
| Ultralytics YOLO | - | AGPL-3.0 | **forbidden** |

## Runtime dependencies in use (M1)

| Package | License | Note |
| --- | --- | --- |
| numpy | BSD-3-Clause | |
| Pillow | MIT-CMU (HPND) | |
| opencv-python-headless | Apache-2.0 | wheels bundle FFmpeg (LGPL-2.1, dynamically linked); confirm this is acceptable for distribution |
| pydantic | MIT | |
| PyYAML | MIT | |

Dev tools (ruff, mypy, pytest, pre-commit) are not shipped. Versions are pinned in `uv.lock`.

| onnxruntime (optional extra) | MIT | |
| torch, torchvision (optional extra, training only) | BSD-3-Clause | |

| fastapi | MIT | |
| starlette | BSD-3-Clause | transitive of fastapi |
| httpx | BSD-3-Clause | |
| uvicorn | BSD-3-Clause | |
| YuNet face model (OpenCV Zoo) | MIT | downloaded at deploy time by tools/fetch_face_model.py, not in the repo |

## Dependency license audit (2026-11-25)

Scanned the full installed tree (runtime and dev, 44 packages) with
`uv run --with pip-licenses pip-licenses --format=csv --with-urls --order=license`. Re-run it before every release.

- **Permissive (Apache-2.0, MIT, BSD, MIT-CMU/HPND, PSF, 0BSD, Zlib):** everything except the two packages below. This includes
  numpy, Pillow, OpenCV, pydantic, PyYAML, FastAPI, Starlette, uvicorn, httpx/httpcore/h11, anyio, opentelemetry-api (pulled in by the web stack).
- **MPL-2.0 (weak, file-level copyleft): `certifi` (runtime, required by httpx) and `pathspec` (development tooling only).**
  MPL-2.0 only requires that modifications to those files be shared; using them unmodified as separate packages is normally fine,
  but it does not meet the project rule "Apache/MIT/BSD only". **Decision needed from the owner:** accept MPL-2.0 for `certifi`, or
  replace the HTTP client. (`pathspec` is not shipped.)
- **LGPL (bundled, not a Python dependency):** the OpenCV wheel ships FFmpeg under LGPL-2.1, dynamically linked. Still to confirm.
- **Not scanned:** optional extras (`onnxruntime`, `torch`), model weights and datasets (each needs its own check when adopted).
