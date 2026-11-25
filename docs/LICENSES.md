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

Transitive dependencies of the web stack were pulled in by `uv add` and have NOT been individually audited yet; run a license scan (for example `pip-licenses`) before any release.
