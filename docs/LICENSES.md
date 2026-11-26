# Licenses of models, datasets and dependencies

Our code: Apache-2.0. Policy below (enforced by `tools/check_licenses.py` in CI).

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

## License policy (decided 2026-11-26)

- **Allowed:** MIT, BSD, Apache-2.0 and equivalents (PSF, HPND, ISC, Zlib, 0BSD).
- **MPL-2.0: accepted only** for these unmodified third-party packages: `certifi` (HTTP certificates), `orjson` (web UI),
  `tqdm` (model library), `pathspec` (development only). MPL-2.0 is file-level: it covers changes to those files, not our code.
  Never copy or modify their source in this repository. Any other MPL package needs a new decision.
- **Forbidden:** GPL, AGPL, SSPL, non-commercial licenses (so no Ultralytics YOLO).
- **Bundled inside wheels, not covered by the package check:** OpenCV ships FFmpeg (LGPL-2.1, dynamically linked); the CUDA
  build of PyTorch ships NVIDIA libraries under NVIDIA's own terms. We do not redistribute either in this repository. Review
  both before shipping a device image or Docker image.

This reflects an engineering reading of package metadata, not legal advice.

## Verified (2026-11-26)

86 installed packages (runtime, vision, app, dev) pass the policy. Models, checked at their source:

| Model | License |
| --- | --- |
| google/owlv2-base-patch16-ensemble | Apache-2.0 |
| IDEA-Research/grounding-dino-tiny | Apache-2.0 |
| facebook/dinov2-small | Apache-2.0 |
| YuNet face detector (OpenCV Zoo) | MIT |

Training data of those pretrained models has not been reviewed. Added in this step: transformers (Apache-2.0), torch and
torchvision (BSD-3-Clause), scipy (BSD-3-Clause), gradio (Apache-2.0).
