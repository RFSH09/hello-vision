# ADR 0002: Permissive licenses only

Status: accepted.

Context: the repo is public and the module will run in a commercial product.

Decision: Apache-2.0 for our code; dependencies, models and datasets must be Apache/MIT/BSD
(or equivalent). No AGPL/GPL code (e.g. Ultralytics YOLO). Candidates: RT-DETR, D-FINE, RTMDet,
YOLOX (detection); DINOv2 or CLIP-style (embeddings); ByteTrack-style tracking.

Consequences: every model and dataset is recorded in `docs/LICENSES.md` before use, including
pretrained weights, which can carry separate terms.
