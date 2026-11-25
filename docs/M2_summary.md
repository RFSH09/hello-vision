# M2 summary

Detection and identification interfaces plus synthetic-only baselines.

- `BaselineDetector` (classical, synthetic only) and `OnnxDetector` (RT-DETR-style outputs, untested on a real model).
- `ColorLayoutEmbedder` (synthetic only) and `OnnxEmbedder` (DINOv2 export, untested on a real model).
- `GalleryIndex` with similarity threshold and margin, producing `unknown` instead of a forced guess.
- Metrics: IoU matching, per-SKU top-1, confusion matrix. Numbers: [M2_synthetic_metrics.md](M2_synthetic_metrics.md).
- Training scaffolding (`training/`): crop-dataset builder (tested), CSV tracker (tested), embedding fine-tune script and configs (NOT run), detector config (no trainer yet).

No learned model has been trained or downloaded. The synthetic numbers prove the plumbing, not recognition quality.
