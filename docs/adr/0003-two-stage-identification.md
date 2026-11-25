# ADR 0003: Class-agnostic detector + embedding gallery

Status: accepted.

Decision: detect "product" and "hand" only; identify SKUs by embedding nearest-neighbour search
against a gallery, with an `unknown` outcome under a similarity threshold.

Consequences: new SKUs need gallery images, not detector retraining; look-alike SKUs need
metric-learning fine-tuning and per-SKU confusion tracking.
