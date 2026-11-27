"""Labelled real crops: embeddings for the multi-prototype gallery and the product filter.

Crops are cut from the ORIGINAL frames (never blurred) of the tracks that were labelled.
Everything stays leave-one-order-out: models for an order never see that order's crops.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hello_vision.capture.types import Image
from hello_vision.datatools.crops import pick, track_candidates
from hello_vision.datatools.label_sheet import NOT_PRODUCT, UNSURE
from hello_vision.identify.embedders import Vector
from hello_vision.identify.index import GalleryIndex
from hello_vision.identify.product_filter import ProductFilter


@dataclass
class LabelledCrops:
    """Embedded crops with their track label and order."""

    vectors: np.ndarray[Any, np.dtype[np.float32]]
    labels: list[str]  # SKU id or NOT_PRODUCT
    orders: list[str]
    tracks: list[str]  # "<video>:<track id>"

    def __len__(self) -> int:
        """Number of crops."""
        return len(self.labels)

    def gallery(self, base: GalleryIndex, exclude_order: str | None) -> GalleryIndex:
        """``base`` (packshots) plus real product crops from all other orders."""
        index = GalleryIndex(base.threshold, base.min_margin)
        for sku, v in zip(*base.export(), strict=True):
            index.add(sku, v)
        for v, label, order in zip(self.vectors, self.labels, self.orders, strict=True):
            if label != NOT_PRODUCT and order != exclude_order:
                index.add(label, v)
        return index

    def product_filter(self, exclude_order: str | None) -> ProductFilter:
        """Classifier trained on all other orders."""
        keep = [i for i, o in enumerate(self.orders) if o != exclude_order]
        y = [0 if self.labels[i] == NOT_PRODUCT else 1 for i in keep]
        return ProductFilter().fit(self.vectors[keep], y)


def labelled_ids(labels: dict[str, str]) -> dict[str, str]:
    """Track id -> label, without the unsure ones."""
    return {k: v for k, v in labels.items() if v != UNSURE}


def extract(
    labels: dict[str, str], results_dir: Path, videos_dir: Path, per_track: int = 8
) -> list[tuple[str, str, Image]]:
    """``(track id, label, raw crop)`` for the labelled tracks of the results they refer to."""
    wanted: dict[str, dict[int, str]] = defaultdict(dict)
    for key, label in labelled_ids(labels).items():
        stem, _, track = key.rpartition(":")
        wanted[stem][int(track)] = label
    out: list[tuple[str, str, Image]] = []
    for stem, tracks in sorted(wanted.items()):
        report = json.loads((results_dir / f"{stem}.json").read_text(encoding="utf-8"))
        chosen = [t for t in report["tracks"] if int(t["track_id"]) in tracks]
        found = track_candidates(videos_dir / f"{stem}.mp4", chosen, min_boxes=1, pad=0.0)
        for track_id, candidates in sorted(found.items()):
            for c in pick(candidates, n=per_track, min_gap_s=0.15):
                out.append((f"{stem}:{track_id}", tracks[track_id], c.crop))
    return out


def load_or_build(
    labels_path: Path,
    results_dir: Path,
    videos_dir: Path,
    embedder: Any,  # noqa: ANN401 - needs embed_batch
    cache_dir: Path,
) -> LabelledCrops:
    """Embed the labelled crops, cached by the content of the labels file."""
    from sim.orders import order_id

    raw = labels_path.read_bytes()
    cache = cache_dir / f"real_crops_{hashlib.sha256(raw).hexdigest()[:16]}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return LabelledCrops(
            d["vectors"], list(d["labels"].astype(str)), list(d["orders"].astype(str)),
            list(d["tracks"].astype(str)),
        )  # fmt: skip
    labels: dict[str, str] = json.loads(raw)["labels"]
    items = extract(labels, results_dir, videos_dir)
    if not items:
        raise ValueError(f"no labelled crops found for {labels_path} in {results_dir}")
    vectors: Sequence[Vector] = embedder.embed_batch([c for _, _, c in items])
    out = LabelledCrops(
        np.stack(vectors),
        [label for _, label, _ in items],
        [order_id(key.rpartition(":")[0]) for key, _, _ in items],
        [key for key, _, _ in items],
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, vectors=out.vectors, labels=np.array(out.labels), orders=np.array(out.orders),
        tracks=np.array(out.tracks),
    )  # fmt: skip
    return out
