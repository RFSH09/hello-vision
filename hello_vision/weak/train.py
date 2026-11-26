"""Training and evaluation of :class:`~hello_vision.weak.model.SessionMIL` on saved features."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hello_vision.weak.features import dequantise
from hello_vision.weak.metrics import OTHER, CartReport, evaluate_carts
from hello_vision.weak.model import SessionMIL, decode, mil_loss
from hello_vision.weak.records import OrderRecord


def choose_classes(train: Sequence[OrderRecord], top: int) -> list[str]:
    """The ``top`` most frequent SKUs in the TRAINING orders, plus ``other`` if any are left."""
    seen: Counter[str] = Counter(s for r in train for s in r.items)
    ranked = [s for s, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked[:top] + ([OTHER] if len(ranked) > top else [])


def target_cart(items: dict[str, int], classes: Sequence[str]) -> dict[str, int]:
    """Map an order to the model's classes (unlisted SKUs are summed into ``other``)."""
    out: Counter[str] = Counter()
    for sku, n in items.items():
        key = sku if sku in classes else OTHER
        if key in classes:
            out[key] += n
        else:
            out[OTHER] += n  # unseen in training and no "other" class: can never be exact
    return dict(out)


def load_batch(
    records: Sequence[OrderRecord], features: Path, device: str
) -> tuple[torch.Tensor, ...]:
    """Load and pad the features of several orders."""
    items: list[Any] = []
    for r in records:
        with np.load(features / f"{r.order_id}.npz") as d:
            items.append(
                (
                    dequantise(d["full"], d["full_scale"]),
                    dequantise(d["crop"], d["crop_scale"]),
                    d["cam"],
                    d["time"],
                )
            )
    frames = max(1, max(len(i[2]) for i in items))
    dim = max(i[0].shape[-1] for i in items)
    b = len(items)
    full = np.zeros((b, frames, 17, dim), np.float32)
    crop = np.zeros_like(full)
    cam = np.zeros((b, frames), np.int64)
    time = np.zeros((b, frames), np.float32)
    mask = np.zeros((b, frames), bool)
    for k, (f, c, m, t) in enumerate(items):
        n = len(m)
        if n:
            full[k, :n], crop[k, :n], cam[k, :n], mask[k, :n] = f, c, m, True
            time[k, :n] = t / max(float(t.max()), 1e-6)
    return tuple(torch.from_numpy(a).to(device) for a in (full, crop, cam, time, mask))


@dataclass
class TrainConfig:
    """Training settings."""

    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.05
    patience: int = 6
    seed: int = 0
    width: int = 192


def predict(
    model: SessionMIL,
    records: Sequence[OrderRecord],
    features: Path,
    classes: Sequence[str],
    device: str,
    batch_size: int = 16,
) -> tuple[dict[str, dict[str, int]], dict[str, np.ndarray[Any, Any]]]:
    """Predicted carts and per-class frame attention for each order."""
    model.eval()
    carts: dict[str, dict[str, int]] = {}
    attention: dict[str, np.ndarray[Any, Any]] = {}
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        with torch.inference_mode():
            out = model(*load_batch(chunk, features, device))
        counts = decode(out).cpu().numpy()
        for k, r in enumerate(chunk):
            carts[r.order_id] = {c: int(n) for c, n in zip(classes, counts[k], strict=True) if n}
            attention[r.order_id] = out["attention"][k].cpu().numpy()
    return carts, attention


def evaluate(
    model: SessionMIL,
    records: Sequence[OrderRecord],
    features: Path,
    classes: Sequence[str],
    device: str,
) -> CartReport:
    """Evaluate on a split."""
    carts, _ = predict(model, records, features, classes, device)
    truth = {r.order_id: target_cart(r.items, classes) for r in records}
    return evaluate_carts(truth, carts, classes)


def train(
    train_records: Sequence[OrderRecord],
    val_records: Sequence[OrderRecord],
    features: Path,
    classes: Sequence[str],
    out_dir: Path,
    cfg: TrainConfig | None = None,
    device: str | None = None,
) -> SessionMIL:
    """Train with mixed precision and early stopping on validation exact-cart accuracy.

    Ties are broken by lower validation loss. The best model is saved to ``out_dir``.
    """
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    with np.load(features / f"{train_records[0].order_id}.npz") as d:
        dim = int(d["full"].shape[-1])
    model = SessionMIL(len(classes), dim=dim, width=cfg.width).to(device)
    targets = [target_cart(r.items, classes) for r in train_records]
    positives = torch.tensor([sum(c in t for t in targets) for c in classes], dtype=torch.float32)
    pos_weight = (((len(targets) - positives) / positives.clamp(min=1)).clamp(1, 20)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def count_tensor(chunk: Sequence[OrderRecord]) -> torch.Tensor:
        rows = [[target_cart(r.items, classes).get(c, 0) for c in classes] for r in chunk]
        return torch.tensor(rows, device=device)

    def val_loss() -> float:
        model.eval()
        total = 0.0
        for i in range(0, len(val_records), cfg.batch_size):
            chunk = val_records[i : i + cfg.batch_size]
            with torch.inference_mode():
                out = model(*load_batch(chunk, features, device))
                total += float(mil_loss(out, count_tensor(chunk), pos_weight)) * len(chunk)
        return total / max(len(val_records), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    best = (-1.0, float("-inf"))
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_records))
        running = 0.0
        for i in range(0, len(order), cfg.batch_size):
            chunk = [train_records[j] for j in order[i : i + cfg.batch_size]]
            batch = load_batch(chunk, features, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = mil_loss(model(*batch), count_tensor(chunk), pos_weight)
            optimiser.zero_grad()
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            running += float(loss.detach()) * len(chunk)
        exact = evaluate(model, val_records, features, classes, device).exact
        loss_v = val_loss()
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(order),
                "val_loss": loss_v,
                "val_exact": exact,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        score = (exact, -loss_v)
        if score > best:
            best, stale = score, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": list(classes),
                    "dim": dim,
                    "width": cfg.width,
                },
                out_dir / "model.pt",
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    state = torch.load(out_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model
