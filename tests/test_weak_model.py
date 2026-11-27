"""Session model and training loop (needs torch; skipped where it is not installed)."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hello_vision.weak.features import quantise  # noqa: E402
from hello_vision.weak.model import SessionMIL, decode, mil_loss  # noqa: E402
from hello_vision.weak.records import OrderRecord  # noqa: E402
from hello_vision.weak.train import (  # noqa: E402
    TrainConfig,
    choose_classes,
    choose_threshold,
    predict,
    target_cart,
    train,
)


def rec(i: int, items: dict[str, int]) -> OrderRecord:
    return OrderRecord(
        order_id=f"o{i}", fridge_key="f", date=date(2026, 9, 1), videos=[], items=items
    )


def test_classes_and_targets() -> None:
    records = [rec(0, {"a": 1, "b": 1}), rec(1, {"a": 1}), rec(2, {"c": 1})]
    classes = choose_classes(records, 1)
    assert classes == ["a", "other"]
    assert target_cart({"a": 2, "b": 1, "c": 1}, classes) == {"a": 2, "other": 2}
    assert choose_classes(records, 10) == ["a", "b", "c"]


def test_forward_shapes_padding_and_empty_session() -> None:
    model = SessionMIL(3, dim=8, width=32).eval()
    full = torch.randn(2, 5, 17, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    out = model(full, full, torch.zeros(2, 5, dtype=torch.long), torch.rand(2, 5), mask)
    assert out["presence"].shape == (2, 3) and out["count"].shape == (2, 3, 3)
    assert torch.isfinite(out["presence"]).all()
    assert torch.allclose(out["attention"][0, :, 3:], torch.zeros(3, 2))  # padding ignored
    assert torch.isfinite(mil_loss(out, torch.tensor([[1, 0, 5], [0, 0, 0]]), torch.ones(3)))
    assert decode(out).max() <= 3


def test_training_learns_a_planted_signal(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    records = []
    for i in range(48):
        sku = "a" if i % 2 else "b"
        records.append(rec(i, {sku: 1}))
        tokens = rng.normal(0, 0.3, (6, 17, 8)).astype(np.float16)
        tokens[2, :, 0 if sku == "a" else 1] += 4  # one informative frame
        stored, scale = quantise(tokens.astype(np.float32))
        np.savez(tmp_path / f"o{i}.npz", full=stored, crop=stored, full_scale=scale, crop_scale=scale,
                 cam=np.zeros(6, np.int8), time=np.arange(6, dtype=np.float32),
                 box=np.zeros((6, 4), np.int16))  # fmt: skip
    cfg = TrainConfig(epochs=25, batch_size=8, lr=3e-3, patience=25, width=32)
    model = train(records[:32], records[32:], tmp_path, ["a", "b"], tmp_path / "m", cfg, "cpu")
    carts, attention = predict(model, records[32:], tmp_path, ["a", "b"], "cpu")
    right = sum(carts[r.order_id] == r.items for r in records[32:])
    assert right >= 14 and (tmp_path / "m" / "model.pt").exists()
    assert attention["o33"].shape == (2, 6) and attention["o33"][0].argmax() == 2

    threshold, safe = choose_threshold(model, records[32:], tmp_path, ["a", "b"], "cpu")
    assert threshold == 0.5 and safe.false_charge_rate <= 0.01
    flagged: set[str] = set()
    carts, _ = predict(
        model, records[32:], tmp_path, ["a", "b"], "cpu",
        threshold=1.01, review_low=0.0, flagged=flagged,
    )  # fmt: skip
    assert not any(carts.values()) and len(flagged) == 16  # unsure -> review, nothing charged
