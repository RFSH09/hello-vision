"""Leakage-free train / val / test split by FRIDGE and DATE.

Test = held-out fridges on the newest days. Val = other held-out fridges on the days just
before. Train = the remaining fridges on the older days. Orders that mix (a held-out fridge on
an old day, a training fridge on a held-out day, ...) are ``excluded``, so no fridge and no
day appears in two splits.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from hello_vision.weak.records import OrderRecord

SPLITS = ("train", "val", "test", "excluded")


def _ranked(keys: Sequence[str], seed: int) -> list[str]:
    return sorted(set(keys), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())


def select_for_download(
    records: Sequence[OrderRecord], assignment: dict[str, str], limit: int
) -> list[OrderRecord]:
    """The orders worth downloading: never ``excluded``; all test, then val, then newest train.

    Choosing after the split means none of the download budget is spent on orders the
    leakage-free rules would throw away.
    """
    priority = {"test": 0, "val": 1, "train": 2}
    usable = [r for r in records if assignment.get(r.order_id) in priority]
    usable.sort(key=lambda r: (priority[assignment[r.order_id]], -r.date.toordinal(), r.order_id))
    return usable[:limit]


def split_orders(
    records: Sequence[OrderRecord],
    val_fridges: float = 0.15,
    test_fridges: float = 0.2,
    val_days: float = 0.12,
    test_days: float = 0.15,
    seed: int = 0,
) -> dict[str, str]:
    """Assign every order id to a split.

    Args:
        records: All orders.
        val_fridges: Share of fridges reserved for validation.
        test_fridges: Share of fridges reserved for testing.
        val_days: Share of distinct days (just before the test days) for validation.
        test_days: Share of distinct days (newest) for testing.
        seed: Changes which fridges are held out.

    Returns:
        ``order_id -> split``.

    Raises:
        ValueError: Not enough fridges or days to make three disjoint splits.
    """
    fridges = _ranked([r.fridge_key for r in records], seed)
    days = sorted({r.date for r in records})
    if len(fridges) < 3 or len(days) < 3:
        raise ValueError("need at least 3 fridges and 3 days for a leakage-free split")
    n_test = max(1, round(test_fridges * len(fridges)))
    n_val = max(1, round(val_fridges * len(fridges)))
    d_test = max(1, round(test_days * len(days)))
    d_val = max(1, round(val_days * len(days)))
    if n_test + n_val >= len(fridges) or d_test + d_val >= len(days):
        raise ValueError("held-out shares leave nothing to train on")
    fridge_split = {f: "test" for f in fridges[:n_test]}
    fridge_split |= {f: "val" for f in fridges[n_test : n_test + n_val]}
    day_split = {d: "test" for d in days[-d_test:]}
    day_split |= {d: "val" for d in days[-d_test - d_val : -d_test]}
    out: dict[str, str] = {}
    for r in records:
        f = fridge_split.get(r.fridge_key, "train")
        d = day_split.get(r.date, "train")
        out[r.order_id] = f if f == d else "excluded"
    return out
