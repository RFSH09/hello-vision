"""Shared training helpers: seeding and plain-CSV experiment tracking."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed ``random`` and numpy (and torch if it is installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class CsvLogger:
    """Append metrics rows to a CSV file (a dependency-free alternative to MLflow).

    Args:
        path: CSV path; the header is written from the first row's keys.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None

    def log(self, **row: float | int | str) -> None:
        """Append one row (all rows must use the same keys)."""
        new = not self.path.exists()
        if self._fields is None:
            self._fields = list(row)
        if list(row) != self._fields:
            raise ValueError(f"row keys {list(row)} differ from header {self._fields}")
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self._fields)
            if new:
                w.writeheader()
            w.writerow(row)
