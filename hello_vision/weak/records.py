"""Order records for weak supervision: only non-personal fields, strict product-name mapping."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

EXPORT_COLUMNS = ("order_id", "fridge_key", "date", "video0_url", "video1_url", "item_name", "qty")


class OrderRecord(BaseModel):
    """One completed order. Extra fields are rejected so personal data cannot slip in."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    fridge_key: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    date: date
    videos: list[str]  # URLs (before download) or file names (after)
    items: dict[str, int]  # gallery SKU -> quantity
    skipped_lines: int = 0  # order lines dropped for an unknown product: cart incomplete


class UnknownProductError(ValueError):
    """Admin product names with no mapping; ``names`` lists all of them."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = sorted(set(names))
        super().__init__(f"unknown product names (add to configs/sku_names.yaml): {self.names}")


def normalise(name: str) -> str:
    """Case, space and punctuation insensitive form (``+`` is kept: ``18+`` is a product)."""
    return re.sub(r"[^\w+]|_", "", name.casefold())


class SkuMapper:
    """Maps admin product names to gallery SKUs."""

    def __init__(self, names: dict[str, str]) -> None:
        self._map: dict[str, str] = {}
        for name, sku in names.items():
            key = normalise(name)
            if self._map.get(key, sku) != sku:
                raise ValueError(f"name {name!r} maps to two different SKUs")
            self._map[key] = sku

    @classmethod
    def load(cls, path: Path) -> SkuMapper:
        """Read ``names:`` from a YAML file."""
        return cls(yaml.safe_load(path.read_text(encoding="utf-8"))["names"])

    def get(self, name: str) -> str | None:
        """The SKU for ``name`` or ``None``."""
        return self._map.get(normalise(name))

    def cart(self, lines: Iterable[tuple[str, int]]) -> dict[str, int]:
        """Convert ``(admin name, qty)`` lines to ``{sku: qty}``.

        Raises:
            UnknownProductError: If any name is unmapped (all of them are reported).
        """
        cart: Counter[str] = Counter()
        unknown: list[str] = []
        for name, qty in lines:
            sku = self.get(name)
            if sku is None:
                unknown.append(name)
            elif qty < 1:
                raise ValueError(f"quantity of {name!r} must be positive, got {qty}")
            else:
                cart[sku] += qty
        if unknown:
            raise UnknownProductError(unknown)
        return dict(cart)


def propose_sku(name: str, price: int | None, prices: dict[str, int]) -> list[str]:
    """Candidate SKUs for an unknown admin name: same price first, then name similarity.

    A proposal only: a person must confirm it before it goes into the name map.
    """
    key = normalise(name)

    def similarity(sku: str) -> float:
        return SequenceMatcher(None, key, normalise(sku)).ratio()

    same = [s for s, p in prices.items() if price is not None and p == price]
    pool = same or list(prices)
    return sorted(pool, key=similarity, reverse=True)[:3]


def read_export(path: Path, mapper: SkuMapper, skip_unknown: bool = False) -> list[OrderRecord]:
    """Read a server-side CSV export (one row per order line, columns :data:`EXPORT_COLUMNS`).

    Raises:
        ValueError: Unexpected columns (to keep personal data out) or bad rows.
        UnknownProductError: Unmapped product names, all listed at once (only when
            ``skip_unknown`` is false; otherwise those lines are dropped and counted in
            ``skipped_lines``, and orders with no known line left are dropped).
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = set(reader.fieldnames or [])
        if columns != set(EXPORT_COLUMNS):
            raise ValueError(
                f"export must have exactly the columns {list(EXPORT_COLUMNS)}; "
                f"got {sorted(columns)}"
            )
        rows = list(reader)
    unknown = [r["item_name"] for r in rows if mapper.get(r["item_name"]) is None]
    if unknown and not skip_unknown:
        raise UnknownProductError(unknown)
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["order_id"], []).append(r)
    out: list[OrderRecord] = []
    for order_id, lines in grouped.items():
        first = lines[0]
        known = [r for r in lines if mapper.get(r["item_name"]) is not None]
        if not known:
            continue
        out.append(
            OrderRecord(
                order_id=order_id,
                fridge_key=first["fridge_key"],
                date=date.fromisoformat(first["date"][:10]),
                videos=[u for u in (first["video0_url"], first["video1_url"]) if u],
                items=mapper.cart((r["item_name"], int(r["qty"])) for r in known),
                skipped_lines=len(lines) - len(known),
            )
        )
    return out


def load_records(path: Path) -> list[OrderRecord]:
    """Read a JSONL record store (later lines replace earlier ones with the same order id)."""
    if not path.exists():
        return []
    out: dict[str, OrderRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = OrderRecord.model_validate(json.loads(line))
            out[r.order_id] = r
    return list(out.values())


def append_record(path: Path, record: OrderRecord) -> None:
    """Append one record (resume-friendly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
