"""SKU catalog models and (de)serialisation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Sku(BaseModel):
    """One sellable product.

    Attributes:
        sku: Stable identifier (letters, digits, ``_`` and ``-`` only; used as a folder name).
        name: Human readable name.
        group: Optional look-alike group (e.g. same-shaped bottles) used for hard-negative
            mining and confusion reporting.
    """

    model_config = ConfigDict(frozen=True)

    sku: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1)
    group: str | None = None


class Catalog(BaseModel):
    """An ordered collection of unique SKUs."""

    skus: list[Sku]

    @model_validator(mode="after")
    def _unique_ids(self) -> Catalog:
        ids = [s.sku for s in self.skus]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate sku ids: {dupes}")
        return self

    def ids(self) -> list[str]:
        """Return all SKU ids in catalog order."""
        return [s.sku for s in self.skus]

    def get(self, sku: str) -> Sku:
        """Return the SKU with id ``sku``.

        Raises:
            KeyError: If the id is not in the catalog.
        """
        for s in self.skus:
            if s.sku == sku:
                return s
        raise KeyError(sku)

    @classmethod
    def load(cls, path: Path) -> Catalog:
        """Load a catalog from a ``.yaml``/``.yml`` or ``.json`` file."""
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        return cls.model_validate(data)

    def dump(self, path: Path) -> None:
        """Write the catalog to ``path`` (``.json`` or YAML by suffix)."""
        data = self.model_dump()
        if path.suffix == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
