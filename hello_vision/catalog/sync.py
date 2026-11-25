"""Parse the backend catalog payload (``GET /api/vision/catalog``, see docs/API.md).

Only parsing lives here; the HTTP client arrives with the service in M5.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from hello_vision.catalog.models import Catalog, Sku


class _PayloadSku(BaseModel):
    id: str
    name: str
    group: str | None = None


class CatalogPayload(BaseModel):
    """Wire format of the catalog endpoint.

    ``planograms`` maps fridge id -> shelf id -> ordered SKU ids.
    """

    skus: list[_PayloadSku]
    planograms: dict[str, dict[str, list[str]]] = {}


def parse_catalog_payload(
    payload: dict[str, Any],
) -> tuple[Catalog, dict[str, dict[str, list[str]]]]:
    """Validate a backend payload and convert it to a :class:`Catalog` plus planograms.

    Args:
        payload: Decoded JSON from the backend.

    Returns:
        The catalog and the per-fridge planograms.

    Raises:
        ValueError: If a planogram references a SKU that is not in the catalog.
    """
    parsed = CatalogPayload.model_validate(payload)
    catalog = Catalog(skus=[Sku(sku=s.id, name=s.name, group=s.group) for s in parsed.skus])
    known = set(catalog.ids())
    for fridge, shelves in parsed.planograms.items():
        for shelf, ids in shelves.items():
            unknown = [i for i in ids if i not in known]
            if unknown:
                raise ValueError(f"planogram {fridge}/{shelf} references unknown SKUs {unknown}")
    return catalog, parsed.planograms
