"""Read item lines and video links from the order admin pages (nothing else is extracted).

The pages also contain customer data. Only inline item-table cells and ``.mp4`` links are
returned, pages are never stored, and a row whose "name" looks like a number string (phone,
card) is dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_VOID = {"input", "br", "img", "meta", "link", "hr", "source", "col", "area", "base", "wbr"}
_TABLE_CLASSES = {"inline-related", "tabular"}
_LONG_NUMBER = re.compile(r"\d(?:[\s()+-]?\d){6,}")


@dataclass
class _Cell:
    text: list[str] = field(default_factory=list)
    input_value: str | None = None

    def value(self) -> str:
        if self.input_value is not None:
            return self.input_value.strip()
        return " ".join(" ".join(self.text).split())


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.rows: list[list[str]] = []
        self.links: list[str] = []
        self.login_form = False
        self._row: list[_Cell] | None = None
        self._cell: _Cell | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k: v or "" for k, v in attrs}
        classes = set(values.get("class", "").split())
        if values.get("id") == "login-form" or (
            tag == "input" and values.get("type") == "password"
        ):
            self.login_form = True
        if tag == "a" and values.get("href", "").split("?")[0].lower().endswith(".mp4"):
            self.links.append(values["href"])
        if tag == "tr" and "empty-form" not in classes and self._in_item_table():
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = _Cell()
            self._row.append(self._cell)
        elif tag == "input" and self._cell is not None and self._cell.input_value is None:
            self._cell.input_value = values.get("value", "")
        elif tag in ("script", "style"):
            self._skip += 1
        if tag not in _VOID:
            self.stack.append((tag, classes))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag == "tr" and self._row is not None:
            self.rows.append([c.value() for c in self._row])
            self._row = self._cell = None
        elif tag == "td":
            self._cell = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._skip:
            self._cell.text.append(data)

    def _in_item_table(self) -> bool:
        """Inside ``.inline-related tbody`` or ``.tabular tbody``."""
        marked = False
        for tag, classes in self.stack:
            marked = marked or bool(classes & _TABLE_CLASSES)
            if marked and tag == "tbody":
                return True
        return False


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    page.close()
    return page


def is_login_page(html: str) -> bool:
    """Whether the server answered with a login form (the session has expired)."""
    return _parse(html).login_form


def parse_price(text: str) -> int | None:
    """Whole so'm from ``"9 000"``, ``"9,000.00"`` or ``"9000"``; ``None`` if not a price."""
    compact = re.sub(r"[\s,]|so.?m|UZS", "", text, flags=re.IGNORECASE)
    match = re.fullmatch(r"(\d{3,7})(?:\.\d+)?", compact)
    return int(match.group(1)) if match else None


def parse_items(html: str) -> list[tuple[str, int]]:
    """``(item name, quantity)`` of every item line (see :func:`parse_item_lines`)."""
    return [(name, qty) for name, qty, _ in parse_item_lines(html)]


def parse_item_lines(html: str) -> list[tuple[str, int, int | None]]:
    """``(name, quantity, unit price)`` from the inline table: Name, Quantity, Price, Amount.

    Each cell contributes its input value if it has an input, else its text; empty cells are
    dropped; a row needs at least three values, of which the second is the name and the third
    the quantity.
    """
    items: list[tuple[str, int, int | None]] = []
    for row in _parse(html).rows:
        values = [v for v in row if v]
        if len(values) < 3 or not re.fullmatch(r"\d{1,3}", values[2]):
            continue
        name, qty = values[1], int(values[2])
        if qty >= 1 and not _LONG_NUMBER.search(name):
            price = parse_price(values[3]) if len(values) > 3 else None
            items.append((name, qty, price))
    return items


def parse_videos(html: str) -> tuple[str, str]:
    """``(camera 0 URL, camera 1 URL)``; empty strings when a link is missing."""
    links = _parse(html).links
    first = next((u for u in links if "_1_0_" in u), "")
    second = next((u for u in links if "_1_1_" in u), "")
    return first, second
