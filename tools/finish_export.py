"""Finish a half-done order export from the admin pages: GET only, at most 1 request/s.

    python tools/finish_export.py --worklist data/hv_worklist.csv \
        --cookie-file data/admin_cookie.txt

Resumable: every fetched order is appended to ``export.partial.jsonl``. When all rows are
done, ``export.csv`` is written and the cookie file is deleted. Stops if the session expires.
The cookie is sent only to the admin host, never followed across redirects, never printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from hello_vision.weak.admin_pages import is_login_page, parse_item_lines, parse_videos
from hello_vision.weak.harvest import (
    Opener,
    RateLimiter,
    admin_opener,
    get,
    keep_awake,
    redact,
)
from hello_vision.weak.records import EXPORT_COLUMNS, OrderRecord, SkuMapper, propose_sku
from hello_vision.weak.splits import select_for_download, split_orders

HOST = "osonvend.uz"


class SessionExpiredError(RuntimeError):
    """The admin answered with its login page."""


class FetchError(RuntimeError):
    """A page could not be fetched after all retries."""


def fetch_page(
    url: str,
    cookie: str,
    limiter: RateLimiter,
    opener: Opener = admin_opener,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """GET one admin page with retries and backoff.

    Raises:
        ValueError: The address is not the admin host (the cookie is not sent).
        SessionExpiredError: Redirect to login, a login form, or access refused.
        FetchError: Still failing after ``retries`` retries.
    """
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or parts.hostname != HOST:
        raise ValueError(f"refusing to send the session cookie to {parts.hostname}")
    error = ""
    for attempt in range(retries + 1):
        if attempt:
            sleep(2.0**attempt)
        limiter.wait()
        try:
            with get(url, {"Cookie": cookie}, opener) as response:
                html: str = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            location = str(exc.headers.get("Location", "")) if exc.headers else ""
            if exc.code in (401, 403) or (exc.code in (301, 302) and "login" in location):
                raise SessionExpiredError(f"HTTP {exc.code}") from None
            error = f"HTTP {exc.code}"
            if 300 <= exc.code < 500:  # a missing or moved order will not fix itself
                break
        except (OSError, urllib.error.URLError) as exc:
            error = type(exc).__name__
        else:
            if is_login_page(html):
                raise SessionExpiredError("login page")
            return html
    raise FetchError(error)


def read_partial(path: Path) -> dict[str, dict[str, Any]]:
    """Orders already fetched (a half-written last line is ignored)."""
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["order_id"]] = row
    return done


def priority_order(rows: list[dict[str, str]], limit: int = 5000) -> list[dict[str, str]]:
    """Fetch the orders that will be downloaded first (the split only needs fridge and date)."""
    records = [
        OrderRecord(
            order_id=r["order_id"],
            fridge_key=r["fridge_key"],
            date=date.fromisoformat(r["date"][:10]),
            videos=[],
            items={},
        )
        for r in rows
    ]
    try:
        chosen = select_for_download(records, split_orders(records), limit)
    except ValueError:
        return rows
    rank = {r.order_id: i for i, r in enumerate(chosen)}
    return sorted(rows, key=lambda r: rank.get(r["order_id"], len(rank)))


def prefilled(row: dict[str, str]) -> dict[str, Any] | None:
    """The data of a row the browser export already completed."""
    if not (row.get("video0_url") and row.get("video1_url") and row.get("items_json")):
        return None
    return {
        "order_id": row["order_id"],
        "items": json.loads(row["items_json"]),
        "videos": [row["video0_url"], row["video1_url"]],
    }


def write_export(
    rows: Sequence[dict[str, str]], done: dict[str, dict[str, Any]], path: Path
) -> dict[str, int]:
    """Write ``export.csv`` (one line per item) and return summary counts."""
    counts = {"orders": 0, "written": 0, "no_videos": 0, "no_items": 0, "lines": 0}
    partial = path.with_suffix(".csv.tmp")
    with partial.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(EXPORT_COLUMNS)
        for row in rows:
            data = prefilled(row) or done.get(row["order_id"])
            if data is None:
                continue
            counts["orders"] += 1
            videos = [*data["videos"], "", ""][:2]
            counts["no_videos"] += not (videos[0] and videos[1])
            counts["no_items"] += not data["items"]
            if data["items"]:
                counts["written"] += 1
            for name, qty in data["items"]:
                writer.writerow(
                    [row["order_id"], row["fridge_key"], row["date"], *videos, name, qty]
                )
                counts["lines"] += 1
    partial.replace(path)
    return counts


def run(
    rows: list[dict[str, str]],
    cookie: str,
    partial: Path,
    limiter: RateLimiter,
    opener: Opener = admin_opener,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_failures: int = 15,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Fetch every missing order; returns all fetched data and the failures."""
    done = read_partial(partial)
    todo = [r for r in rows if prefilled(r) is None and r["order_id"] not in done]
    print(f"{len(rows)} orders, {len(todo)} to fetch ({2 * len(todo)} requests)", flush=True)
    failures: list[str] = []
    streak = 0
    start = time.monotonic()
    for n, row in enumerate(todo, start=1):
        try:
            order = fetch_page(row["order_page"], cookie, limiter, opener, sleep=sleep)
            completed = fetch_page(row["completed_page"], cookie, limiter, opener, sleep=sleep)
        except FetchError as exc:
            failures.append(f"{row['order_id']}: {exc}")
            streak += 1
            if streak >= max_consecutive_failures:
                print(f"stopping: {streak} orders in a row failed ({exc})", flush=True)
                break
            continue
        streak = 0
        lines = parse_item_lines(order)
        data = {
            "order_id": row["order_id"],
            "items": [(name, qty) for name, qty, _ in lines],
            "prices": {name: price for name, _, price in lines if price is not None},
            "videos": list(parse_videos(completed)),
        }
        with partial.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        done[row["order_id"]] = data
        if n <= 3 or n % 100 == 0 or n == len(todo):
            left = (time.monotonic() - start) / n * (len(todo) - n) / 3600
            print(
                f"{n}/{len(todo)}: {len(data['items'])} item lines, "
                f"{sum(map(bool, data['videos']))} videos, about {left:.1f} h left",
                flush=True,
            )
    return done, failures


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 done, 1 incomplete, 3 session expired."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=Path("data/hv_worklist.csv"))
    parser.add_argument("--cookie-file", type=Path, default=Path("data/admin_cookie.txt"))
    parser.add_argument("--partial", type=Path, default=Path("data/export.partial.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/export.csv"))
    parser.add_argument("--names", type=Path, default=Path("configs/sku_names.yaml"))
    args = parser.parse_args(argv)
    with args.worklist.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not args.cookie_file.is_file():
        print(f"cookie file not found: {args.cookie_file}")
        return 3
    cookie = args.cookie_file.read_text(encoding="utf-8-sig").strip()
    if not cookie.startswith("sessionid=") or "\n" in cookie:
        print("the cookie file must contain one line: sessionid=...")
        return 3
    keep_awake()
    try:
        done, failures = run(priority_order(rows), cookie, args.partial, RateLimiter())
    except SessionExpiredError as exc:
        print(f"SESSION EXPIRED ({exc}). Put a fresh cookie in the file and run again.")
        return 3
    counts = write_export(rows, done, args.out)
    complete = counts["orders"] == len(rows)
    print(
        f"orders done: {counts['orders']}/{len(rows)} | without videos: {counts['no_videos']} | "
        f"without items: {counts['no_items']} | item lines written: {counts['lines']}"
    )
    mapper = SkuMapper.load(args.names)
    with args.out.open(encoding="utf-8", newline="") as fh:
        unknown = Counter(
            r["item_name"] for r in csv.DictReader(fh) if mapper.get(r["item_name"]) is None
        )
    print(f"unknown product names ({len(unknown)}); the import will stop until they are mapped:")
    if unknown:
        known_prices = yaml.safe_load(args.names.read_text(encoding="utf-8")).get("prices", {})
        seen = {n: p for d in done.values() for n, p in d.get("prices", {}).items()}
        print("| admin name | price | lines | proposed (confirm before use) |")
        print("| --- | --- | --- | --- |")
        for name, n in unknown.most_common():
            price = seen.get(name)
            proposal = ", ".join(propose_sku(name, price, known_prices))
            print(f"| {redact(name)} | {price or '?'} | {n} | {proposal} |")
    for line in failures[:20]:
        print("failed", line)
    if complete and not failures:
        args.cookie_file.unlink(missing_ok=True)
        print(f"export complete -> {args.out}; cookie file deleted")
        return 0
    print(f"export INCOMPLETE -> {args.out}; cookie file kept for another run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
