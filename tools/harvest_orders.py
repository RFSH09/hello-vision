"""Collect order history for training, politely (sequential, at most 1 request per second).

    import-export  turn a server-side CSV export into records (preferred: no admin load)
    probe          save redacted copies of admin pages, to write the page parser
    split          assign train / val / test by fridge and date (before downloading)
    download       fetch videos of the usable orders (resumable, checksummed)

Everything is written under git-ignored folders. Only GET requests are ever sent.

Examples:
    python tools/harvest_orders.py import-export --csv data/export.csv
    python tools/harvest_orders.py probe --cookie-file data/admin_cookie.txt URL [URL ...]
    python tools/harvest_orders.py split --records data/history/orders.jsonl
    python tools/harvest_orders.py download --dest data/history/videos --limit 5000
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from hello_vision.weak.harvest import RateLimiter, admin_opener, download_videos, get, redact
from hello_vision.weak.records import SkuMapper, UnknownProductError, load_records, read_export
from hello_vision.weak.splits import SPLITS, select_for_download, split_orders

ADMIN_HOSTS = {"osonvend.uz", "www.osonvend.uz"}


def import_export(args: argparse.Namespace) -> int:
    """CSV export -> records.jsonl."""
    try:
        records = read_export(args.csv, SkuMapper.load(args.names))
    except UnknownProductError as exc:
        print(f"STOPPED: {exc}")
        return 2
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    days = sorted(r.date for r in records)
    print(
        f"{len(records)} orders, {len({r.fridge_key for r in records})} fridges, "
        f"{days[0] if days else '-'} to {days[-1] if days else '-'} -> {args.records}"
    )
    return 0


def probe(args: argparse.Namespace) -> int:
    """Save redacted copies of a few admin pages."""
    cookie = args.cookie_file.read_text(encoding="utf-8").strip()
    limiter = RateLimiter()
    args.out.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(args.urls):
        if urllib.parse.urlparse(url).hostname not in ADMIN_HOSTS:
            print(f"refused: {url} is not an admin address; the cookie is not sent elsewhere")
            return 2
        limiter.wait()
        with get(url, {"Cookie": cookie}, admin_opener) as response:
            html = response.read().decode("utf-8", errors="replace")
        target = args.out / f"page{i}.html"
        target.write_text(redact(html), encoding="utf-8")
        print(f"saved {target} ({len(html)} characters, numbers and e-mails masked)")
    print("Look through these files and delete any customer names before sharing them.")
    return 0


def download(args: argparse.Namespace) -> int:
    """Fetch the videos."""
    records = load_records(args.records)
    splits_file = args.records.with_name("splits.json")
    if not splits_file.exists():
        print(f"STOPPED: run `split` first ({splits_file} is missing)")
        return 2
    assignment = json.loads(splits_file.read_text(encoding="utf-8"))
    chosen = select_for_download(records, assignment, args.limit)
    counts = Counter(assignment[r.order_id] for r in chosen)
    print(f"downloading {len(chosen)} of {len(records)} orders: {dict(counts)}", flush=True)
    report = download_videos(
        chosen, args.dest, RateLimiter(args.interval), min_free_gb=args.min_free_gb
    )
    args.dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(splits_file, args.dest / "splits.json")  # train_mil looks next to the records
    print(
        f"downloaded {report.downloaded}, skipped {report.skipped}, failed {len(report.failed)}; "
        f"disk used by videos: {report.bytes_on_disk / 1e9:.1f} GB"
        + (" (stopped: low disk space)" if report.stopped_for_space else "")
    )
    for line in report.failed[:20]:
        print("  failed", line)
    return 1 if report.failed or report.stopped_for_space else 0


def split(args: argparse.Namespace) -> int:
    """Write ``splits.json`` next to the records."""
    records = load_records(args.records)
    assignment = split_orders(records, seed=args.seed)
    target = args.records.with_name("splits.json")
    target.write_text(json.dumps(assignment), encoding="utf-8")
    counts = Counter(assignment.values())
    for name in SPLITS:
        members = [r for r in records if assignment[r.order_id] == name]
        print(
            f"{name:9s} {counts[name]:6d} orders, "
            f"{len({r.fridge_key for r in members}):3d} fridges, "
            f"{len({r.date for r in members}):3d} days"
        )
    print(f"-> {target}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-export")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--names", type=Path, default=Path("configs/sku_names.yaml"))
    p.add_argument("--records", type=Path, default=Path("data/history/orders.jsonl"))
    p.set_defaults(run=import_export)
    p = sub.add_parser("probe")
    p.add_argument("urls", nargs="+")
    p.add_argument("--cookie-file", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/admin_samples"))
    p.set_defaults(run=probe)
    p = sub.add_parser("download")
    p.add_argument("--records", type=Path, default=Path("data/history/orders.jsonl"))
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=13.0,
        help="stop before free space drops below this (room for features)",
    )
    p.set_defaults(run=download)
    p = sub.add_parser("split")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=split)
    args = parser.parse_args(argv)
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
