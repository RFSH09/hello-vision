"""Unattended run from an order export to the first report.

Waits for the export file, checks it, then runs import, split, download, feature extraction,
training and the report. Everything it writes is under git-ignored folders.

    uv run --extra vision python tools/run_pipeline.py --export data/export_first3000.csv \
        --report out/mil_first --sizes 1000

Progress is appended to ``data/PROGRESS.md``. Resumable: downloads and features are skipped
if they already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from hello_vision.weak.harvest import keep_awake
from hello_vision.weak.records import EXPORT_COLUMNS, SkuMapper, propose_sku


def note(progress: Path, text: str) -> None:
    """One timestamped line in the progress file (and on screen)."""
    line = f"- {datetime.now():%Y-%m-%d %H:%M}  {text}"
    print(line, flush=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wait_for(path: Path, poll_s: float, settle_s: float = 60.0) -> None:
    """Block until ``path`` exists and has stopped growing (a copy may still be running)."""
    while True:
        if path.is_file() and path.stat().st_size > 0:
            size = path.stat().st_size
            time.sleep(settle_s)
            if path.stat().st_size == size:
                return
        else:
            time.sleep(poll_s)


def read_rows(path: Path) -> list[dict[str, str]]:
    """CSV rows."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def validate(export: Path, worklist: Path, names: Path) -> dict[str, Any]:
    """Check an export against the worklist.

    Returns:
        Counts, unknown names with proposals, and a spot check of orders whose items the
        browser export had already recorded in the worklist.

    Raises:
        ValueError: Wrong columns.
    """
    rows = read_rows(export)
    if not rows or set(rows[0]) != set(EXPORT_COLUMNS):
        raise ValueError(f"{export}: expected columns {list(EXPORT_COLUMNS)} and at least one row")
    orders: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        orders.setdefault(r["order_id"], []).append(r)
    mapper = SkuMapper.load(names)
    unknown = Counter(r["item_name"] for r in rows if mapper.get(r["item_name"]) is None)
    work = {r["order_id"]: r for r in read_rows(worklist)} if worklist.is_file() else {}
    # orders the worklist has in the same date range but the export lacks (e.g. no items)
    oldest = min(r["date"][:10] for r in rows)
    newer = [i for i, w in work.items() if w["date"][:10] > oldest]
    mismatched = sum(
        1
        for i, lines in orders.items()
        if i in work
        and (
            lines[0]["fridge_key"] != work[i]["fridge_key"]
            or lines[0]["date"][:10] != work[i]["date"][:10]
        )
    )
    checked: list[str] = []
    bad: list[str] = []
    for i in sorted(orders, reverse=True):
        w = work.get(i)
        if not w or not w.get("items_json"):
            continue
        want: Counter[tuple[str, int]] = Counter(
            (n, int(q)) for n, q in json.loads(w["items_json"])
        )
        got: Counter[tuple[str, int]] = Counter((r["item_name"], int(r["qty"])) for r in orders[i])
        videos = (orders[i][0]["video0_url"], orders[i][0]["video1_url"])
        same = want == got and videos == (w["video0_url"], w["video1_url"])
        checked.append(i)
        if not same:
            bad.append(i)
        if len(checked) >= 5:
            break
    import yaml

    prices = yaml.safe_load(names.read_text(encoding="utf-8")).get("prices", {})
    return {
        "rows": len(rows),
        "orders": len(orders),
        "without_videos": sum(
            not (v[0]["video0_url"] and v[0]["video1_url"]) for v in orders.values()
        ),
        "missing_in_range": sum(i not in orders for i in newer),
        "not_in_worklist": sum(i not in work for i in orders),
        "fridge_or_date_mismatch": mismatched,
        "spot_checked": len(checked),
        "spot_mismatched": bad,
        "unknown": [
            (name, n, propose_sku(name, None, prices)) for name, n in unknown.most_common()
        ],
    }


def validation_markdown(export: Path, v: dict[str, Any]) -> str:
    """Human readable validation report."""
    lines = [
        f"# Validation of {export.name}",
        "",
        f"- rows {v['rows']}, orders {v['orders']}",
        f"- orders without both videos: {v['without_videos']}",
        f"- worklist orders newer than the oldest exported day but absent (no items?): "
        f"{v['missing_in_range']}",
        f"- orders not in the worklist: {v['not_in_worklist']}",
        f"- fridge or date differing from the worklist: {v['fridge_or_date_mismatch']}",
        f"- spot check against pre-recorded orders: {v['spot_checked']} checked, "
        f"{len(v['spot_mismatched'])} differ {v['spot_mismatched']}",
    ]
    if v["unknown"]:
        lines += [
            "",
            "Unknown product names (lines skipped). The export has no price column, so these",
            "proposals use the name only; confirm before use.",
            "",
            "| admin name | lines | proposed |",
            "| --- | --- | --- |",
            *[f"| {n} | {c} | {', '.join(p)} |" for n, c, p in v["unknown"]],
        ]
    else:
        lines.append("- unknown product names: none")
    return "\n".join(lines) + "\n"


def tool(args: Sequence[str]) -> int:
    """Run one of the repo's tools with the current interpreter."""
    return subprocess.run([sys.executable, *args], check=False).returncode  # noqa: S603


def download_and_extract(
    records: Path, videos: Path, features: Path, limit: int, min_free: float, progress: Path
) -> None:
    """Download in the background and extract features from finished orders meanwhile."""
    command = [
        sys.executable, "tools/harvest_orders.py", "download", "--records", str(records),
        "--dest", str(videos), "--limit", str(limit), "--min-free-gb", str(min_free),
    ]  # fmt: skip
    extract = [
        "tools/extract_features.py", "--records", str(videos / "orders.local.jsonl"),
        "--videos", str(videos), "--out", str(features),
    ]  # fmt: skip
    process = subprocess.Popen(command)  # noqa: S603
    reported = time.monotonic()
    while process.poll() is None:
        time.sleep(90)
        if (videos / "orders.local.jsonl").exists():
            tool(extract)
        if time.monotonic() - reported > 1800:
            reported = time.monotonic()
            done = len(list(videos.glob("*.mp4")))
            note(
                progress,
                f"downloading: {done} videos, {len(list(features.glob('*.npz')))} featured",
            )
    note(progress, f"download finished with exit code {process.returncode}")
    if tool(extract) != 0:
        note(progress, "warning: features failed for some orders (see the log)")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="*", default=[])
    p.add_argument("--history", type=Path, default=Path("data/history"))
    p.add_argument("--worklist", type=Path, default=Path("data/hv_worklist.csv"))
    p.add_argument("--names", type=Path, default=Path("configs/sku_names.yaml"))
    p.add_argument("--progress", type=Path, default=Path("data/PROGRESS.md"))
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--min-free-gb", type=float, default=13.0)
    p.add_argument("--poll", type=float, default=600.0)
    a = p.parse_args(argv)

    keep_awake()
    note(a.progress, f"waiting for {a.export.name}")
    wait_for(a.export, a.poll)
    v = validate(a.export, a.worklist, a.names)
    a.report.mkdir(parents=True, exist_ok=True)
    text = validation_markdown(a.export, v)
    (a.report / "validation.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    note(
        a.progress,
        f"{a.export.name}: {v['orders']} orders, {v['without_videos']} without videos, "
        f"{len(v['unknown'])} unknown names, spot check {v['spot_checked']} checked / "
        f"{len(v['spot_mismatched'])} differ",
    )
    wrong = len(v["spot_mismatched"]) + v["fridge_or_date_mismatch"]
    if wrong > max(1, 0.01 * v["orders"]) or len(v["spot_mismatched"]) > 1:
        note(a.progress, "STOPPED: the export disagrees with the worklist; needs a person")
        return 2

    records = a.history / "orders.jsonl"
    for step in (
        ["import-export", "--csv", str(a.export), "--names", str(a.names), "--records",
         str(records), "--skip-unknown"],
        ["split", "--records", str(records)],
    ):  # fmt: skip
        if tool(["tools/harvest_orders.py", *step]) != 0:
            note(a.progress, f"STOPPED: {step[0]} failed")
            return 1
    splits = Counter(json.loads(records.with_name("splits.json").read_text("utf-8")).values())
    note(a.progress, f"imported and split: {dict(splits)}")
    videos, features = a.history / "videos", a.history / "features"
    download_and_extract(records, videos, features, a.limit, a.min_free_gb, a.progress)
    note(a.progress, f"features ready for {len(list(features.glob('*.npz')))} orders; training")
    train = [
        "tools/train_mil.py", "--records", str(videos / "orders.local.jsonl"), "--features",
        str(features), "--videos", str(videos), "--splits", str(records.with_name("splits.json")),
        "--out", str(a.report), "--top", "10",
    ]  # fmt: skip
    if a.sizes:
        train += ["--sizes", *map(str, a.sizes)]
    if tool(train) != 0:
        note(a.progress, "STOPPED: training failed")
        return 1
    note(a.progress, f"DONE: report at {a.report / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
