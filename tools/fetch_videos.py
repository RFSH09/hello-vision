"""Download the videos listed in a private ``manifest.json`` into the same private folder.

Manifest: ``[{"video_id": ..., "files": {"<name>.mp4": "<https url>"}, ...}]``. Existing files
are skipped, each failure is retried once, and sizes and failures are reported. URLs are never
printed (they point at private recordings).

Usage: ``uv run python tools/fetch_videos.py --manifest data/videos/manifest.json``
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def fetch(url: str, dest: Path, attempts: int = 2) -> tuple[str, int, str]:
    """Download ``url`` to ``dest`` atomically.

    Returns:
        ``(status, size in bytes, error)`` with status ``downloaded``, ``skipped`` or ``failed``.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        return "skipped", dest.stat().st_size, ""
    if not url.startswith("https://"):
        return "failed", 0, "not an https URL"
    part = dest.with_name(dest.name + ".part")
    error = ""
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as fh:  # noqa: S310
                expected = int(r.headers.get("Content-Length") or 0)
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            size = part.stat().st_size
            if size == 0 or (expected and size != expected):
                raise OSError(f"incomplete download: {size} of {expected} bytes")
            part.replace(dest)
            return "downloaded", size, ""
        except (OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
    part.unlink(missing_ok=True)
    return "failed", 0, error


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exit code 1 if any file failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/videos/manifest.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    folder = args.manifest.parent
    jobs: list[tuple[str, str]] = []
    for order in json.loads(args.manifest.read_text(encoding="utf-8")):
        for name, url in order["files"].items():
            if Path(name).name != name:
                raise SystemExit(f"unsafe file name in manifest: {name!r}")
            jobs.append((name, url))
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: (j[0], *fetch(j[1], folder / j[0])), jobs))
    for name, status, size, error in results:
        print(f"{status:10s} {size / 1e6:7.1f} MB  {name}  {error}")
    failed = [r for r in results if r[1] == "failed"]
    total = sum(r[2] for r in results) / 1e6
    print(f"\n{len(results) - len(failed)}/{len(results)} ok, {total:.0f} MB, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
