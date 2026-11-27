"""Polite, resumable, read-only data collection: rate limit, GET only, checksums.

Admin pages are parsed only once their layout is known (see ``redact`` and the ``probe``
command); until then orders come from a server-side CSV export.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hello_vision.weak.records import OrderRecord, append_record

Opener = Callable[[urllib.request.Request], Any]


class RateLimiter:
    """At most one request per ``interval`` seconds (never below 1 s for remote servers)."""

    def __init__(
        self,
        interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = max(1.0, interval)
        self._clock = clock
        self._sleep = sleep
        self._next = float("-inf")

    def wait(self) -> None:
        """Block until the next request is allowed."""
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


def _open(request: urllib.request.Request) -> Any:  # noqa: ANN401
    return urllib.request.urlopen(request, timeout=120)  # noqa: S310 - https checked by caller


def get(url: str, headers: dict[str, str] | None = None, opener: Opener = _open) -> Any:  # noqa: ANN401
    """One HTTPS GET (this module never sends any other method)."""
    if not url.startswith("https://"):
        raise ValueError("only https URLs are allowed")
    return opener(urllib.request.Request(url, headers=headers or {}, method="GET"))  # noqa: S310


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        return None  # never follow: a session cookie must not travel to another address


def admin_opener(request: urllib.request.Request) -> Any:  # noqa: ANN401
    """Opener for authenticated admin pages: redirects (e.g. to a login page) are errors."""
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=60)


def redact(html: str) -> str:
    """Mask e-mails and long digit runs (phone, card, account numbers) in a saved sample page.

    Names cannot be detected reliably: a person must still look over the file.
    """
    html = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", html)
    # digit runs of 7+, allowing spaces, dashes and brackets between digits
    return re.sub(r"\+?\d(?:[\s()-]?\d){6,}", "[number]", html)


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (no effect elsewhere)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


@dataclass
class DownloadReport:
    """Outcome of a download run."""

    downloaded: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    bytes_on_disk: int = 0
    stopped_for_space: bool = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_videos(
    records: Sequence[OrderRecord],
    dest: Path,
    limiter: RateLimiter,
    opener: Opener = _open,
    min_free_gb: float = 5.0,
    attempts: int = 2,
    progress: Callable[[str], None] = print,
) -> DownloadReport:
    """Download every order's videos sequentially.

    Files are written atomically; their size and SHA-256 are recorded in
    ``checksums.jsonl``. A file already recorded with a matching size is skipped, so the run
    can be interrupted and restarted. A local record file ``orders.local.jsonl`` maps orders
    to the downloaded file names. Stops before free space drops below ``min_free_gb``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    sums = dest / "checksums.jsonl"
    known: dict[str, dict[str, Any]] = {}
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            known[row["file"]] = row
    report = DownloadReport()
    for n, record in enumerate(records, start=1):
        if shutil.disk_usage(dest).free < min_free_gb * 1e9:
            report.stopped_for_space = True
            progress(f"stopping: less than {min_free_gb} GB free in {dest}")
            break
        names: list[str] = []
        complete = bool(record.videos)
        for cam, url in enumerate(record.videos):
            name = f"{record.order_id}_cam{cam}.mp4"
            path = dest / name
            row = known.get(name)
            if row and path.is_file() and path.stat().st_size == row["bytes"]:
                report.skipped += 1
                names.append(name)
                continue
            for _ in range(attempts):
                limiter.wait()
                part = path.with_suffix(".part")
                try:
                    with get(url, opener=opener) as response, part.open("wb") as fh:
                        expected = int(response.headers.get("Content-Length") or 0)
                        shutil.copyfileobj(response, fh, 1 << 20)
                    size = part.stat().st_size
                    if size == 0 or (expected and size != expected):
                        raise OSError("incomplete download")
                    part.replace(path)
                    row = {"file": name, "bytes": size, "sha256": _sha256(path)}
                    with sums.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(row) + "\n")
                    report.downloaded += 1
                    names.append(name)
                    break
                except (OSError, ValueError, urllib.error.URLError) as exc:
                    part.unlink(missing_ok=True)
                    error = type(exc).__name__
            else:
                complete = False
                report.failed.append(f"{name}: {error}")  # never log the URL
        if complete:
            append_record(dest / "orders.local.jsonl", record.model_copy(update={"videos": names}))
        if n % 50 == 0:
            progress(f"{n}/{len(records)} orders, {report.downloaded} files downloaded")
    report.bytes_on_disk = sum(p.stat().st_size for p in dest.glob("*.mp4"))
    return report
