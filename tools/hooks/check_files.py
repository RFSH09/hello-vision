"""Pre-commit guard: refuse files that must never reach the public repository.

Blocks real media, model weights, credentials, anything under ``data/`` (and a few
sibling private directories), and files above a size limit.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import PurePosixPath

BLOCKED_SUFFIXES = frozenset(
    {
        # media
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".heic",
        # weights / engines / arrays
        ".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".engine", ".trt", ".rknn",
        ".npz", ".npy", ".faiss",
        # secrets
        ".pem", ".key", ".p12", ".pfx", ".crt", ".token",
        # runtime databases
        ".sqlite", ".sqlite3", ".db",
    }
)  # fmt: skip
BLOCKED_TOP_DIRS = frozenset({"data", "datasets", "private", "secrets", "evidence"})
ENV_ALLOWED = frozenset({".env.example"})


def check_path(path: str, size: int | None, max_bytes: int) -> list[str]:
    r"""Return the reasons ``path`` must not be committed (empty list if it is fine).

    Args:
        path: Repo-relative path using ``/`` or ``\\`` separators.
        size: File size in bytes, or ``None`` if the file does not exist on disk.
        max_bytes: Maximum allowed size in bytes.

    Returns:
        A list of human-readable problems.
    """
    problems: list[str] = []
    p = PurePosixPath(path.replace("\\", "/"))
    if p.parts and p.parts[0] in BLOCKED_TOP_DIRS:
        problems.append(f"under private directory '{p.parts[0]}/'")
    if p.suffix.lower() in BLOCKED_SUFFIXES:
        problems.append(f"blocked file type '{p.suffix.lower()}'")
    name = p.name.lower()
    if (name == ".env" or name.startswith(".env.")) and name not in ENV_ALLOWED:
        problems.append("environment file")
    if size is not None and size > max_bytes:
        problems.append(f"{size // 1024} KB exceeds limit of {max_bytes // 1024} KB")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a non-zero exit code if any file is rejected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-kb", type=int, default=512)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    failed = False
    for f in args.files:
        size = os.path.getsize(f) if os.path.isfile(f) else None
        for problem in check_path(f, size, args.max_kb * 1024):
            print(f"BLOCKED {f}: {problem}", file=sys.stderr)
            failed = True
    if failed:
        print(
            "Real data, media, weights and secrets belong in private storage, not git "
            "(docs/PRIVACY.md).",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
