"""commit-msg guard: reject AI/tool attribution lines in commit messages."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

FORBIDDEN = re.compile(
    r"^\s*co-authored-by:|generated with|noreply@anthropic\.com|\U0001f916",
    re.IGNORECASE | re.MULTILINE,
)


def find_violations(message: str) -> list[str]:
    """Return the offending fragments found in ``message``.

    Args:
        message: Full commit message text.

    Returns:
        The matched fragments (empty if the message is clean).
    """
    return [m.group(0).strip() for m in FORBIDDEN.finditer(message)]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``check_commit_msg.py <commit-msg-file>``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: check_commit_msg.py <commit-msg-file>", file=sys.stderr)
        return 2
    violations = find_violations(Path(args[0]).read_text(encoding="utf-8", errors="replace"))
    for v in violations:
        print(f"Forbidden attribution in commit message: {v!r}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
