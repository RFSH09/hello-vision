"""pre-push guard: the whole history must have one author and no attribution trailers.

Equivalent to eyeballing ``git log --format='%an <%ae>%n%b' | sort -u`` before each push.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

from check_commit_msg import find_violations


def _git_log(fmt: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", "log", f"--format={fmt}"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def audit(identities: Sequence[str], messages: str) -> list[str]:
    """Return the problems found in a history.

    Args:
        identities: One ``"Name <email>"`` string per commit.
        messages: Concatenated full commit messages.

    Returns:
        Human-readable problems (empty if the history is clean).
    """
    problems: list[str] = []
    unique = sorted(set(identities))
    if len(unique) > 1:
        problems.append("more than one author identity: " + "; ".join(unique))
    problems.extend(f"attribution text in history: {v!r}" for v in find_violations(messages))
    return problems


def main() -> int:
    """CLI entry point; prints the unique identities and fails on any problem."""
    identities = [ln for ln in _git_log("%an <%ae>").splitlines() if ln.strip()]
    messages = _git_log("%B")
    for ident in sorted(set(identities)):
        print(f"author: {ident}")
    problems = audit(identities, messages)
    for p in problems:
        print(f"PUSH BLOCKED: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
