"""Fail if an installed Python package breaks the project's license policy.

Policy (docs/LICENSES.md): permissive licenses are allowed; MPL-2.0 only for the listed,
unmodified third-party packages; GPL, AGPL, SSPL and non-commercial licenses are forbidden;
anything else needs a human decision. Reads package metadata only (not a legal review).

Usage: ``uv run python tools/check_licenses.py``
"""

from __future__ import annotations

import re
from importlib import metadata

MPL_EXCEPTIONS = frozenset({"certifi", "orjson", "pathspec", "tqdm"})
PERMISSIVE = re.compile(
    r"\b(MIT|BSD|Apache|PSF|Python Software Foundation|HPND|ISC|Zlib|0BSD|Unlicense|CC0)",
    re.IGNORECASE,
)
# \bGPL does not match "LGPL"
FORBIDDEN = re.compile(
    r"\bA?GPL|Affero|\bGNU General Public|SSPL|Non-?Commercial|CC-BY-NC", re.IGNORECASE
)
MPL = re.compile(r"\bMPL|Mozilla Public", re.IGNORECASE)


def license_text(expression: str | None, classifiers: list[str], field: str | None) -> str:
    """The most reliable license statement a package gives (SPDX, classifiers, free text)."""
    if expression:
        return expression
    from_classifiers = [
        c.split("::")[-1].strip() for c in classifiers if c.startswith("License ::")
    ]
    useful = [c for c in from_classifiers if c != "OSI Approved"]
    if useful:
        return " AND ".join(useful)
    return (field or "").strip()[:200]


def verdict(name: str, text: str) -> str:
    """``ok``, ``exception`` (accepted MPL), ``forbidden`` or ``review``."""
    if FORBIDDEN.search(text):
        return "forbidden"
    if MPL.search(text):
        return "exception" if name.lower() in MPL_EXCEPTIONS else "review"
    return "ok" if PERMISSIVE.search(text) else "review"


def main() -> int:
    """Check every installed distribution; exit code 1 on ``forbidden`` or ``review``."""
    failed = False
    count = 0
    for dist in metadata.distributions():
        meta = dist.metadata
        name = str(meta["Name"])
        text = license_text(
            (meta.get_all("License-Expression") or [None])[0],
            meta.get_all("Classifier") or [],
            (meta.get_all("License") or [None])[0],
        )
        result = verdict(name, text)
        count += 1
        if result != "ok":
            print(f"{result:9s} {name}: {text.splitlines()[0] if text else '(none stated)'}")
        failed = failed or result in ("forbidden", "review")
    print(f"{count} packages checked: {'FAILED' if failed else 'policy satisfied'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
