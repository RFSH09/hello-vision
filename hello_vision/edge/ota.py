"""Model store with checksum-verified installs, atomic activation and rollback.

Layout under ``root``::

    versions/<version>/<files...>
    current.json      {"current": "<version>", "previous": "<version> | null"}

The running service reports ``current`` as ``model_version`` in every result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

_VERSION = re.compile(r"^[A-Za-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    """Versioned model directory with rollback."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "versions").mkdir(parents=True, exist_ok=True)

    def _state(self) -> dict[str, str | None]:
        p = self.root / "current.json"
        if not p.exists():
            return {"current": None, "previous": None}
        data: dict[str, str | None] = json.loads(p.read_text(encoding="utf-8"))
        return data

    def _save(self, state: dict[str, str | None]) -> None:
        tmp = self.root / "current.json.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.root / "current.json")  # atomic switch

    @property
    def current(self) -> str | None:
        """The active model version, if any."""
        return self._state()["current"]

    def path_of(self, version: str, name: str) -> Path:
        """Path of a file inside an installed version."""
        return self.root / "versions" / version / name

    def install(self, version: str, files: dict[str, tuple[Path, str]]) -> None:
        """Copy model files into a new version after verifying their SHA-256.

        Args:
            version: New version label (letters, digits, ``._-``).
            files: ``{target name: (source path, expected sha256)}``.

        Raises:
            ValueError: Unsafe or already installed version, or a checksum mismatch (nothing is
                installed in that case).
        """
        if not _VERSION.match(version):
            raise ValueError(f"unsafe version {version!r}")
        dest = self.root / "versions" / version
        if dest.exists():
            raise ValueError(f"version {version!r} already installed")
        for name, (src, digest) in files.items():
            if sha256_file(src) != digest:
                raise ValueError(f"checksum mismatch for {name}")
        tmp = dest.with_name(dest.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name, (src, _) in files.items():
            shutil.copy2(src, tmp / name)
        os.replace(tmp, dest)

    def activate(self, version: str, health_check: Callable[[str], bool]) -> bool:
        """Switch to ``version``; roll back automatically if ``health_check`` fails.

        Returns:
            True if the new version is active, False if the previous version was restored.
        """
        if not (self.root / "versions" / version).is_dir():
            raise ValueError(f"version {version!r} is not installed")
        state = self._state()
        self._save({"current": version, "previous": state["current"]})
        if health_check(version):
            return True
        self._save(state)
        return False

    def rollback(self) -> str:
        """Switch back to the previous version and return it.

        Raises:
            RuntimeError: If there is no previous version.
        """
        state = self._state()
        if not state["previous"]:
            raise RuntimeError("no previous version to roll back to")
        self._save({"current": state["previous"], "previous": state["current"]})
        return state["previous"]
