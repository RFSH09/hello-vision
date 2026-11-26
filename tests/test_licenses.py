import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from check_licenses import license_text, main, verdict


def test_verdicts() -> None:
    assert verdict("x", "MIT License") == "ok"
    assert verdict("x", "BSD-3-Clause AND 0BSD") == "ok"
    assert verdict("certifi", "Mozilla Public License 2.0 (MPL 2.0)") == "exception"
    assert verdict("other", "MPL-2.0") == "review"
    assert verdict("x", "GNU Affero General Public License v3") == "forbidden"
    assert verdict("x", "AGPL-3.0") == "forbidden"
    assert verdict("x", "GPL-2.0-only") == "forbidden"
    assert verdict("x", "LGPL-2.1") == "review"
    assert verdict("x", "") == "review"


def test_license_text_priority() -> None:
    classifiers = ["License :: OSI Approved :: MIT License"]
    assert license_text("Apache-2.0", classifiers, "x") == "Apache-2.0"
    assert license_text(None, classifiers, "x") == "MIT License"
    assert license_text(None, [], " BSD ") == "BSD"


def test_installed_environment_satisfies_the_policy() -> None:
    assert main() == 0
