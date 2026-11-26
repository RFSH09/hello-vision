"""Unattended pipeline: export validation and incomplete-order handling (synthetic)."""

import csv
import json
import threading
import time
from pathlib import Path

import pytest

from hello_vision.weak.records import EXPORT_COLUMNS, SkuMapper, read_export
from tools.run_pipeline import note, validate, validation_markdown, wait_for


def write(path: Path, columns: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def line(
    order: str, name: str, qty: int = 1, day: str = "2026-11-20", v1: str = "https://v/b"
) -> list[str]:
    return [order, "F1", day, "https://v/a", v1, name, str(qty)]


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path, Path]:
    export = write(tmp_path / "e.csv", list(EXPORT_COLUMNS), [
        line("3", "Cola"), line("3", "Mystery bar"), line("2", "Cola", 2),
        line("1", "Mystery bar", v1="", day="2026-11-19"),
    ])  # fmt: skip
    columns = ["order_id", "fridge_key", "date", "order_page", "completed_page", "video0_url",
               "video1_url", "items_json"]  # fmt: skip

    def work(i: str, day: str, items: object) -> list[str]:
        text = json.dumps(items) if items else ""
        videos = ["https://v/a", "https://v/b"] if items else ["", ""]
        return [i, "F1", day, "p", "q", *videos, text]

    worklist = write(tmp_path / "w.csv", columns, [
        work("3", "2026-11-20", [["Cola", 1], ["Mystery bar", 1]]), work("2", "2026-11-20", [["Cola", 1]]),
        work("1", "2026-11-19", None), work("9", "2026-11-20", None),
    ])  # fmt: skip
    names = tmp_path / "n.yaml"
    names.write_text("names: {Cola: cola}\nprices: {cola: 9000, mystery-x: 5000}\n")
    return export, worklist, names


def test_validation_counts_spot_check_and_unknown_names(files: tuple[Path, Path, Path]) -> None:
    export, worklist, names = files
    v = validate(export, worklist, names)
    assert (v["orders"], v["rows"], v["without_videos"]) == (3, 4, 1)
    assert v["missing_in_range"] == 1 and v["not_in_worklist"] == 0
    assert v["spot_checked"] == 2 and v["spot_mismatched"] == ["2"]  # quantity differs
    [(name, count, proposal)] = v["unknown"]
    assert name == "Mystery bar" and count == 2 and "mystery-x" in proposal
    assert "| Mystery bar | 2 |" in validation_markdown(export, v)
    with pytest.raises(ValueError):
        validate(write(export.with_name("bad.csv"), ["a"], [["1"]]), worklist, names)


def test_skipping_unknown_lines_marks_the_order_incomplete(files: tuple[Path, Path, Path]) -> None:
    export, _, _ = files
    records = {r.order_id: r for r in read_export(export, SkuMapper({"Cola": "cola"}), True)}
    assert sorted(records) == ["2", "3"]  # order 1 had no known line left
    assert records["3"].items == {"cola": 1} and records["3"].skipped_lines == 1
    assert records["2"].skipped_lines == 0


def test_wait_for_waits_until_the_file_stops_growing(tmp_path: Path) -> None:
    target = tmp_path / "x.csv"

    def writer() -> None:
        time.sleep(0.05)
        target.write_text("a")
        time.sleep(0.02)
        target.write_text("ab")

    t = threading.Thread(target=writer)
    t.start()
    wait_for(target, poll_s=0.01, settle_s=0.15)
    t.join()
    assert target.read_text() == "ab"


def test_note_appends(tmp_path: Path) -> None:
    note(tmp_path / "P.md", "one")
    note(tmp_path / "P.md", "two")
    assert (tmp_path / "P.md").read_text().count("\n") == 2
