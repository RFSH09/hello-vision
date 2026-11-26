"""Weak supervision: records, privacy, splits, harvesting, features and metrics (synthetic)."""

import io
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from hello_vision.synth.session import write_video
from hello_vision.weak.features import dequantise, extract_session, motion_frames, quantise
from hello_vision.weak.harvest import RateLimiter, download_videos, get, redact
from hello_vision.weak.metrics import evaluate_carts, to_markdown
from hello_vision.weak.records import (
    EXPORT_COLUMNS,
    OrderRecord,
    SkuMapper,
    UnknownProductError,
    append_record,
    load_records,
    read_export,
)
from hello_vision.weak.splits import select_for_download, split_orders
from tools.harvest_orders import main as harvest_main

ROOT = Path(__file__).resolve().parents[1]


def record(i: int, fridge: str = "f1", day: int = 0, **kw: Any) -> OrderRecord:
    base = {"order_id": f"o{i}", "fridge_key": fridge, "date": date(2026, 9, 1) + timedelta(day),
            "videos": [], "items": {"a": 1}}  # fmt: skip
    return OrderRecord(**{**base, **kw})


def test_real_name_map_handles_spelling_variants() -> None:
    mapper = SkuMapper.load(ROOT / "configs" / "sku_names.yaml")
    assert mapper.get("cola.k 0.5L") == mapper.get("Cola 0.5l") == "coca-cola-05"
    assert mapper.get("  SENDVICH ") == "sandwich" and mapper.get("18+ 0.45l") == "18plus-045"
    assert mapper.cart([("Cola 0.5l", 1), ("cola.k 0.5L", 2)]) == {"coca-cola-05": 3}
    with pytest.raises(UnknownProductError) as err:
        mapper.cart([("Mystery", 1), ("Cola 0.5l", 1), ("Other thing", 1)])
    assert err.value.names == ["Mystery", "Other thing"]
    with pytest.raises(ValueError):
        mapper.cart([("Cola 0.5l", 0)])
    with pytest.raises(ValueError):
        SkuMapper({"A b": "x", "ab": "y"})


def test_records_cannot_hold_personal_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        record(1, phone="+998901234567")
    store = tmp_path / "r.jsonl"
    append_record(store, record(1))
    append_record(store, record(1, items={"b": 2}))
    assert [r.items for r in load_records(store)] == [{"b": 2}]
    assert load_records(tmp_path / "missing") == []


def write_csv(path: Path, rows: list[list[str]], columns: tuple[str, ...] = EXPORT_COLUMNS) -> Path:
    path.write_text("\n".join(",".join(r) for r in [list(columns), *rows]), encoding="utf-8")
    return path


def test_export_import_and_refusals(tmp_path: Path) -> None:
    mapper = SkuMapper({"Cola": "cola", "Bun": "bun"})
    rows = [["o1", "F1", "2026-09-01 10:00", "https://x/a.mp4", "https://x/b.mp4", "Cola", "2"],
            ["o1", "F1", "2026-09-01 10:00", "https://x/a.mp4", "https://x/b.mp4", "Bun", "1"]]  # fmt: skip
    [r] = read_export(write_csv(tmp_path / "e.csv", rows), mapper)
    assert r.items == {"cola": 2, "bun": 1} and len(r.videos) == 2 and r.date == date(2026, 9, 1)
    with pytest.raises(ValueError, match="exactly the columns"):
        read_export(write_csv(tmp_path / "p.csv", [], (*EXPORT_COLUMNS, "phone")), mapper)
    bad = write_csv(tmp_path / "u.csv", [[*rows[0][:5], "Nope", "1"]])
    with pytest.raises(UnknownProductError):
        read_export(bad, mapper)
    names = tmp_path / "n.yaml"
    names.write_text("names: {Cola: cola, Bun: bun}\n")
    out = tmp_path / "orders.jsonl"
    assert (
        harvest_main(
            [
                "import-export",
                "--csv",
                str(tmp_path / "e.csv"),
                "--names",
                str(names),
                "--records",
                str(out),
            ]
        )
        == 0
    )
    assert (
        harvest_main(
            ["import-export", "--csv", str(bad), "--names", str(names), "--records", str(out)]
        )
        == 2
    )


def test_split_never_shares_a_fridge_or_a_day() -> None:
    records = [record(i, f"f{i % 10}", (i // 10) % 30) for i in range(900)]
    assignment = split_orders(records)
    for attribute in ("fridge_key", "date"):
        seen: dict[str, set[Any]] = {}
        for r in records:
            seen.setdefault(assignment[r.order_id], set()).add(getattr(r, attribute))
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                assert a == b or not (seen[a] & seen[b]), (attribute, a, b)
    newest = max(r.date for r in records if assignment[r.order_id] == "train")
    assert newest < min(r.date for r in records if assignment[r.order_id] == "test")
    assert assignment == split_orders(records)
    with pytest.raises(ValueError):
        split_orders(records[:2])


def test_split_cli(tmp_path: Path) -> None:
    store = tmp_path / "o.jsonl"
    for i in range(60):
        append_record(store, record(i, f"f{i % 6}", i % 10))
    assert harvest_main(["split", "--records", str(store)]) == 0
    assert len(json.loads((tmp_path / "splits.json").read_text())) == 60


def test_rate_limiter_never_faster_than_one_per_second() -> None:
    now, slept = {"t": 0.0}, []

    def sleep(s: float) -> None:
        slept.append(s)
        now["t"] += s

    limiter = RateLimiter(0.01, clock=lambda: now["t"], sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert limiter.interval == 1.0 and slept == [1.0, 1.0]


def test_get_is_https_and_get_only() -> None:
    seen: list[urllib.request.Request] = []
    get("https://x/y", opener=seen.append)
    assert seen[0].get_method() == "GET"
    with pytest.raises(ValueError):
        get("http://x/y", opener=seen.append)


def test_redaction_masks_numbers_and_emails() -> None:
    text = redact("tel +998 90 123-45-67, card 8600 1234 5678 9012, a.b@mail.uz, qty 2, 57g")
    assert "123" not in text and "8600" not in text and "@" not in text
    assert "qty 2" in text and "57g" in text


def test_probe_refuses_other_hosts(tmp_path: Path) -> None:
    (tmp_path / "c").write_text("sessionid=secret")
    code = harvest_main(
        [
            "probe",
            "--cookie-file",
            str(tmp_path / "c"),
            "--out",
            str(tmp_path / "s"),
            "https://evil.example/x",
        ]
    )
    assert code == 2 and not list((tmp_path / "s").glob("*"))


class Response(io.BytesIO):
    def __init__(self, data: bytes, length: int | None = None) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}


def test_download_resumes_checksums_and_reports(tmp_path: Path) -> None:
    calls: list[str] = []

    def opener(request: urllib.request.Request) -> Response:
        calls.append(request.full_url)
        if "bad" in request.full_url:
            return Response(b"short", 999)
        return Response(b"video-bytes")

    limiter = RateLimiter(sleep=lambda s: None)
    records = [
        record(1, videos=["https://x/a", "https://x/b"]),
        record(2, videos=["https://x/bad"]),
    ]
    first = download_videos(
        records, tmp_path, limiter, opener, min_free_gb=0, progress=lambda s: None
    )
    assert (first.downloaded, first.skipped, len(first.failed)) == (2, 0, 1)
    assert "https" not in first.failed[0]
    assert [r.order_id for r in load_records(tmp_path / "orders.local.jsonl")] == ["o1"]
    again = download_videos(records[:1], tmp_path, limiter, opener, min_free_gb=0)
    assert (again.downloaded, again.skipped) == (0, 2) and again.bytes_on_disk == 22
    (tmp_path / "o1_cam0.mp4").write_bytes(b"x")  # damaged file is fetched again
    assert download_videos(records[:1], tmp_path, limiter, opener, min_free_gb=0).downloaded == 1
    full = download_videos(
        records, tmp_path, limiter, opener, min_free_gb=1e9, progress=lambda s: None
    )
    assert full.stopped_for_space


class FakeEncoder:
    def tokens(self, images: list[np.ndarray]) -> np.ndarray:
        return np.ones((len(images), 17, 8), np.float32)


def moving_video(path: Path) -> None:
    frames = []
    for i in range(40):
        f = np.full((180, 320, 3), 60, np.uint8)
        if 10 <= i < 30:  # something moves through the door zone
            f[120:160, 10 + i * 8 : 50 + i * 8] = 230
        frames.append(f)
    write_video(frames, path, fps=10)


def test_motion_filter_keeps_only_moving_moments(tmp_path: Path) -> None:
    moving_video(tmp_path / "v.mp4")
    frames = motion_frames(tmp_path / "v.mp4", fps=5)
    assert frames and all(0.9 <= f.time_s <= 3.3 for f in frames)
    x1, y1, x2, y2 = frames[2].box
    assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 180 and y2 > 100
    assert len(motion_frames(tmp_path / "v.mp4", fps=5, max_frames=3)) == 3
    with pytest.raises(FileNotFoundError):
        motion_frames(tmp_path / "no.mp4")


def test_extract_session(tmp_path: Path) -> None:
    moving_video(tmp_path / "a.mp4")
    write_video([np.full((180, 320, 3), 60, np.uint8)] * 10, tmp_path / "still.mp4", fps=10)
    n = extract_session(
        [tmp_path / "a.mp4", tmp_path / "still.mp4"], FakeEncoder(), tmp_path / "o.npz"
    )
    with np.load(tmp_path / "o.npz") as d:
        assert n == len(d["time"]) > 0 and d["full"].shape == (n, 17, 8) == d["crop"].shape
        assert set(d["cam"]) == {0} and d["box"].shape == (n, 4)
        assert d["full"].dtype == np.int8
        assert np.allclose(dequantise(d["full"], d["full_scale"]), 1.0, atol=0.01)
    assert extract_session([tmp_path / "still.mp4"], FakeEncoder(), tmp_path / "e.npz") == 0


def test_cart_metrics() -> None:
    truth = {"1": {"a": 1}, "2": {"a": 2}, "3": {"b": 1}, "4": {}}
    pred = {"1": {"a": 1}, "2": {"a": 1}, "3": {"a": 1}, "4": {}}
    r = evaluate_carts(truth, pred, ["a", "b"])
    assert r.exact == 0.5 and r.disagreements == ["2", "3"] and r.count_accuracy == 0.5
    assert r.per_sku["a"] == (2 / 3, 1.0, 2) and r.per_sku["b"] == (0.0, 0.0, 1)
    assert r.confusion[("b", "a")] == 1 and "50.0%" in to_markdown(r, "t")


def test_quantisation_is_close_and_safe_for_zeros() -> None:
    x = np.random.default_rng(0).normal(0, 3, (4, 17, 32)).astype(np.float32)
    stored, scale = quantise(x)
    assert stored.dtype == np.int8
    assert np.abs(dequantise(stored, scale) - x).max() <= scale / 2 + 1e-6
    zeros, one = quantise(np.zeros((2, 3), np.float32))
    assert one == 1.0 and not dequantise(zeros, one).any()


def test_download_selection_skips_excluded_and_keeps_held_out(tmp_path: Path) -> None:
    records = [record(i, f"f{i % 10}", (i // 10) % 30) for i in range(900)]
    assignment = split_orders(records)
    held = sum(v in ("test", "val") for v in assignment.values())
    chosen = select_for_download(records, assignment, held + 10)
    used = [assignment[r.order_id] for r in chosen]
    assert "excluded" not in used and used.count("train") == 10
    newest = max(r.date for r in records if assignment[r.order_id] == "train")
    assert all(r.date == newest for r in chosen if assignment[r.order_id] == "train")
    store = tmp_path / "o.jsonl"
    append_record(store, record(1))
    assert harvest_main(["download", "--records", str(store), "--dest", str(tmp_path / "v")]) == 2
