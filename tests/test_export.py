"""Admin page parsing and the resumable export (synthetic HTML, fake server)."""

import csv
import email.message
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hello_vision.weak.admin_pages import (
    is_login_page,
    parse_item_lines,
    parse_items,
    parse_price,
    parse_videos,
)
from hello_vision.weak.harvest import RateLimiter
from hello_vision.weak.records import SkuMapper, propose_sku, read_export
from tools.finish_export import (
    FetchError,
    SessionExpiredError,
    fetch_page,
    main,
    prefilled,
    priority_order,
    run,
    write_export,
)

ORDER = """
<html><body><input value="+998901234567">
<table><tbody><tr><td>x</td><td>Not an inline</td><td>3</td></tr></tbody></table>
<div class="inline-group"><div class="tabular inline-related last-related"><table>
<thead><tr><th>Name</th><th>Quantity</th></tr></thead><tbody>
<tr class="form-row"><td class="original"><input type="hidden" value="55"><input value="9"></td>
  <td><p>Cola 0.5l</p></td><td><p>2</p></td><td><p>9 000</p></td><td><p>18 000</p></td></tr>
<tr class="form-row"><td class="original"><p>obj</p></td><td><input value="Sendvich"><br></td>
  <td><input type="number" value="1"></td><td></td></tr>
<tr class="form-row"><td>o</td><td>+998 90 123 45 67</td><td>1</td></tr>
<tr class="form-row"><td>o</td><td>Total</td><td>18 000</td></tr>
<tr class="empty-form"><td>o</td><td>Template</td><td>1</td></tr>
</tbody></table></div></div></body></html>
"""
VIDEOS = """<a href="https://v.test/a_1_1_x.mp4">cam1</a><a href="/other.pdf">x</a>
<a href="https://v.test/a_1_0_x.mp4?sig=1">cam0</a>"""
LOGIN = '<form id="login-form"><input type="password" name="password"></form>'


def test_parse_items_only_from_inline_rows_without_personal_numbers() -> None:
    assert parse_items(ORDER) == [("Cola 0.5l", 2), ("Sendvich", 1)]
    assert parse_items("<p>nothing</p>") == []


def test_parse_videos_by_camera() -> None:
    assert parse_videos(VIDEOS) == (
        "https://v.test/a_1_0_x.mp4?sig=1",
        "https://v.test/a_1_1_x.mp4",
    )
    assert parse_videos("<a href='x_1_0_.mp4'>") == ("x_1_0_.mp4", "")
    assert is_login_page(LOGIN) and not is_login_page(ORDER)


class Page(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers: dict[str, str] = {}


def error(code: int, location: str = "") -> urllib.error.HTTPError:
    headers = email.message.Message()
    if location:
        headers["Location"] = location
    return urllib.error.HTTPError("u", code, "m", headers, None)


def limiter() -> RateLimiter:
    return RateLimiter(sleep=lambda s: None)


def test_fetch_sends_get_with_cookie_only_to_the_admin() -> None:
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request) -> Page:
        seen.append(request)
        return Page(b"<p>ok</p>")

    assert (
        fetch_page("https://osonvend.uz/admin/x", "sessionid=s", limiter(), opener) == "<p>ok</p>"
    )
    assert seen[0].get_method() == "GET" and seen[0].headers["Cookie"] == "sessionid=s"
    for url in ("https://evil.test/", "http://osonvend.uz/", "https://osonvend.uz.evil.test/"):
        with pytest.raises(ValueError):
            fetch_page(url, "sessionid=s", limiter(), opener)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "answer", [error(302, "/admin/login/?next=/x"), error(403), Page(LOGIN.encode())]
)
def test_expired_session_stops(answer: object) -> None:
    def opener(request: urllib.request.Request) -> Page:
        if isinstance(answer, urllib.error.HTTPError):
            raise answer
        return Page(LOGIN.encode())

    with pytest.raises(SessionExpiredError):
        fetch_page("https://osonvend.uz/x", "sessionid=s", limiter(), opener)


def test_retries_with_backoff_then_gives_up() -> None:
    waits: list[float] = []
    calls = {"n": 0}

    def flaky(request: urllib.request.Request) -> Page:
        calls["n"] += 1
        if calls["n"] < 3:
            raise error(502)
        return Page(b"fine")

    assert fetch_page("https://osonvend.uz/x", "c", limiter(), flaky, sleep=waits.append) == "fine"
    assert waits == [2.0, 4.0]

    def down(request: urllib.request.Request) -> Page:
        raise urllib.error.URLError("offline")

    waits.clear()
    with pytest.raises(FetchError):
        fetch_page("https://osonvend.uz/x", "c", limiter(), down, sleep=waits.append)
    assert waits == [2.0, 4.0, 8.0]
    with pytest.raises(FetchError, match="404"):
        fetch_page(
            "https://osonvend.uz/x", "c", limiter(), lambda r: (_ for _ in ()).throw(error(404))
        )


def row(i: int, **kw: str) -> dict[str, str]:
    base = {"order_id": str(i), "fridge_key": f"F{i % 5}", "date": f"2026-11-{1 + i % 9:02d}",
            "order_page": f"https://osonvend.uz/order/{i}", "completed_page": f"https://osonvend.uz/done/{i}",
            "video0_url": "", "video1_url": "", "items_json": ""}  # fmt: skip
    return {**base, **kw}


def server(request: urllib.request.Request) -> Page:
    if request.full_url.endswith("/order/3"):
        return Page(b"<p>empty</p>")
    return Page((ORDER if "/order/" in request.full_url else VIDEOS).encode())


def test_run_resumes_and_export_includes_prefilled_rows(tmp_path: Path) -> None:
    done_row = row(
        1, video0_url="https://v/a", video1_url="https://v/b", items_json='[["Cola 0.5l",3]]'
    )
    rows = [done_row, row(2), row(3)]
    assert prefilled(done_row) and prefilled(rows[1]) is None
    partial = tmp_path / "p.jsonl"
    done, failures = run(rows, "c", partial, limiter(), server)
    assert sorted(done) == ["2", "3"] and not failures
    partial.write_text(partial.read_text() + '{"order_id": "half')  # interrupted write
    again, _ = run(rows, "c", partial, limiter(), lambda r: pytest.fail("must not fetch again"))
    assert sorted(again) == ["2", "3"]
    counts = write_export(rows, again, tmp_path / "export.csv")
    assert counts == {"orders": 3, "written": 2, "no_videos": 0, "no_items": 1, "lines": 3}
    records = read_export(
        tmp_path / "export.csv", SkuMapper({"Cola 0.5l": "cola", "Sendvich": "s"})
    )
    assert {r.order_id: r.items for r in records} == {"1": {"cola": 3}, "2": {"cola": 2, "s": 1}}


def test_priority_order_keeps_every_row() -> None:
    rows = [row(i) for i in range(90)]
    ordered = priority_order(rows, limit=20)
    assert sorted(r["order_id"] for r in ordered) == sorted(r["order_id"] for r in rows)
    assert priority_order(rows[:2]) == rows[:2]


def test_cli_completes_and_deletes_cookie_only_when_done(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        row(1, video0_url="https://v/a", video1_url="https://v/b", items_json='[["Mystery",1]]')
    ]
    work = tmp_path / "w.csv"
    with work.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cookie = tmp_path / "c.txt"
    cookie.write_text("wrong\n")
    args = ["--worklist", str(work), "--cookie-file", str(cookie), "--partial", str(tmp_path / "p"),
            "--out", str(tmp_path / "e.csv"), "--names", str(tmp_path / "n.yaml")]  # fmt: skip
    (tmp_path / "n.yaml").write_text("names: {}\n")
    assert main(args) == 3 and cookie.exists()
    cookie.write_text("sessionid=topsecret\n")
    assert main(args) == 0 and not cookie.exists()
    out = capsys.readouterr().out
    assert "| Mystery |" in out and "topsecret" not in out
    assert (tmp_path / "e.csv").read_text().count("\n") == 2


def test_prices_and_proposals() -> None:
    assert [parse_price(x) for x in ("9 000", "9,000.00", "14000 so'm", "2", "abc")] == [
        9000, 9000, 14000, None, None
    ]  # fmt: skip
    assert parse_item_lines(ORDER) == [("Cola 0.5l", 2, 9000), ("Sendvich", 1, None)]
    prices = {"gorilla-045": 14000, "twix-xtra-75": 14000, "lays-70": 17000}
    assert propose_sku("Twix 99g", 14000, prices)[0] == "twix-xtra-75"
    assert "lays-70" not in propose_sku("Twix 99g", 14000, prices)
    assert propose_sku("Lays x", None, prices)[0] == "lays-70"


def test_price_list_covers_every_mapped_product() -> None:
    import yaml

    data = yaml.safe_load((Path(__file__).parents[1] / "configs" / "sku_names.yaml").read_text())
    assert set(data["names"].values()) <= set(data["prices"]) and len(data["prices"]) == 30
