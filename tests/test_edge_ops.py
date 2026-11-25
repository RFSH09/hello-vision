import hashlib
import json
import logging
import os
import socket
import threading
from pathlib import Path

import pytest

from hello_vision.edge.benchmark import device_info, time_callable
from hello_vision.edge.benchmark import main as bench_main
from hello_vision.edge.export import TARGETS, plan_export, rknn_script
from hello_vision.edge.ota import ModelStore, sha256_file
from hello_vision.logging_config import JsonFormatter, configure
from hello_vision.service.main import build_app
from hello_vision.service.watchdog import notify, start_watchdog, watchdog_interval


def test_time_callable_and_device_info() -> None:
    n = {"c": 0}

    def fn() -> None:
        n["c"] += 1

    r = time_callable("x", fn, runs=5, warmup=2)
    assert n["c"] == 7 and r.runs == 5 and r.fps > 0 and r.p50_ms <= r.p95_ms
    info = device_info()
    assert info["power_w"] is None and info["cost"] is None  # honest: not measurable by software


def test_benchmark_cli_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    assert bench_main(["--runs", "2", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["stages"]) == 3 and "SYNTHETIC" in data["models"]


def test_export_plans() -> None:
    m, out = Path("m/det.onnx"), Path("out")
    assert plan_export(m, "tensorrt", out)[0] == "trtexec" and "--fp16" in plan_export(
        m, "tensorrt", out
    )
    assert "--fp16" not in plan_export(m, "tensorrt", out, fp16=False)
    assert plan_export(m, "openvino", out)[0] == "ovc"
    assert plan_export(m, "rknn", out)[-1] == "rk3588"
    assert plan_export(m, "onnx-cpu", out)[0] == "python"
    assert set(TARGETS) == {"onnx-cpu", "tensorrt", "openvino", "rknn"}
    with pytest.raises(ValueError):
        plan_export(m, "tpu", out)
    assert "do_quantization=True" in rknn_script(True)


def make_file(path: Path, data: bytes) -> tuple[Path, str]:
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_model_store_install_activate_rollback(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    f1 = make_file(tmp_path / "a.onnx", b"v1")
    f2 = make_file(tmp_path / "b.onnx", b"v2")
    store.install("1.0", {"det.onnx": f1})
    assert store.current is None
    assert store.activate("1.0", lambda v: True) and store.current == "1.0"
    store.install("1.1", {"det.onnx": f2})
    assert store.path_of("1.1", "det.onnx").read_bytes() == b"v2"
    assert not store.activate("1.1", lambda v: False) and store.current == "1.0"  # health failed
    assert store.activate("1.1", lambda v: True) and store.current == "1.1"
    assert store.rollback() == "1.0" and store.current == "1.0"
    assert sha256_file(f1[0]) == f1[1]


def test_model_store_rejects_bad_installs(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    good = make_file(tmp_path / "a.onnx", b"x")
    with pytest.raises(ValueError, match="checksum"):
        store.install("2.0", {"det.onnx": (good[0], "0" * 64)})
    assert not (tmp_path / "models" / "versions" / "2.0").exists()  # nothing half-installed
    with pytest.raises(ValueError, match="unsafe"):
        store.install("../x", {"d": good})
    store.install("2.0", {"det.onnx": good})
    with pytest.raises(ValueError, match="already"):
        store.install("2.0", {"det.onnx": good})
    with pytest.raises(ValueError, match="not installed"):
        store.activate("9.9", lambda v: True)
    with pytest.raises(RuntimeError):
        store.rollback()


def test_json_logging(capsys: pytest.CaptureFixture[str]) -> None:
    configure()
    logging.getLogger("hello_vision").info("hello", extra={"session_id": "s1"})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["msg"] == "hello" and rec["session_id"] == "s1" and rec["level"] == "INFO"
    try:
        raise ValueError("boom")
    except ValueError:
        r = logging.LogRecord("x", logging.ERROR, "f", 1, "bad", None, __import__("sys").exc_info())
    assert "boom" in json.loads(JsonFormatter().format(r))["exc"]


def test_watchdog_helpers(tmp_path: Path) -> None:
    assert notify("READY=1", {}) is False
    assert (
        watchdog_interval({}) is None and watchdog_interval({"WATCHDOG_USEC": "60000000"}) == 30.0
    )
    if not hasattr(socket, "AF_UNIX") or os.name == "nt":
        pytest.skip("AF_UNIX datagram sockets are used by systemd on Linux")
    path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    srv.settimeout(2)
    assert notify("WATCHDOG=1", {"NOTIFY_SOCKET": path}) and srv.recv(64) == b"WATCHDOG=1"


def test_watchdog_thread_only_when_configured() -> None:
    stop = threading.Event()
    assert start_watchdog(lambda: True, stop, {}) is None
    t = start_watchdog(lambda: True, stop, {"WATCHDOG_USEC": "2000"})
    assert t is not None
    stop.set()
    t.join(2)


def test_build_app_refuses_without_face_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for k, v in {
        "VISION_FRIDGE_ID": "f", "VISION_BACKEND_URL": "http://x", "VISION_HMAC_SECRET": "s",
        "VISION_SESSIONS_ROOT": str(tmp_path), "VISION_EVIDENCE_ROOT": str(tmp_path / "e"),
        "VISION_QUEUE_ROOT": str(tmp_path / "q"),
    }.items():  # fmt: skip
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("VISION_FACE_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="VISION_FACE_MODEL"):
        build_app()
