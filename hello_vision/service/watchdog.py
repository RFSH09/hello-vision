"""systemd watchdog / readiness notifications (``sd_notify`` without extra dependencies)."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Callable


def notify(message: str, env: dict[str, str] | None = None) -> bool:
    """Send ``message`` (e.g. ``READY=1`` or ``WATCHDOG=1``) to systemd if it is listening.

    Returns:
        True if a message was sent; False when not running under systemd (``NOTIFY_SOCKET`` unset).
    """
    addr = (env if env is not None else os.environ).get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):  # abstract socket namespace
        addr = "\0" + addr[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:  # type: ignore[attr-defined,unused-ignore]
        sock.connect(addr)
        sock.sendall(message.encode())
    return True


def watchdog_interval(env: dict[str, str] | None = None) -> float | None:
    """Half of ``WATCHDOG_USEC`` in seconds (the recommended ping period), or ``None`` if unset."""
    usec = (env if env is not None else os.environ).get("WATCHDOG_USEC")
    return int(usec) / 2_000_000 if usec else None


def start_watchdog(
    is_healthy: Callable[[], bool], stop: threading.Event, env: dict[str, str] | None = None
) -> threading.Thread | None:
    """Ping systemd every half watchdog period, but only while ``is_healthy()`` is true.

    If the service stops being healthy the pings stop and systemd restarts it.
    """
    interval = watchdog_interval(env)
    if interval is None:
        return None

    def loop() -> None:
        while not stop.wait(interval):
            if is_healthy():
                notify("WATCHDOG=1", env)

    t = threading.Thread(target=loop, name="watchdog", daemon=True)
    t.start()
    return t
