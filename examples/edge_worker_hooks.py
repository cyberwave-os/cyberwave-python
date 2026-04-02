"""Edge worker hook system — interactive live dashboard.

Simulates a robot streaming sensor data and renders a non-scrolling
terminal dashboard that updates in place.  Press **q + Enter** or
**Ctrl-C** to stop cleanly.

Concepts covered
----------------
* Single-channel hooks — ``@cw.on_joint_states``, ``@cw.on_imu``
* Multi-channel synchronized hook — ``@cw.on_synchronized``
* Generic custom-channel hook — ``@cw.on_data``
* ``HookContext`` fields: ``channel``, ``twin_uuid``, ``timestamp``
* ``cw.data.latest()`` called from inside a hook callback
* Drop-oldest back-pressure: IMU publishes at 20 Hz; the hook sleeps
  40 ms so roughly every other sample is dropped — visible in the Hz
  column of the dashboard

Publisher schedule
------------------
  joint_states   10 Hz — sinusoidal joint angles
  imu            20 Hz — gravity vector + small noise
  diagnostics     1 Hz — CPU / uptime counters

In-process broker
-----------------
An in-process Zenoh router is started so no external daemon or
multicast discovery is required.  Works on WSL2, containers, and Linux.
See ``zenoh_fanout.py`` for a detailed explanation.

Usage::

    python examples/edge_worker_hooks.py
"""

from __future__ import annotations

import atexit
import json
import math
import re
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import zenoh

from cyberwave.data.api import DataBus
from cyberwave.data.backend import Sample
from cyberwave.data.header import decode as wire_decode
from cyberwave.data.zenoh_backend import ZenohBackend
from cyberwave.workers.context import HookContext
from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.runtime import WorkerRuntime

# ── Constants ─────────────────────────────────────────────────────────

TWIN_UUID = "00000000-0000-0000-0000-000000000099"
DASHBOARD_WIDTH = 70
DASHBOARD_ROWS = 22

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ── ANSI helpers ──────────────────────────────────────────────────────

_A = {
    "home": "\033[H",
    "clear": "\033[2J",
    "hide_cur": "\033[?25l",
    "show_cur": "\033[?25h",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}

_cursor_hidden = False


def _restore_cursor() -> None:
    """Ensure the cursor is visible on exit, however the process ends."""
    if _cursor_hidden:
        sys.stdout.write(_A["show_cur"])
        sys.stdout.flush()


atexit.register(_restore_cursor)


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _line(text: str = "", width: int = DASHBOARD_WIDTH) -> str:
    """Pad or truncate *text* to exactly *width* visible characters."""
    vis = _visible_len(text)
    if vis > width:
        # Truncate: walk the string keeping visible char count
        out: list[str] = []
        count = 0
        i = 0
        while i < len(text) and count < width - 1:
            m = _ANSI_RE.match(text, i)
            if m:
                out.append(m.group())
                i = m.end()
            else:
                out.append(text[i])
                count += 1
                i += 1
        out.append(_A["reset"] + "…")
        return "".join(out) + "\n"
    return text + " " * (width - vis) + "\n"


def _rule(width: int = DASHBOARD_WIDTH) -> str:
    return _line("  " + "─" * (width - 4), width)


# ── Shared live state ──────────────────────────────────────────────────


@dataclass
class _HookSnap:
    """Immutable snapshot of one hook's stats for the renderer."""

    count: int
    hz: str
    last_value: str
    stale: bool


@dataclass
class _HookStats:
    """Per-hook counters and last-value string, updated by hook callbacks."""

    count: int = 0
    last_ts: float = 0.0
    last_value: str = "—"
    _recent: deque[float] = field(
        default_factory=lambda: deque(maxlen=30), repr=False
    )

    def record(self, value: str) -> None:
        now = time.time()
        self.count += 1
        self.last_ts = now
        self.last_value = value
        self._recent.append(now)

    def snapshot(self) -> _HookSnap:
        """Capture a consistent point-in-time snapshot (call under lock)."""
        now = time.time()
        if len(self._recent) >= 2:
            span = self._recent[-1] - self._recent[0]
            hz = f"{(len(self._recent) - 1) / span:4.1f}" if span > 0 else " —"
        else:
            hz = " —"
        age = now - self.last_ts if self.last_ts else 999.0
        return _HookSnap(
            count=self.count,
            hz=hz,
            last_value=self.last_value,
            stale=age > 2.0,
        )


@dataclass
class _State:
    start: float = field(default_factory=time.time)
    hooks: dict[str, _HookStats] = field(default_factory=dict)
    events: list[tuple[float, str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def touch(self, name: str, value: str) -> None:
        with self.lock:
            if name not in self.hooks:
                self.hooks[name] = _HookStats()
            self.hooks[name].record(value)

    def emit(self, name: str, detail: str) -> None:
        with self.lock:
            self.events.append((time.time(), name, detail))
            if len(self.events) > 5:
                self.events.pop(0)

    def snapshot(self) -> tuple[
        float, dict[str, _HookSnap], list[tuple[float, str, str]]
    ]:
        """Capture a consistent snapshot of all state under the lock."""
        with self.lock:
            uptime = time.time() - self.start
            hooks = {k: v.snapshot() for k, v in self.hooks.items()}
            events = list(self.events)
        return uptime, hooks, events


# ── Dashboard renderer ────────────────────────────────────────────────


HOOK_ORDER = [
    "on_joint_states",
    "on_imu  (slow cb)",
    "on_synchronized",
    "on_data/diagnostics",
]


def _render_loop(state: _State, stop: threading.Event) -> None:
    """Redraw the dashboard at ~10 Hz until *stop* is set."""
    global _cursor_hidden
    W = DASHBOARD_WIDTH
    a = _A

    sys.stdout.write(a["hide_cur"] + a["clear"])
    sys.stdout.flush()
    _cursor_hidden = True

    try:
        while not stop.is_set():
            uptime, hooks_snap, events_snap = state.snapshot()
            now = time.time()

            out = a["home"]

            # Header
            out += _line()
            out += _line(
                f"{a['bold']}{a['cyan']}  Edge Worker Hooks  —  "
                f"live dashboard{a['reset']}",
                W,
            )
            out += _line(
                f"  {a['dim']}Press  q + Enter  or  Ctrl-C  "
                f"to stop{a['reset']}",
                W,
            )
            out += _line()

            # Hook table
            out += _line(
                f"  {a['bold']}{'HOOK':<22}{'COUNT':>7}{'Hz':>6}  "
                f"{'LAST VALUE'}{a['reset']}",
                W,
            )
            out += _rule(W)

            for name in HOOK_ORDER:
                if name in hooks_snap:
                    snap = hooks_snap[name]
                    color = a["dim"] if snap.stale else a["green"]
                    out += _line(
                        f"  {color}{name:<22}{a['reset']}"
                        f"{snap.count:>7}"
                        f"{snap.hz:>6}"
                        f"  {snap.last_value}",
                        W,
                    )
                else:
                    out += _line(
                        f"  {a['dim']}{name:<22}"
                        f"{'—':>7}{'—':>6}  waiting…{a['reset']}",
                        W,
                    )

            out += _line()

            # Events
            out += _line(f"  {a['bold']}EVENTS{a['reset']}", W)
            out += _rule(W)
            if events_snap:
                for ts, ename, detail in events_snap:
                    age_s = now - ts
                    color = a["yellow"] if age_s < 3 else a["dim"]
                    out += _line(
                        f"  {color}↑ {ename:<22}{a['reset']} {detail}", W
                    )
                for _ in range(5 - len(events_snap)):
                    out += _line()
            else:
                out += _line(f"  {a['dim']}none yet{a['reset']}", W)
                for _ in range(4):
                    out += _line()

            out += _line()

            # Status bar
            status = "running" if not stop.is_set() else "stopped"
            out += _line(
                f"  {a['dim']}uptime {uptime:5.0f}s   "
                f"runtime: {status}{a['reset']}",
                W,
            )

            sys.stdout.write(out)
            sys.stdout.flush()
            time.sleep(0.1)

    finally:
        sys.stdout.write(a["show_cur"])
        sys.stdout.flush()
        _cursor_hidden = False


# ── Infrastructure helpers ─────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_broker() -> tuple[Any, list[str]]:
    port = _find_free_port()
    endpoints = [f"tcp/127.0.0.1:{port}"]
    cfg = zenoh.Config()
    cfg.insert_json5("listen/endpoints", json.dumps(endpoints))
    cfg.insert_json5("transport/shared_memory/enabled", "false")
    return zenoh.open(cfg), endpoints


def _make_backend(endpoints: list[str]) -> ZenohBackend:
    return ZenohBackend(connect=endpoints, shared_memory=False)


def _decode_json(payload: bytes) -> Any:
    _, raw = wire_decode(payload)
    return json.loads(raw)


# ── Minimal Cyberwave client stand-in ─────────────────────────────────


class _Config:
    def __init__(self, twin_uuid: str) -> None:
        self.twin_uuid = twin_uuid


class _FakeCw:
    def __init__(self, data_bus: DataBus, twin_uuid: str, state: _State) -> None:
        self._hook_registry = HookRegistry()
        self._data_bus = data_bus
        self._state = state
        self.config = _Config(twin_uuid)

    @property
    def data(self) -> DataBus:
        return self._data_bus

    @property
    def on_joint_states(self):  # type: ignore[return]
        return self._hook_registry.on_joint_states

    @property
    def on_imu(self):  # type: ignore[return]
        return self._hook_registry.on_imu

    @property
    def on_data(self):  # type: ignore[return]
        return self._hook_registry.on_data

    @property
    def on_synchronized(self):  # type: ignore[return]
        return self._hook_registry.on_synchronized

    def publish_event(
        self,
        twin_uuid: str,
        event_type: str,
        data: dict[str, Any],
        *,
        source: str = "edge_node",
    ) -> None:
        detail = "  ".join(f"{k}={v}" for k, v in data.items())
        self._state.emit(event_type, detail)


# ── Sensor publisher ──────────────────────────────────────────────────


def _publisher(data_bus: DataBus, stop: threading.Event) -> None:
    t0 = time.time()
    imu_seq = diag_seq = 0

    JOINT_PERIOD = 1.0 / 10  # 10 Hz
    IMU_PERIOD = 1.0 / 20  # 20 Hz
    DIAG_PERIOD = 1.0  # 1 Hz

    last_joint = t0
    last_imu = t0
    last_diag = t0

    while not stop.is_set():
        now = time.time()
        t = now - t0

        if now - last_joint >= JOINT_PERIOD:
            last_joint = now
            data_bus.publish(
                "joint_states",
                {
                    "q1": round(math.sin(t * 0.5) * 1.57, 3),
                    "q2": round(math.cos(t * 0.3) * 0.8, 3),
                    "q3": round(math.sin(t * 0.7) * 0.4, 3),
                    "seq": int(t * 10),
                },
            )

        if now - last_imu >= IMU_PERIOD:
            last_imu = now
            noise = (imu_seq % 7 - 3) * 0.01
            data_bus.publish(
                "imu",
                {
                    "ax": round(noise, 3),
                    "ay": round(noise * 0.5, 3),
                    "az": round(9.81 + noise * 0.1, 3),
                    "seq": imu_seq,
                },
            )
            imu_seq += 1

        if now - last_diag >= DIAG_PERIOD:
            last_diag = now
            data_bus.publish(
                "diagnostics",
                {
                    "uptime_s": round(t, 1),
                    "cpu_pct": round(20 + (diag_seq % 5) * 3.1, 1),
                },
            )
            diag_seq += 1

        time.sleep(0.02)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    stop = threading.Event()
    state = _State()

    # Infrastructure
    _broker, endpoints = _start_broker()
    backend = _make_backend(endpoints)
    data_bus = DataBus(backend, TWIN_UUID)
    cw = _FakeCw(data_bus, TWIN_UUID, state)

    # ── Hook definitions ──────────────────────────────────────────────

    @cw.on_joint_states(TWIN_UUID)
    def on_joints(payload: bytes, ctx: HookContext) -> None:
        j = _decode_json(payload)
        state.touch(
            "on_joint_states",
            f"q1={j['q1']:+.3f}  q2={j['q2']:+.3f}  q3={j['q3']:+.3f}",
        )

    @cw.on_imu(TWIN_UUID)
    def on_imu_data(payload: bytes, ctx: HookContext) -> None:
        imu = _decode_json(payload)
        time.sleep(0.04)  # simulate 40 ms inference — triggers drop-oldest
        latest_j = cw.data.latest("joint_states", timeout_s=0.1)
        extra = f"  (joints q1={latest_j['q1']:+.3f})" if latest_j else ""
        state.touch("on_imu  (slow cb)", f"az={imu['az']:.3f} m/s²{extra}")

    @cw.on_synchronized(TWIN_UUID, ["joint_states", "imu"], tolerance_ms=150)
    def on_synced(samples: dict[str, Sample], ctx: HookContext) -> None:
        state.touch("on_synchronized", "both channels within 150 ms")

    @cw.on_data(TWIN_UUID, "diagnostics")
    def on_diagnostics(payload: bytes, ctx: HookContext) -> None:
        d = _decode_json(payload)
        state.touch(
            "on_data/diagnostics",
            f"uptime={d['uptime_s']}s  cpu={d['cpu_pct']}%",
        )
        # Simulated CPU threshold: the publisher cycles through 20→23→26→29→32 %
        # so this fires roughly twice per 5-second cycle to demonstrate publish_event.
        if d["cpu_pct"] > 28:
            cw.publish_event(
                TWIN_UUID,
                "high_cpu (simulated)",
                {"cpu_pct": d["cpu_pct"], "uptime_s": d["uptime_s"]},
            )

    # ── Start runtime + publisher ─────────────────────────────────────

    runtime = WorkerRuntime(cw)
    runtime.start()
    time.sleep(0.15)

    threading.Thread(
        target=_publisher, args=(data_bus, stop), daemon=True
    ).start()

    # ── Renderer ──────────────────────────────────────────────────────

    threading.Thread(
        target=_render_loop, args=(state, stop), daemon=True
    ).start()

    # ── Signal + keyboard-quit watcher ───────────────────────────────

    def _on_sigint(signum: int, frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _on_sigint)

    def _watch_keys() -> None:
        for line in sys.stdin:
            if line.strip().lower() == "q":
                stop.set()
                break

    threading.Thread(target=_watch_keys, daemon=True).start()

    # ── Block until stop ──────────────────────────────────────────────

    while not stop.is_set():
        time.sleep(0.25)

    # ── Shutdown ──────────────────────────────────────────────────────

    runtime.stop()
    backend.close()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print(f"\033[{DASHBOARD_ROWS}B")
    with state.lock:
        print("\n=== Final counts ===")
        for name, stats in state.hooks.items():
            print(f"  {name:<24} {stats.count:>5} dispatches")
    print()


if __name__ == "__main__":
    main()
