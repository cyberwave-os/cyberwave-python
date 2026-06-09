"""
Edge Worker Hooks — demonstrates all hook types with simulated sensor data.

Covers: @cw.on_joint_states, @cw.on_imu, @cw.on_synchronized, @cw.on_data.
Uses an in-process Zenoh broker — no external dependencies.

Requirements:
    pip install cyberwave[zenoh]
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Any

import zenoh

from cyberwave.data.api import DataBus
from cyberwave.data.header import decode as wire_decode
from cyberwave.data.zenoh_backend import ZenohBackend
from cyberwave.workers.context import HookContext
from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.runtime import WorkerRuntime

TWIN_UUID = "00000000-0000-0000-0000-000000000099"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _decode_json(payload: bytes) -> Any:
    _, raw = wire_decode(payload)
    return json.loads(raw)


class _FakeCw:
    """Minimal stand-in for the Cyberwave client to register hooks."""

    def __init__(self, data_bus: DataBus, twin_uuid: str) -> None:
        self._hook_registry = HookRegistry()
        self._data_bus = data_bus

        class _Cfg:
            def __init__(self, uuid: str) -> None:
                self.twin_uuid = uuid

        self.config = _Cfg(twin_uuid)

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


def _publish_sensor_data(data_bus: DataBus, stop: threading.Event) -> None:
    t0 = time.time()
    while not stop.is_set():
        t = time.time() - t0
        data_bus.publish(
            "joint_states",
            {
                "q1": round(math.sin(t * 0.5) * 1.57, 3),
                "q2": round(math.cos(t * 0.3) * 0.8, 3),
            },
        )
        data_bus.publish(
            "imu",
            {
                "ax": round(math.sin(t) * 0.01, 3),
                "ay": 0.0,
                "az": 9.81,
            },
        )
        time.sleep(0.1)


def main() -> None:
    stop = threading.Event()

    port = _find_free_port()
    cfg = zenoh.Config()
    cfg.insert_json5("listen/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
    cfg.insert_json5("transport/shared_memory/enabled", "false")
    broker = zenoh.open(cfg)

    backend = ZenohBackend(connect=[f"tcp/127.0.0.1:{port}"], shared_memory=False)
    data_bus = DataBus(backend, TWIN_UUID)
    cw = _FakeCw(data_bus, TWIN_UUID)

    @cw.on_joint_states(TWIN_UUID)
    def on_joints(payload: bytes, ctx: HookContext) -> None:
        j = _decode_json(payload)
        print(f"[joints]       q1={j['q1']:+.3f}  q2={j['q2']:+.3f}")

    @cw.on_imu(TWIN_UUID)
    def on_imu_data(payload: bytes, ctx: HookContext) -> None:
        imu = _decode_json(payload)
        print(f"[imu]          az={imu['az']:.3f} m/s²")

    @cw.on_synchronized(TWIN_UUID, ["joint_states", "imu"], tolerance_ms=150)
    def on_synced(samples: dict, ctx: HookContext) -> None:
        j = _decode_json(samples["joint_states"].payload)
        imu = _decode_json(samples["imu"].payload)
        print(f"[synchronized] q1={j['q1']:+.3f}  az={imu['az']:.3f}")

    runtime = WorkerRuntime(cw)
    runtime.start()
    time.sleep(0.15)

    threading.Thread(
        target=_publish_sensor_data, args=(data_bus, stop), daemon=True
    ).start()

    print("Publishing sensor data… Press Ctrl+C to stop.\n")
    try:
        while not stop.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop.set()

    runtime.stop()
    backend.close()
    broker.close()


if __name__ == "__main__":
    main()
