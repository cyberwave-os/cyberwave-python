"""
Zenoh Fan-out — one publisher (A) delivers to three subscribers (B, C, D).

Subscriber B is closed halfway through to show C and D are unaffected.

Requirements:
    pip install cyberwave[zenoh]
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from typing import Any

from cyberwave.data.api import DataBus
from cyberwave.data.zenoh_backend import ZenohBackend

UUID = "dddddddd-0000-4000-d000-000000000004"
CHANNEL = "telemetry"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_broker(port: int) -> Any:
    import zenoh

    cfg = zenoh.Config()
    cfg.insert_json5("listen/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
    cfg.insert_json5("transport/shared_memory/enabled", "false")
    return zenoh.open(cfg)


def _backend(connect: list[str]) -> ZenohBackend:
    return ZenohBackend(connect=connect, shared_memory=False)


def subscriber(
    name: str,
    connect: list[str],
    *,
    ready: threading.Event,
    stop: threading.Event,
    counts: dict[str, int],
) -> None:
    bus = DataBus(_backend(connect), UUID)

    def on_msg(value: object) -> None:
        counts[name] = counts.get(name, 0) + 1

    sub = bus.subscribe(CHANNEL, on_msg, policy="fifo")
    ready.set()
    stop.wait()
    sub.close()
    bus.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenoh fan-out: A → B, C, D")
    parser.add_argument("--connect", metavar="ENDPOINT")
    parser.add_argument("--count", type=int, default=6, metavar="N")
    args = parser.parse_args()
    n = args.count

    broker = None
    if args.connect:
        connect = [args.connect]
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
    time.sleep(0.2)

    counts: dict[str, int] = {}
    stops = {name: threading.Event() for name in ("B", "C", "D")}
    readies = {name: threading.Event() for name in ("B", "C", "D")}

    threads = [
        threading.Thread(
            target=subscriber,
            kwargs=dict(
                name=name,
                connect=connect,
                ready=readies[name],
                stop=stops[name],
                counts=counts,
            ),
            daemon=True,
        )
        for name in ("B", "C", "D")
    ]
    for t in threads:
        t.start()
    for ev in readies.values():
        ev.wait()
    time.sleep(0.3)

    close_b_at = n // 2
    bus = DataBus(_backend(connect), UUID)
    for seq in range(n):
        bus.publish(CHANNEL, {"seq": seq})
        if seq == close_b_at - 1:
            stops["B"].set()
        time.sleep(0.1)
    bus.close()

    time.sleep(0.5)
    stops["C"].set()
    stops["D"].set()
    for t in threads:
        t.join(timeout=5)
    if broker:
        broker.close()

    print(f"\nPublished: {n}")
    print(f"B closed after ~{close_b_at} messages")
    for name in ("B", "C", "D"):
        print(f"  {name} received: {counts.get(name, 0)}")


if __name__ == "__main__":
    main()
