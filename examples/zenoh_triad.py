"""
Zenoh Triad — three simulated processes (A → B → C) connected via the data layer.

A publishes, B relays with transformation, C receives and prints the chain.

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

UUID_A = "aaaaaaaa-0000-4000-a000-000000000001"
UUID_B = "bbbbbbbb-0000-4000-b000-000000000002"
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


def process_a(
    connect: list[str], n: int, *, ready: threading.Event, done: threading.Event
) -> None:
    bus = DataBus(_backend(connect), UUID_A)
    ready.wait()
    for seq in range(n):
        bus.publish(CHANNEL, {"seq": seq, "chain": ["A"]})
        print(f"[A] sent seq={seq}")
        time.sleep(0.15)
    done.set()
    bus.close()


def process_b(
    connect: list[str], n: int, *, ready: threading.Event, done: threading.Event
) -> None:
    be = _backend(connect)
    bus_in, bus_out = DataBus(be, UUID_A), DataBus(be, UUID_B)
    count = [0]

    def on_msg(value: object) -> None:
        if isinstance(value, dict):
            value["chain"] = value.get("chain", []) + ["B"]
            bus_out.publish(CHANNEL, value)
            print(f"[B] relayed seq={value.get('seq')}")
            count[0] += 1

    sub = bus_in.subscribe(CHANNEL, on_msg, policy="fifo")
    ready.set()
    done.wait()
    time.sleep(1)
    sub.close()
    be.close()


def process_c(
    connect: list[str], n: int, *, ready: threading.Event, results: list[dict]
) -> None:
    bus = DataBus(_backend(connect), UUID_B)
    collected = threading.Event()

    def on_msg(value: object) -> None:
        if isinstance(value, dict):
            value["chain"] = value.get("chain", []) + ["C"]
            print(f"[C] received seq={value.get('seq')}  chain={value['chain']}")
            results.append(value)
            if len(results) >= n:
                collected.set()

    sub = bus.subscribe(CHANNEL, on_msg, policy="fifo")
    ready.set()
    collected.wait(timeout=30)
    sub.close()
    bus.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenoh triad: A → B → C")
    parser.add_argument("--connect", metavar="ENDPOINT")
    parser.add_argument("--count", type=int, default=5, metavar="N")
    args = parser.parse_args()

    broker = None
    if args.connect:
        connect = [args.connect]
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
    time.sleep(0.2)

    b_ready, c_ready, a_done = threading.Event(), threading.Event(), threading.Event()
    results: list[dict] = []
    both_ready = threading.Event()

    def _wait_both() -> None:
        b_ready.wait()
        c_ready.wait()
        both_ready.set()

    threads = [
        threading.Thread(
            target=process_c,
            kwargs=dict(connect=connect, n=args.count, ready=c_ready, results=results),
            daemon=True,
        ),
        threading.Thread(
            target=process_b,
            kwargs=dict(connect=connect, n=args.count, ready=b_ready, done=a_done),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    threading.Thread(target=_wait_both, daemon=True).start()
    t_a = threading.Thread(
        target=process_a,
        kwargs=dict(connect=connect, n=args.count, ready=both_ready, done=a_done),
        daemon=True,
    )
    t_a.start()
    t_a.join(timeout=30)
    for t in threads:
        t.join(timeout=10)

    if broker:
        broker.close()

    print(f"\nSummary: {len(results)}/{args.count} messages reached C")
    if len(results) == args.count:
        print("All messages traversed A → B → C successfully.")


if __name__ == "__main__":
    main()
