"""
Zenoh Benchmark — latency and throughput for the A → B → C data pipeline.

Supports JSON and numpy frame payloads in latency or throughput mode.

Requirements:
    pip install cyberwave[zenoh]

Usage:
    python examples/zenoh_bench.py
    python examples/zenoh_bench.py --payload frame --send-rate 30
    python examples/zenoh_bench.py --send-rate 0 --count 2000
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import threading
import time
from typing import Any

import numpy as np

from cyberwave.data.api import DataBus
from cyberwave.data.zenoh_backend import ZenohBackend

FRAME_SHAPE = (480, 640, 3)
FRAME_CHANNEL = "frames"
_TS_DTYPE = np.dtype("<u8")
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


def _make_frame(sent_ns: int) -> np.ndarray:
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    frame.ravel()[:8].view(_TS_DTYPE)[0] = sent_ns
    return frame


def _frame_ts(frame: np.ndarray) -> int:
    return int(frame.ravel()[:8].view(_TS_DTYPE)[0])


def _sender(
    connect: list[str],
    n: int,
    warmup: int,
    rate: float,
    payload: str,
    *,
    ready: threading.Event,
    timing: dict[str, int],
    done: threading.Event,
) -> None:
    bus = DataBus(_backend(connect), UUID_A)
    ch = FRAME_CHANNEL if payload == "frame" else CHANNEL
    interval = 1 / rate if rate > 0 else 0
    ready.wait()

    for i in range(warmup):
        if payload == "frame":
            bus.publish(ch, _make_frame(0))
        else:
            bus.publish(ch, {"seq": -(i + 1), "sent_ns": time.perf_counter_ns()})
        if interval:
            time.sleep(interval)
    time.sleep(0.1)

    timing["start_ns"] = time.perf_counter_ns()
    for seq in range(n):
        ns = time.perf_counter_ns()
        if payload == "frame":
            bus.publish(ch, _make_frame(ns))
        else:
            bus.publish(ch, {"seq": seq, "sent_ns": ns})
        if interval:
            time.sleep(interval)
    done.set()
    bus.close()


def _relay(
    connect: list[str], payload: str, *, ready: threading.Event, done: threading.Event
) -> None:
    be = _backend(connect)
    ch = FRAME_CHANNEL if payload == "frame" else CHANNEL
    bus_in, bus_out = DataBus(be, UUID_A), DataBus(be, UUID_B)
    sub = bus_in.subscribe(
        ch, lambda v: bus_out.publish(ch, v) if v else None, policy="fifo"
    )
    ready.set()
    done.wait()
    time.sleep(1)
    sub.close()
    be.close()


def _sink(
    connect: list[str],
    n: int,
    payload: str,
    *,
    ready: threading.Event,
    timing: dict[str, int],
    latencies: list[int],
) -> None:
    bus = DataBus(_backend(connect), UUID_B)
    ch = FRAME_CHANNEL if payload == "frame" else CHANNEL
    collected = threading.Event()

    def on_msg(value: object) -> None:
        recv = time.perf_counter_ns()
        if isinstance(value, np.ndarray):
            sent = _frame_ts(value)
            if sent == 0:
                return
        elif isinstance(value, dict):
            if value.get("seq", -1) < 0:
                return
            sent = value.get("sent_ns")
            if sent is None:
                return
        else:
            return
        latencies.append(recv - sent)
        if len(latencies) >= n:
            timing["end_ns"] = recv
            collected.set()

    sub = bus.subscribe(ch, on_msg, policy="fifo")
    ready.set()
    collected.wait(timeout=120)
    sub.close()
    bus.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zenoh data-layer benchmark (A → B → C)"
    )
    parser.add_argument("--connect", metavar="ENDPOINT")
    parser.add_argument("--count", type=int, default=500, metavar="N")
    parser.add_argument("--warmup", type=int, default=20, metavar="W")
    parser.add_argument(
        "--send-rate", type=float, default=None, dest="send_rate", metavar="R"
    )
    parser.add_argument("--payload", choices=["json", "frame"], default="json")
    args = parser.parse_args()

    broker = None
    if args.connect:
        connect = [args.connect]
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
    time.sleep(0.2)

    rate = (
        args.send_rate
        if args.send_rate is not None
        else (30 if args.payload == "frame" else 200)
    )
    print(
        f"Benchmark: {args.count} msgs, payload={args.payload}, "
        f"rate={'flooded' if rate == 0 else f'{rate:.0f}/s'}"
    )

    b_ready, c_ready, a_done = threading.Event(), threading.Event(), threading.Event()
    both_ready = threading.Event()
    timing: dict[str, int] = {}
    latencies: list[int] = []

    def _wait_both() -> None:
        b_ready.wait()
        c_ready.wait()
        both_ready.set()

    threads = [
        threading.Thread(
            target=_sink,
            kwargs=dict(
                connect=connect,
                n=args.count,
                payload=args.payload,
                ready=c_ready,
                timing=timing,
                latencies=latencies,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=_relay,
            kwargs=dict(
                connect=connect, payload=args.payload, ready=b_ready, done=a_done
            ),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    threading.Thread(target=_wait_both, daemon=True).start()

    timeout = max(120, args.count / rate * 2) if rate > 0 else 120
    t_a = threading.Thread(
        target=_sender,
        kwargs=dict(
            connect=connect,
            n=args.count,
            warmup=args.warmup,
            rate=rate,
            payload=args.payload,
            ready=both_ready,
            timing=timing,
            done=a_done,
        ),
        daemon=True,
    )
    t_a.start()
    t_a.join(timeout=timeout)
    for t in threads:
        t.join(timeout=15)
    if broker:
        broker.close()

    n = len(latencies)
    if n == 0:
        print("No messages received.")
        return

    sorted_lat = sorted(latencies)
    elapsed_s = (timing.get("end_ns", 0) - timing.get("start_ns", 0)) / 1e9
    fps = n / elapsed_s if elapsed_s > 0 else 0

    def us(ns: float) -> str:
        return f"{ns / 1000:.1f} µs"

    def pct(p: float) -> int:
        return sorted_lat[min(int(p / 100 * n), n - 1)]

    mode = "throughput" if rate == 0 else f"latency ({rate:.0f}/s)"
    print(f"\n── Results [{mode}] ({n}/{args.count} msgs) ──")
    print(f"  p50={us(pct(50))}  p95={us(pct(95))}  p99={us(pct(99))}")
    print(
        f"  mean={us(statistics.mean(latencies))}  "
        f"min={us(sorted_lat[0])}  max={us(sorted_lat[-1])}"
    )
    print(f"  throughput={fps:.0f} msg/s  elapsed={elapsed_s * 1000:.1f} ms")


if __name__ == "__main__":
    main()
