"""Latency and throughput benchmark for the Cyberwave Zenoh data layer.

Measures the end-to-end cost of the full A → B → C pipeline for two payload types:

  json  (default): small dict  → JSON encode → … → JSON decode
  frame          : 480×640 RGB → numpy binary → … → numpy decode

Each message carries a ``sent_ns`` timestamp embedded by A
(``time.perf_counter_ns()``).  C records ``recv_ns`` at the top of its
callback and computes ``recv_ns - sent_ns`` per message.  Because all three
sessions run in the same OS process the monotonic clock is valid across
threads without any clock-sync overhead.

Two operating modes
-------------------
**Latency mode** (``--send-rate N``, default 200 msg/s):
  A sends at a controlled rate, keeping the pipeline well below saturation.
  The queue never builds so latency ≈ true A→C transit time.  Use this to
  answer "how fast is a single message?"

**Throughput mode** (``--send-rate 0``):
  A floods as fast as possible.  Messages queue up; latency numbers reflect
  queue wait + transit and are not meaningful as one-way latency estimates.
  Use this to answer "what is the maximum delivery rate?"

The first ``--warmup`` messages are always discarded so the pipeline reaches
steady state before timing begins.

Prerequisites
-------------
::

    pip install 'cyberwave[zenoh]'

Usage
-----
::

    # latency mode — 500 JSON messages at 200 msg/s (default)
    python examples/zenoh_bench.py

    # frame payload — 480×640 RGB numpy arrays at 30 fps (default for frame mode)
    python examples/zenoh_bench.py --payload frame

    # frame payload — flood mode to find max frame throughput
    python examples/zenoh_bench.py --payload frame --send-rate 0 --count 200

    # latency mode — explicit send rate
    python examples/zenoh_bench.py --send-rate 500 --count 1000

    # throughput mode — flood as fast as possible
    python examples/zenoh_bench.py --send-rate 0 --count 2000

    # external Zenoh router (skip the in-process broker)
    python examples/zenoh_bench.py --connect tcp/localhost:7447

Note: all sessions share one OS process over TCP loopback.  Latency
reflects encode + IPC + decode overhead, not a real network path.

In-process broker
-----------------
An in-process Zenoh session listens on a free TCP port so all benchmark
sessions connect deterministically without relying on multicast scouting
(important for WSL2 / container environments).

On standard Linux (bare-metal or a VM with multicast enabled), Zenoh's
peer-to-peer scouting would discover sessions automatically — no broker
needed.  The in-process broker is used here for portability and test
determinism: each run picks a random free port so parallel runs never
interfere with each other or with a real Zenoh router on port 7447.
POSIX shared memory is also available on real Linux (zero-copy delivery),
but is disabled here via ``shared_memory=False`` so the benchmark runs
unchanged in WSL2 and containers.
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

# ── Frame payload helpers ─────────────────────────────────────────────────────

# 480×640 RGB — matches a typical SD camera frame (~900 KB uncompressed).
FRAME_SHAPE = (480, 640, 3)
FRAME_CHANNEL = "frames"  # well-known stream channel for numpy frames

_TS_DTYPE = np.dtype("<u8")  # little-endian uint64


def _make_frame(sent_ns: int) -> np.ndarray:
    """Return a zeroed 480×640 RGB frame with *sent_ns* packed into the first 8 bytes."""
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    frame.ravel()[:8].view(_TS_DTYPE)[0] = sent_ns
    return frame


def _frame_ts(frame: np.ndarray) -> int:
    """Extract the timestamp packed into the first 8 bytes of *frame*."""
    return int(frame.ravel()[:8].view(_TS_DTYPE)[0])

# ── Twin UUIDs (must match the roles used in zenoh_triad.py) ────────────────

UUID_A = "aaaaaaaa-0000-4000-a000-000000000001"
UUID_B = "bbbbbbbb-0000-4000-b000-000000000002"

CHANNEL = "telemetry"


# ── Infrastructure helpers ───────────────────────────────────────────────────


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


def _make_backend(connect: list[str]) -> ZenohBackend:
    return ZenohBackend(connect=connect, shared_memory=False)


def _make_bus(backend: ZenohBackend, twin_uuid: str) -> DataBus:
    return DataBus(backend, twin_uuid)


# ── Benchmark process functions ──────────────────────────────────────────────


def _sender(
    connect: list[str],
    n_messages: int,
    n_warmup: int,
    rate_hz: float,
    payload_type: str,
    *,
    ready: threading.Event,
    timing: dict[str, int],
    done: threading.Event,
) -> None:
    """A: publish with embedded send timestamps.

    When *rate_hz* > 0 messages are paced so the pipeline is never saturated.
    When *rate_hz* == 0 messages are sent as fast as possible (throughput mode).

    For ``payload_type="frame"`` the timestamp is packed into the first 8 bytes
    of a 480×640 RGB numpy array.  Warmup frames carry ts=0 so the sink
    discards them.  For ``payload_type="json"`` the timestamp is a dict field
    and warmup messages carry ``seq < 0``.
    """
    backend = _make_backend(connect)
    channel = FRAME_CHANNEL if payload_type == "frame" else CHANNEL
    bus = _make_bus(backend, UUID_A)
    ready.wait()

    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0

    for i in range(n_warmup):
        if payload_type == "frame":
            bus.publish(channel, _make_frame(0))  # ts=0 marks warmup
        else:
            bus.publish(channel, {"seq": -(i + 1), "sent_ns": time.perf_counter_ns()})
        if interval_s:
            time.sleep(interval_s)
    time.sleep(0.1)  # let warmup drain through B

    timing["start_ns"] = time.perf_counter_ns()
    for seq in range(n_messages):
        sent_ns = time.perf_counter_ns()
        if payload_type == "frame":
            bus.publish(channel, _make_frame(sent_ns))
        else:
            bus.publish(channel, {"seq": seq, "sent_ns": sent_ns})
        if interval_s:
            time.sleep(interval_s)

    done.set()
    bus.close()


def _relay(
    connect: list[str],
    payload_type: str,
    *,
    ready: threading.Event,
    done: threading.Event,
) -> None:
    """B: forward every message unchanged."""
    backend = _make_backend(connect)
    channel = FRAME_CHANNEL if payload_type == "frame" else CHANNEL
    bus_in = _make_bus(backend, UUID_A)
    bus_out = _make_bus(backend, UUID_B)

    def on_msg(value: object) -> None:
        if value is not None:
            bus_out.publish(channel, value)

    sub = bus_in.subscribe(channel, on_msg, policy="fifo")
    ready.set()

    done.wait()
    time.sleep(1.0)  # drain in-flight messages
    sub.close()
    backend.close()


def _sink(
    connect: list[str],
    n_messages: int,
    payload_type: str,
    *,
    ready: threading.Event,
    timing: dict[str, int],
    latencies_ns: list[int],
) -> None:
    """C: record per-message A→C latency."""
    backend = _make_backend(connect)
    channel = FRAME_CHANNEL if payload_type == "frame" else CHANNEL
    bus = _make_bus(backend, UUID_B)

    collected = threading.Event()

    def on_msg(value: object) -> None:
        recv_ns = time.perf_counter_ns()
        if isinstance(value, np.ndarray):
            sent_ns = _frame_ts(value)
            if sent_ns == 0:
                return  # warmup frame — discard
        elif isinstance(value, dict):
            if value.get("seq", -1) < 0:
                return  # warmup — discard
            sent_ns = value.get("sent_ns")
            if sent_ns is None:
                return
        else:
            return
        latencies_ns.append(recv_ns - sent_ns)
        if len(latencies_ns) >= n_messages:
            timing["end_ns"] = recv_ns
            collected.set()

    sub = bus.subscribe(channel, on_msg, policy="fifo")
    ready.set()

    collected.wait(timeout=120.0)
    sub.close()
    bus.close()


# ── Results ──────────────────────────────────────────────────────────────────


def _print_results(
    latencies_ns: list[int],
    n_sent: int,
    n_warmup: int,
    rate_hz: float,
    payload_type: str,
    timing: dict[str, int],
) -> None:
    n = len(latencies_ns)
    if n == 0:
        print("No messages received.")
        return

    sorted_lat = sorted(latencies_ns)

    def µs(ns: float) -> str:
        return f"{ns / 1_000:.1f} µs"

    def pct(p: float) -> int:
        return sorted_lat[min(int(p / 100 * n), n - 1)]

    mean_ns = statistics.mean(latencies_ns)
    stdev_ns = statistics.stdev(latencies_ns) if n > 1 else 0.0

    if payload_type == "frame":
        frame_bytes = int(np.prod(FRAME_SHAPE))
        payload_label = f"numpy frame {FRAME_SHAPE[1]}×{FRAME_SHAPE[0]} RGB  ({frame_bytes / 1024:.0f} KB)"
    else:
        payload_label = "JSON dict"

    elapsed_s = (timing.get("end_ns", 0) - timing.get("start_ns", 0)) / 1e9
    throughput = n / elapsed_s if elapsed_s > 0 else 0.0

    if payload_type == "frame" and elapsed_s > 0:
        mb_s = (n * int(np.prod(FRAME_SHAPE))) / elapsed_s / 1024 / 1024
        thru_label = f"{throughput:.0f} fps  ({mb_s:.0f} MB/s)"
    else:
        thru_label = f"{throughput:.0f} msg/s"

    mode = "throughput (flooded)" if rate_hz == 0 else f"latency (send-rate={rate_hz:.0f} msg/s)"
    print(f"\n── A → B → C  [{mode}]  ({n}/{n_sent} msgs) ────")
    print(f"  payload  {payload_label}")
    print(f"  min    {µs(sorted_lat[0])}")
    print(f"  p50    {µs(pct(50))}")
    print(f"  p95    {µs(pct(95))}")
    print(f"  p99    {µs(pct(99))}")
    print(f"  max    {µs(sorted_lat[-1])}")
    print(f"  mean   {µs(mean_ns)}  (σ = {µs(stdev_ns)})")
    if rate_hz == 0:
        print(
            f"\n  Note: latency is inflated by queue buildup in throughput mode.\n"
            f"  Use --rate to pace sends and get true transit latency.\n"
            f"  min ({µs(sorted_lat[0])}) is the best estimate of actual pipeline latency."
        )
    print(f"  ──")
    print(f"  msgs   {n_sent}  (warmup {n_warmup} discarded)")
    print(f"  time   {elapsed_s * 1_000:.1f} ms")
    print(f"  thru   {thru_label}")
    print(
        "\nNote: in-process TCP loopback — measures encode/IPC/decode, not real network."
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zenoh data-layer latency/throughput benchmark (A → B → C)"
    )
    parser.add_argument(
        "--connect",
        metavar="ENDPOINT",
        help="External Zenoh router, e.g. tcp/localhost:7447. "
        "Omit to start an in-process broker automatically.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        metavar="N",
        help="Number of messages to time (default: 500)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        metavar="W",
        help="Warmup messages discarded before timing starts (default: 20)",
    )
    parser.add_argument(
        "--send-rate",
        type=float,
        default=None,
        metavar="R",
        dest="send_rate",
        help="A's publish rate in msg/s (default: 200 for json, 30 for frame). "
        "Set to 0 for throughput mode (flood as fast as possible).",
    )
    parser.add_argument(
        "--payload",
        choices=["json", "frame"],
        default="json",
        help="Payload type: 'json' (small dict, default) or "
        "'frame' (480×640 RGB numpy array, ~900 KB)",
    )
    args = parser.parse_args()
    payload_type = args.payload

    broker: Any = None
    if args.connect:
        connect = [args.connect]
        print(f"Using external router: {args.connect}")
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
        print(f"Started in-process broker on port {port}")

    time.sleep(0.2)

    default_rate = 30.0 if payload_type == "frame" else 200.0
    rate_hz = args.send_rate if args.send_rate is not None else default_rate
    mode_label = "flooded (throughput mode)" if rate_hz == 0 else f"{rate_hz:.0f} msg/s"
    frame_info = f"  payload={payload_type}" + (
        f" ({FRAME_SHAPE[1]}×{FRAME_SHAPE[0]} RGB)" if payload_type == "frame" else ""
    )
    print(
        f"Sending {args.count} messages (+ {args.warmup} warmup) "
        f"through A → B → C at {mode_label}{frame_info} …"
    )

    b_ready = threading.Event()
    c_ready = threading.Event()
    a_done = threading.Event()
    timing: dict[str, int] = {}
    latencies_ns: list[int] = []

    both_ready = threading.Event()

    def _wait_for_both() -> None:
        b_ready.wait()
        c_ready.wait()
        both_ready.set()

    timeout_s = max(120.0, args.count / rate_hz * 2) if rate_hz > 0 else 120.0

    thread_c = threading.Thread(
        target=_sink,
        kwargs=dict(
            connect=connect,
            n_messages=args.count,
            payload_type=payload_type,
            ready=c_ready,
            timing=timing,
            latencies_ns=latencies_ns,
        ),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=_relay,
        kwargs=dict(connect=connect, payload_type=payload_type, ready=b_ready, done=a_done),
        daemon=True,
    )
    thread_a = threading.Thread(
        target=_sender,
        kwargs=dict(
            connect=connect,
            n_messages=args.count,
            n_warmup=args.warmup,
            rate_hz=rate_hz,
            payload_type=payload_type,
            ready=both_ready,
            timing=timing,
            done=a_done,
        ),
        daemon=True,
    )

    thread_c.start()
    thread_b.start()
    threading.Thread(target=_wait_for_both, daemon=True).start()
    thread_a.start()

    thread_a.join(timeout=timeout_s)
    thread_b.join(timeout=15.0)
    thread_c.join(timeout=15.0)

    if broker is not None:
        try:
            broker.close()
        except Exception:
            pass

    _print_results(latencies_ns, args.count, args.warmup, rate_hz, payload_type, timing)


if __name__ == "__main__":
    main()
