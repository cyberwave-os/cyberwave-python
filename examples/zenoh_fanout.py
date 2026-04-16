"""Fan-out demo: one publisher (A) → three independent subscribers (B, C, D).

This example demonstrates that a single ``DataBus.publish()`` call reaches all
active subscribers independently.  Closing one subscriber does not affect the
others.

Architecture::

    Publisher A
        │
        ▼  DataBus / ZenohBackend
    ┌───────────────────────┐
    │  Zenoh broker (TCP)   │
    └──┬──────┬──────┬──────┘
       │      │      │
       ▼      ▼      ▼
    Sub B   Sub C   Sub D

Key observations
----------------
* All three subscribers declare interest on the same channel before A starts
  publishing — guaranteed by the synchronisation barrier.
* Each subscriber counts how many messages it receives; the summary shows
  delivery counts for all three.
* Subscriber B is intentionally closed halfway through to demonstrate that
  C and D keep receiving messages unaffected.

In-process broker
-----------------
An in-process Zenoh session listens on a free TCP port so all sessions
discover each other reliably without multicast scouting (important for
WSL2 / container environments).  Pass ``--connect`` to use an external router.

On standard Linux (bare-metal or a VM with multicast enabled), Zenoh's
peer-to-peer scouting would discover sessions automatically — no broker
needed.  The in-process broker is used here for portability and test
determinism: each run picks a random free port so parallel runs never
interfere with each other or with a real Zenoh router on port 7447.
POSIX shared memory is also available on real Linux (zero-copy delivery),
but is disabled here via ``shared_memory=False`` so the example runs
unchanged in WSL2 and containers.

Prerequisites
-------------
::

    pip install 'cyberwave[zenoh]'

Usage
-----
::

    python examples/zenoh_fanout.py
    python examples/zenoh_fanout.py --count 10
    python examples/zenoh_fanout.py --connect tcp/localhost:7447
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


# ---------------------------------------------------------------------------
# Broker / backend helpers
# ---------------------------------------------------------------------------


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


def _make_bus(backend: ZenohBackend) -> DataBus:
    return DataBus(backend, UUID)


# ---------------------------------------------------------------------------
# Subscriber worker
# ---------------------------------------------------------------------------


def run_subscriber(
    name: str,
    connect: list[str],
    *,
    ready: threading.Event,
    stop: threading.Event,
    counts: dict[str, int],
) -> None:
    """Subscribe to CHANNEL; increment counts[name] for every message received."""
    backend = _make_backend(connect)
    bus = _make_bus(backend)

    def on_message(value: object) -> None:
        if isinstance(value, dict):
            seq = value.get("seq", "?")
            print(f"[{name}] received seq={seq}")
        counts[name] = counts.get(name, 0) + 1

    sub = bus.subscribe(CHANNEL, on_message, policy="fifo")
    ready.set()

    stop.wait()

    sub.close()
    bus.close()


# ---------------------------------------------------------------------------
# Publisher worker
# ---------------------------------------------------------------------------


def run_publisher(
    connect: list[str],
    n_messages: int,
    *,
    ready: threading.Event,
    close_b_at: int | None,
    close_b_event: threading.Event,
) -> None:
    """Publish *n_messages* on CHANNEL after *ready* is set."""
    backend = _make_backend(connect)
    bus = _make_bus(backend)

    ready.wait()

    print(f"[A] publishing {n_messages} messages")
    for seq in range(n_messages):
        bus.publish(CHANNEL, {"seq": seq, "origin": "A"})
        print(f"[A] sent seq={seq}")
        if close_b_at is not None and seq == close_b_at - 1:
            close_b_event.set()
            print(f"[A] signalled B to close after seq={seq}")
        time.sleep(0.1)

    bus.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zenoh fan-out: A publishes to B, C, D simultaneously"
    )
    parser.add_argument(
        "--connect",
        metavar="ENDPOINT",
        help="External Zenoh router endpoint.  When omitted an in-process broker is started.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        metavar="N",
        help="Number of messages A publishes (default: 6)",
    )
    args = parser.parse_args()
    n = args.count

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

    counts: dict[str, int] = {}
    ready_b = threading.Event()
    ready_c = threading.Event()
    ready_d = threading.Event()
    stop_b = threading.Event()
    stop_c = threading.Event()
    stop_d = threading.Event()

    # B will be closed halfway through to show the others are unaffected.
    close_b_at = n // 2
    close_b_event = threading.Event()

    threads = [
        threading.Thread(
            target=run_subscriber,
            kwargs=dict(name="B", connect=connect, ready=ready_b, stop=stop_b, counts=counts),
            daemon=True,
        ),
        threading.Thread(
            target=run_subscriber,
            kwargs=dict(name="C", connect=connect, ready=ready_c, stop=stop_c, counts=counts),
            daemon=True,
        ),
        threading.Thread(
            target=run_subscriber,
            kwargs=dict(name="D", connect=connect, ready=ready_d, stop=stop_d, counts=counts),
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()

    # Wait until all three have declared their subscriptions.
    for ev in (ready_b, ready_c, ready_d):
        ev.wait()

    # Extra settle time so the subscription advertisements propagate.
    time.sleep(0.3)

    all_ready = threading.Event()
    all_ready.set()

    pub_thread = threading.Thread(
        target=run_publisher,
        kwargs=dict(
            connect=connect,
            n_messages=n,
            ready=all_ready,
            close_b_at=close_b_at,
            close_b_event=close_b_event,
        ),
        daemon=True,
    )
    pub_thread.start()

    # Close B when the publisher signals (halfway through).
    def _close_b_when_signalled() -> None:
        close_b_event.wait()
        time.sleep(0.05)
        print("[B] closing mid-stream")
        stop_b.set()

    threading.Thread(target=_close_b_when_signalled, daemon=True).start()

    pub_thread.join(timeout=30.0)

    # Give subscribers time to drain the last messages.
    time.sleep(0.5)
    stop_c.set()
    stop_d.set()

    for t in threads:
        t.join(timeout=5.0)

    if broker is not None:
        try:
            broker.close()
        except Exception:
            pass

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"Messages published by A : {n}")
    print(f"B closed after          : ~{close_b_at} messages")
    for name in ("B", "C", "D"):
        received = counts.get(name, 0)
        print(f"Messages received by {name}  : {received}")

    c_ok = counts.get("C", 0) == n
    d_ok = counts.get("D", 0) == n
    if c_ok and d_ok:
        print("\nFan-out successful: C and D received all messages.")
    else:
        missing_c = n - counts.get("C", 0)
        missing_d = n - counts.get("D", 0)
        if missing_c:
            print(f"\nWarning: C missed {missing_c} message(s).")
        if missing_d:
            print(f"\nWarning: D missed {missing_d} message(s).")


if __name__ == "__main__":
    main()
