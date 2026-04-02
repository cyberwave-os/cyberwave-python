"""Three simulated processes (A → B → C) connected via the Cyberwave data layer.

This example shows how the two layers of the data stack fit together:

* **Transport layer** — :class:`~cyberwave.data.zenoh_backend.ZenohBackend`
  opens a Zenoh session and handles raw bytes pub/sub over the wire.
* **Data layer** — :class:`~cyberwave.data.api.DataBus` sits on top and
  provides typed Python values (dicts, numpy arrays, …) with automatic
  wire-format encoding/decoding.

Architecture::

    Process A             Process B                  Process C
    ─────────────         ──────────────────────      ─────────────
    DataBus(UUID_A)  ──►  DataBus(UUID_A)  [sub]      DataBus(UUID_B)
    publishes              DataBus(UUID_B)  [pub]  ──►  subscribes
    "telemetry"            transforms & forwards       prints chain

Each process owns an independent ZenohBackend (Zenoh session).  In a real
deployment each block would be a separate OS process or container.  Here they
run as threads so the whole example fits in one file.

Zenoh key space (canonical ``cw`` prefix, no double-prefix)::

    cw/<UUID_A>/data/telemetry   # A → B hop
    cw/<UUID_B>/data/telemetry   # B → C hop

Key prefix note
---------------
``ZenohBackend`` is constructed with ``key_prefix=""`` so it forwards the key
string from ``DataBus`` unchanged.  ``DataBus`` already prepends ``"cw"`` and
builds the canonical ``cw/<uuid>/data/<channel>`` key.  Using the default
``ZenohBackend(key_prefix="cw")`` together with ``DataBus`` would produce a
double-prefix ``cw/cw/<uuid>/data/<channel>``.

In-process broker
-----------------
A lightweight Zenoh session is started at ``tcp/localhost:<port>`` with
``listen/endpoints`` so all three process sessions can connect to it
deterministically.  This avoids reliance on multicast scouting, which may
be disabled in some WSL2 / container environments.  Pass ``--connect`` to
point at an existing external router instead.

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

    # self-contained with in-process broker (default):
    python examples/zenoh_triad.py

    # external Zenoh router (skip the in-process broker):
    python examples/zenoh_triad.py --connect tcp/localhost:7447

    # send more messages:
    python examples/zenoh_triad.py --count 10

See also ``zenoh_bench.py`` for a latency/throughput benchmark over the same
A → B → C pipeline.
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

# ── Stable twin UUIDs for the three simulated processes ─────────────────────

UUID_A = "aaaaaaaa-0000-4000-a000-000000000001"
UUID_B = "bbbbbbbb-0000-4000-b000-000000000002"

CHANNEL = "telemetry"


# ── Broker helpers ───────────────────────────────────────────────────────────


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_broker(port: int) -> Any:
    """Open a Zenoh session that listens on *port*.

    Other sessions connect to this as their router, giving deterministic
    discovery without relying on multicast scouting.
    """
    import zenoh

    cfg = zenoh.Config()
    cfg.insert_json5("listen/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
    cfg.insert_json5("transport/shared_memory/enabled", "false")
    return zenoh.open(cfg)


# ── Backend / bus factories ──────────────────────────────────────────────────


def _make_backend(connect: list[str]) -> ZenohBackend:
    """Open a new Zenoh session connected to *connect* endpoints.

    ``key_prefix=""`` lets DataBus supply the full canonical key
    ``cw/<uuid>/data/<channel>`` without an extra layer of prefixing.
    """
    return ZenohBackend(key_prefix="", connect=connect, shared_memory=False)


def _make_bus(backend: ZenohBackend, twin_uuid: str) -> DataBus:
    """Wrap *backend* in a DataBus scoped to *twin_uuid*."""
    return DataBus(backend, twin_uuid)


# ── Process implementations ──────────────────────────────────────────────────


def run_process_a(
    connect: list[str],
    n_messages: int,
    *,
    ready: threading.Event,
    done: threading.Event,
) -> None:
    """Process A: publish *n_messages* telemetry samples, then signal done."""
    backend = _make_backend(connect)
    bus = _make_bus(backend, UUID_A)

    # Wait until B and C have declared their subscriptions.
    ready.wait()

    print(f"[A] publishing {n_messages} messages on '{CHANNEL}'")
    for seq in range(n_messages):
        payload = {"seq": seq, "origin": "A", "chain": ["A"]}
        bus.publish(CHANNEL, payload)
        print(f"[A] sent     seq={seq}  chain={payload['chain']}")
        time.sleep(0.15)

    done.set()
    bus.close()


def run_process_b(
    connect: list[str],
    n_messages: int,
    *,
    ready: threading.Event,
    done: threading.Event,
) -> None:
    """Process B: relay A's telemetry after appending 'B' to the chain.

    Uses **two DataBus instances on a single backend** — one scoped to UUID_A
    for subscribing to A's channel, one scoped to UUID_B for publishing its
    own transformed output.
    """
    backend = _make_backend(connect)
    bus_in = _make_bus(backend, UUID_A)   # subscribe to A
    bus_out = _make_bus(backend, UUID_B)  # publish as B

    received_count = [0]

    def on_message(value: object) -> None:
        if not isinstance(value, dict):
            return
        value["chain"] = value.get("chain", []) + ["B"]
        seq = value.get("seq", "?")
        print(f"[B] relayed  seq={seq}  chain={value['chain']}")
        bus_out.publish(CHANNEL, value)
        received_count[0] += 1

    sub = bus_in.subscribe(CHANNEL, on_message, policy="fifo")
    ready.set()

    # Keep running until A is done and all messages have been forwarded.
    done.wait()
    deadline = time.time() + 5.0
    while received_count[0] < n_messages and time.time() < deadline:
        time.sleep(0.05)

    sub.close()
    backend.close()


def run_process_c(
    connect: list[str],
    n_messages: int,
    *,
    ready: threading.Event,
    results: list[dict],
) -> None:
    """Process C: collect all messages from B and print the completed chain."""
    backend = _make_backend(connect)
    bus = _make_bus(backend, UUID_B)  # subscribe to B's channel

    collected = threading.Event()
    buffer: list[dict] = []

    def on_message(value: object) -> None:
        if not isinstance(value, dict):
            return
        value["chain"] = value.get("chain", []) + ["C"]
        seq = value.get("seq", "?")
        print(f"[C] received seq={seq}  chain={value['chain']}")
        buffer.append(value)
        if len(buffer) >= n_messages:
            collected.set()

    sub = bus.subscribe(CHANNEL, on_message, policy="fifo")
    ready.set()

    collected.wait(timeout=30.0)
    results.extend(buffer)

    sub.close()
    bus.close()


# ── Orchestrator ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zenoh triad: A → B → C data flow example"
    )
    parser.add_argument(
        "--connect",
        metavar="ENDPOINT",
        help="External Zenoh router endpoint, e.g. tcp/localhost:7447. "
        "When omitted an in-process broker is started automatically.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        metavar="N",
        help="Number of messages A publishes (default: 5)",
    )
    args = parser.parse_args()
    n = args.count

    # Start an in-process broker when no external router is given so all
    # sessions can discover each other reliably without multicast scouting.
    broker: Any = None
    if args.connect:
        connect = [args.connect]
        print(f"Using external router: {args.connect}")
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
        print(f"Started in-process broker on port {port}")

    # Small pause to let the broker begin accepting connections.
    time.sleep(0.2)

    # Synchronisation primitives.
    b_ready = threading.Event()
    c_ready = threading.Event()
    a_done = threading.Event()
    results: list[dict] = []

    # Start C then B so their subscriptions are active before A publishes.
    thread_c = threading.Thread(
        target=run_process_c,
        kwargs=dict(connect=connect, n_messages=n, ready=c_ready, results=results),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=run_process_b,
        kwargs=dict(connect=connect, n_messages=n, ready=b_ready, done=a_done),
        daemon=True,
    )

    thread_c.start()
    thread_b.start()

    # A waits until both B and C have declared their subscriptions.
    both_ready = threading.Event()

    def _wait_for_both() -> None:
        b_ready.wait()
        c_ready.wait()
        both_ready.set()

    threading.Thread(target=_wait_for_both, daemon=True).start()

    thread_a = threading.Thread(
        target=run_process_a,
        kwargs=dict(connect=connect, n_messages=n, ready=both_ready, done=a_done),
        daemon=True,
    )
    thread_a.start()

    thread_a.join(timeout=30.0)
    thread_b.join(timeout=10.0)
    thread_c.join(timeout=10.0)

    if broker is not None:
        try:
            broker.close()
        except Exception:
            pass

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"Messages published by A : {n}")
    print(f"Messages received by C  : {len(results)}")
    for msg in sorted(results, key=lambda m: m.get("seq", 0)):
        print(f"  seq={msg['seq']}  chain={msg['chain']}")

    if len(results) == n and all(msg["chain"] == ["A", "B", "C"] for msg in results):
        print("\nAll messages traversed A → B → C successfully.")
    else:
        missing = n - len(results)
        print(f"\nWarning: {missing} message(s) did not reach C.")


if __name__ == "__main__":
    main()
