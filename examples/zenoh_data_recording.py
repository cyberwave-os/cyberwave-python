"""Record and replay demo using the Cyberwave data layer.

This example walks through the full record → replay lifecycle:

1. **Record phase** — a ``ZenohBackend`` is started, joint-state dicts are
   published at ~10 Hz, and ``record()`` captures every raw sample to disk
   in a timestamped directory.
2. **Replay phase** — a fresh backend and subscriber are opened, then
   ``replay()`` re-publishes all recorded samples at ``speed`` × real-time.
   The subscriber prints each replayed value and verifies that all samples
   arrived in order.

Data schema (``joint_states`` channel)::

    {
        "seq": int,          # monotonically increasing index
        "j1": float,         # joint 1 angle in radians
        "j2": float,
        "j3": float,
        "timestamp_ms": float,
    }

In-process broker
-----------------
A lightweight Zenoh session is started on a free TCP port so recording and
replay sessions connect deterministically.  Pass ``--connect`` to use an
external Zenoh router instead.

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

    python examples/zenoh_data_recording.py
    python examples/zenoh_data_recording.py --count 20 --speed 4.0
    python examples/zenoh_data_recording.py --connect tcp/localhost:7447
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from cyberwave.data.api import DataBus
from cyberwave.data.recording import record, replay
from cyberwave.data.zenoh_backend import ZenohBackend

UUID = "eeeeeeee-0000-4000-e000-000000000005"
CHANNEL = "joint_states"
PUBLISH_HZ = 10.0


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
# Phases
# ---------------------------------------------------------------------------


def record_phase(
    connect: list[str],
    n_messages: int,
    rec_dir: Path,
) -> list[dict]:
    """Publish *n_messages* joint-state samples while recording them to *rec_dir*.

    Returns the list of published payloads for later verification.
    """
    backend = _make_backend(connect)
    bus = _make_bus(backend)

    # Build the full Zenoh key that DataBus uses for this channel so we can
    # pass it to record().
    from cyberwave.data.keys import build_key

    key = build_key(UUID, CHANNEL)

    published: list[dict] = []

    with record(backend, [key], rec_dir) as session:
        # Give the recording subscription time to be declared.
        time.sleep(0.2)

        print(f"[record] publishing {n_messages} joint-state samples at {PUBLISH_HZ} Hz")
        interval = 1.0 / PUBLISH_HZ
        for seq in range(n_messages):
            t = seq * interval
            payload = {
                "seq": seq,
                "j1": math.sin(t),
                "j2": math.cos(t),
                "j3": math.sin(2 * t),
                "timestamp_ms": time.time() * 1000,
            }
            bus.publish(CHANNEL, payload)
            published.append(payload)
            print(
                f"[record] seq={seq:3d}  j1={payload['j1']:+.3f}"
                f"  j2={payload['j2']:+.3f}  j3={payload['j3']:+.3f}"
            )
            time.sleep(interval)

        # Flush: wait for the recording subscriber to capture all samples.
        time.sleep(0.3)

    print(
        f"[record] done — {session.sample_count} sample(s) written to '{rec_dir}'"
    )
    bus.close()
    return published


def replay_phase(
    connect: list[str],
    rec_dir: Path,
    speed: float,
    published: list[dict],
) -> None:
    """Replay the recording and verify all messages arrive back in order."""
    backend = _make_backend(connect)
    bus = _make_bus(backend)

    replayed: list[dict] = []
    all_received = threading.Event()
    n_expected = len(published)

    def on_message(value: object) -> None:
        if isinstance(value, dict):
            replayed.append(value)
            seq = value.get("seq", "?")
            print(
                f"[replay] seq={seq:3d}  j1={value.get('j1', 0):+.3f}"
                f"  j2={value.get('j2', 0):+.3f}  j3={value.get('j3', 0):+.3f}"
            )
        if len(replayed) >= n_expected:
            all_received.set()

    from cyberwave.data.keys import build_key

    key = build_key(UUID, CHANNEL)

    sub = backend.subscribe(key, lambda s: on_message(_decode_sample(s)), policy="fifo")
    time.sleep(0.2)

    speed_label = f"{speed:.1f}×" if speed > 0 else "instant (speed=0)"
    print(f"\n[replay] replaying {n_expected} sample(s) at {speed_label}")
    result = replay(backend, rec_dir, speed=speed)
    print(f"[replay] replay() returned: {result.samples_published} sample(s) published")

    all_received.wait(timeout=max(10.0, n_expected / PUBLISH_HZ * 2))

    sub.close()
    bus.close()

    print(f"\n[replay] received {len(replayed)}/{n_expected} sample(s)")

    # Verify order by sequence number.
    seqs = [m.get("seq") for m in replayed if isinstance(m.get("seq"), int)]
    if seqs == sorted(seqs):
        print("[replay] All samples arrived in order.")
    else:
        out_of_order = sum(
            1 for a, b in zip(seqs, sorted(seqs)) if a != b
        )
        print(f"[replay] Warning: {out_of_order} sample(s) arrived out of order.")


def _decode_sample(sample: Any) -> Any:
    """Decode a raw backend Sample back to a Python value."""
    from cyberwave.data.header import decode

    header, payload = decode(sample.payload)
    import json as _json

    if hasattr(header, "content_type") and header.content_type == "application/json":
        return _json.loads(payload)
    try:
        return _json.loads(payload)
    except Exception:
        return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zenoh data recording: record joint states then replay"
    )
    parser.add_argument(
        "--connect",
        metavar="ENDPOINT",
        help="External Zenoh router endpoint.  When omitted an in-process broker is started.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        metavar="N",
        help="Number of joint-state samples to record (default: 10)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        metavar="S",
        help="Replay speed multiplier (default: 2.0; use 0 for instant replay)",
    )
    args = parser.parse_args()

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

    with tempfile.TemporaryDirectory(prefix="cw_recording_") as tmp:
        rec_dir = Path(tmp)
        print(f"Recording directory: {rec_dir}\n")

        print("── Record phase ─────────────────────────────────────────────────────")
        published = record_phase(connect, args.count, rec_dir)

        print("\n── Replay phase ─────────────────────────────────────────────────────")
        replay_phase(connect, rec_dir, args.speed, published)

    if broker is not None:
        try:
            broker.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
