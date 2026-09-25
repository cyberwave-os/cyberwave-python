"""
Data Recording — record sensor data to disk and replay it.

Uses an in-process Zenoh broker. Records joint_states at 10 Hz, then replays.

Requirements:
    pip install cyberwave[zenoh]
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

from cyberwave.data.api import DataBus
from cyberwave.data.recording import record, replay
from cyberwave.data.zenoh_backend import ZenohBackend

UUID = "eeeeeeee-0000-4000-e000-000000000005"
CHANNEL = "joint_states"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and replay sensor data")
    parser.add_argument("--connect", metavar="ENDPOINT")
    parser.add_argument("--count", type=int, default=10, metavar="N")
    parser.add_argument("--speed", type=float, default=2.0, metavar="S")
    args = parser.parse_args()

    broker = None
    if args.connect:
        connect = [args.connect]
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
    time.sleep(0.2)

    with tempfile.TemporaryDirectory(prefix="cw_recording_") as tmp:
        rec_dir = Path(tmp)

        # Record phase
        backend = _backend(connect)
        bus = DataBus(backend, UUID)
        from cyberwave.data.keys import build_key

        key = build_key(UUID, CHANNEL)

        print(f"Recording {args.count} samples…")
        with record(backend, [key], rec_dir) as session:
            time.sleep(0.2)
            for seq in range(args.count):
                t = seq * 0.1
                bus.publish(CHANNEL, {"seq": seq, "j1": math.sin(t), "j2": math.cos(t)})
                print(f"  seq={seq}  j1={math.sin(t):+.3f}")
                time.sleep(0.1)
            time.sleep(0.3)
        print(f"Recorded {session.sample_count} samples to {rec_dir}")
        bus.close()

        # Replay phase
        backend2 = _backend(connect)
        replayed: list[dict] = []

        def _decode(sample: Any) -> Any:
            from cyberwave.data.header import decode

            _, payload = decode(sample.payload)
            return json.loads(payload)

        sub = backend2.subscribe(
            key, lambda s: replayed.append(_decode(s)), policy="fifo"
        )
        time.sleep(0.2)

        print(f"\nReplaying at {args.speed}x…")
        result = replay(backend2, rec_dir, speed=args.speed)
        time.sleep(max(2, args.count / 10 / args.speed * 2))
        sub.close()
        backend2.close()

        print(f"Replayed {result.samples_published} → received {len(replayed)}")

    if broker:
        broker.close()


if __name__ == "__main__":
    main()
