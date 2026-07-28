"""
Zenoh SHM zero-copy — measure the win over copy-over-loopback.

A publisher (SDK ``ZenohBackend``) streams 720p-RGB-sized frames to a
same-host consumer.  With ``--mode shm`` each frame is allocated from a
shared-memory pool and only a descriptor crosses the transport; with
``--mode copy`` the full frame is copied.  ``--mode compare`` runs both and
prints the delta.

The consumer uses a raw zenoh subscriber so it can call ``payload.as_shm()``
— the definitive per-sample zero-copy detector (the SDK ``Sample`` exposes
only decoded bytes).

Zero-copy requires, on Linux/containers:
    * ``ipc: host``            (shared /dev/shm + IPC namespace across containers)
    * ``ulimits: memlock: -1`` (lift the 8 MB RLIMIT_MEMLOCK lock budget)
This is the same on aarch64 (Jetson) and x86_64.

Requirements:
    pip install cyberwave[zenoh]

Usage:
    python examples/zenoh_shm_zerocopy.py                 # compare shm vs copy
    python examples/zenoh_shm_zerocopy.py --mode shm
    python examples/zenoh_shm_zerocopy.py --width 1920 --height 1080 --fps 30
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import struct
import threading
import time
from typing import Any

import zenoh

from cyberwave.data.zenoh_backend import ZenohBackend

KEY = "shmdemo/frame"
HDR = "<4sQd"  # magic, counter, send-monotonic-seconds
HDR_LEN = struct.calcsize(HDR)
MAGIC = b"\xca\xfe\xba\xbe"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _consumer(
    endpoint: str,
    n: int,
    warmup: int,
    *,
    ready: threading.Event,
    result: dict[str, Any],
) -> None:
    """Raw zenoh subscriber: records per-frame latency and SHM-backed count."""
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "false")
    cfg.insert_json5("connect/endpoints", json.dumps([endpoint]))
    cfg.insert_json5("transport/shared_memory/enabled", "true")
    session = zenoh.open(cfg)

    lat: list[tuple[int, float]] = []
    shm_hits = 0
    total = 0
    lock = threading.Lock()
    done = threading.Event()

    def on_sample(s: Any) -> None:
        nonlocal shm_hits, total
        recv = time.monotonic()
        is_shm = s.payload.as_shm() is not None
        raw = s.payload.to_bytes()
        if raw[0:4] != MAGIC:
            return
        _, i, sent = struct.unpack(HDR, raw[0:HDR_LEN])
        with lock:
            total += 1
            shm_hits += 1 if is_shm else 0
            lat.append((i, (recv - sent) * 1000.0))
            if len(lat) >= n:
                done.set()

    sub = session.declare_subscriber(KEY, on_sample)
    ready.set()
    done.wait(timeout=warmup / 30.0 + n / 30.0 + 10.0)

    with lock:
        vals = sorted(v for i, v in lat if i >= warmup)
        result["latencies_ms"] = vals
        result["shm_hits"] = shm_hits
        result["total"] = total
    sub.undeclare()
    session.close()


def _run_mode(mode: str, args: argparse.Namespace) -> dict[str, Any]:
    frame_bytes = args.width * args.height * 3
    port = _find_free_port()
    endpoint = f"tcp/127.0.0.1:{port}"

    pub = ZenohBackend(
        listen=[endpoint],
        shared_memory=(mode == "shm"),
        shm_pool_bytes=max(64 * 1024 * 1024, frame_bytes * 48),
    )
    if mode == "shm" and not pub.shm_enabled:
        # Don't crash: the backend itself falls back to the copy path when the
        # pool can't be created, so do the same here. Most hosts running this
        # without `ipc: host` + `ulimits: memlock: -1` hit this — the run below
        # still completes, just without a zero-copy leg to compare against.
        print(
            "[shm] SHM pool unavailable on this host (needs `ipc: host` + "
            "`ulimits: memlock: -1` — see docs-mintlify edge-worker.mdx) — "
            "falling back to the copy path for this run.",
            flush=True,
        )

    ready = threading.Event()
    result: dict[str, Any] = {}
    n = args.count
    warmup = args.warmup
    t = threading.Thread(
        target=_consumer,
        kwargs=dict(
            endpoint=endpoint, n=n + warmup, warmup=warmup, ready=ready, result=result
        ),
        daemon=True,
    )
    t.start()
    ready.wait()
    time.sleep(0.5)  # let the link come up + SHM negotiate

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    buf = bytearray(frame_bytes)
    for i in range(n + warmup):
        struct.pack_into(HDR, buf, 0, MAGIC, i, time.monotonic())
        pub.publish(KEY, bytes(buf))
        if interval:
            time.sleep(interval)

    t.join(timeout=15.0)
    stats = pub.stats()
    result["shm_frames"] = stats["shm_frames"]
    result["copy_frames"] = stats["copy_frames"]
    result["frame_bytes"] = frame_bytes
    pub.close()
    return result


def _summarize(mode: str, r: dict[str, Any]) -> dict[str, float]:
    vals = r.get("latencies_ms", [])
    if not vals:
        print(f"[{mode}] no frames received")
        return {}

    def pct(p: float) -> float:
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    summary = {
        "mean": statistics.fmean(vals),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": vals[-1],
    }
    print(
        f"[{mode}] n={len(vals)} frame={r['frame_bytes'] / 1e6:.2f}MB  "
        f"shm_backed={r['shm_hits']}/{r['total']}  "
        f"(publisher shm_frames={r['shm_frames']} copy_frames={r['copy_frames']})"
    )
    print(
        f"       mean={summary['mean']:.3f}ms  p50={summary['p50']:.3f}  "
        f"p90={summary['p90']:.3f}  p99={summary['p99']:.3f}  "
        f"max={summary['max']:.3f}ms"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenoh SHM zero-copy latency demo")
    parser.add_argument("--mode", choices=["shm", "copy", "compare"], default="compare")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--count", type=int, default=360, help="measured frames")
    parser.add_argument("--warmup", type=int, default=40)
    args = parser.parse_args()

    modes = ["copy", "shm"] if args.mode == "compare" else [args.mode]
    summaries: dict[str, dict[str, float]] = {}
    for mode in modes:
        summaries[mode] = _summarize(mode, _run_mode(mode, args))

    if args.mode == "compare" and summaries.get("copy") and summaries.get("shm"):
        print("\n── gain (copy → shm) ──")
        for k in ("mean", "p50", "p90", "p99", "max"):
            c, s = summaries["copy"][k], summaries["shm"][k]
            if s > 0:
                print(f"  {k:>4}: {c:.3f} → {s:.3f} ms   ({c / s:.1f}× lower)")


if __name__ == "__main__":
    main()
