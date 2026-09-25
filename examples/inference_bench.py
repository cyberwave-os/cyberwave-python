"""
Inference Benchmark — end-to-end frame → Zenoh → model.predict → detection pipeline.

Requirements:
    pip install cyberwave[zenoh,ml]

Usage:
    python examples/inference_bench.py --model yolov8n.pt
    python examples/inference_bench.py --model yolov8n.pt --send-rate 0 --count 200
    python examples/inference_bench.py --model yolov8n.pt --output results.json
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

TWIN_UUID = "aaaaaaaa-aaaa-4000-a000-000000000001"
FRAME_CHANNEL = "frames"
_TS_DTYPE = np.dtype("<u8")


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


def _make_frame(h: int, w: int, sent_ns: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame.ravel()[:8].view(_TS_DTYPE)[0] = sent_ns
    return frame


def _publisher(
    connect: list[str],
    n: int,
    warmup: int,
    rate: float,
    res: tuple[int, int],
    *,
    ready: threading.Event,
    timing: dict[str, int],
    done: threading.Event,
) -> None:
    bus = DataBus(_backend(connect), TWIN_UUID)
    h, w = res[1], res[0]
    interval = 1 / rate if rate > 0 else 0
    ready.wait()

    for _ in range(warmup):
        bus.publish(FRAME_CHANNEL, _make_frame(h, w, 0))
        if interval:
            time.sleep(interval)
    time.sleep(0.3)

    timing["start_ns"] = time.perf_counter_ns()
    for _ in range(n):
        bus.publish(FRAME_CHANNEL, _make_frame(h, w, time.perf_counter_ns()))
        if interval:
            time.sleep(interval)
    done.set()
    bus.close()


def _worker(
    connect: list[str],
    model_path: str,
    runtime: str | None,
    device: str,
    *,
    ready: threading.Event,
    done: threading.Event,
    timing: dict[str, int],
    latencies: list[float],
) -> None:
    from cyberwave.models.manager import ModelManager

    mgr = ModelManager(default_device=device)
    loaded = mgr.load_from_file(model_path, runtime=runtime, device=device)
    loaded.warm_up()

    bus = DataBus(_backend(connect), TWIN_UUID)

    def on_frame(value: object) -> None:
        if not isinstance(value, np.ndarray):
            return
        t0 = time.monotonic()
        try:
            loaded.predict(value, confidence=0.25)
        except Exception:
            pass
        latencies.append((time.monotonic() - t0) * 1000)
        timing["end_ns"] = time.perf_counter_ns()

    sub = bus.subscribe(FRAME_CHANNEL, on_frame, policy="latest")
    ready.set()
    done.wait()
    time.sleep(1.5)
    sub.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end inference benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resolution", default="640x480", metavar="WxH")
    parser.add_argument("--count", type=int, default=100, metavar="N")
    parser.add_argument("--warmup", type=int, default=10, metavar="W")
    parser.add_argument(
        "--send-rate", type=float, default=30.0, dest="send_rate", metavar="R"
    )
    parser.add_argument("--connect", metavar="ENDPOINT")
    parser.add_argument("--output", metavar="FILE")
    args = parser.parse_args()

    res = tuple(int(x) for x in args.resolution.lower().split("x"))

    broker = None
    if args.connect:
        connect = [args.connect]
    else:
        port = _find_free_port()
        broker = _start_broker(port)
        connect = [f"tcp/127.0.0.1:{port}"]
    time.sleep(0.2)

    rate_label = "flooded" if args.send_rate == 0 else f"{args.send_rate:.0f} fps"
    print(
        f"Benchmark: {args.count} frames at {rate_label}, "
        f"model={args.model}, res={args.resolution}"
    )

    worker_ready, pub_done = threading.Event(), threading.Event()
    timing: dict[str, int] = {}
    latencies: list[float] = []

    t_w = threading.Thread(
        target=_worker,
        kwargs=dict(
            connect=connect,
            model_path=args.model,
            runtime=args.runtime,
            device=args.device,
            ready=worker_ready,
            done=pub_done,
            timing=timing,
            latencies=latencies,
        ),
        daemon=True,
    )
    t_p = threading.Thread(
        target=_publisher,
        kwargs=dict(
            connect=connect,
            n=args.count,
            warmup=args.warmup,
            rate=args.send_rate,
            res=res,
            ready=worker_ready,
            timing=timing,
            done=pub_done,
        ),
        daemon=True,
    )
    t_w.start()
    t_p.start()

    timeout = max(120, args.count / args.send_rate * 3) if args.send_rate > 0 else 300
    t_p.join(timeout=timeout)
    t_w.join(timeout=15)
    if broker:
        broker.close()

    n = len(latencies)
    if n == 0:
        print("No inferences completed.")
        return

    s = sorted(latencies)
    elapsed_s = (timing.get("end_ns", 0) - timing.get("start_ns", 0)) / 1e9
    fps = n / elapsed_s if elapsed_s > 0 else 0

    print(f"\n── Results ({n}/{args.count} frames) ──")
    print(
        f"  p50={s[int(n * 0.5)]:.2f}ms  "
        f"p95={s[int(n * 0.95)]:.2f}ms  "
        f"p99={s[int(n * 0.99)]:.2f}ms"
    )
    print(f"  avg={statistics.mean(s):.2f}ms  min={s[0]:.2f}ms")
    print(
        f"  throughput={fps:.1f} fps  "
        f"drop={max(0, args.count - n) / args.count * 100:.1f}%"
    )

    if args.output:
        results = {
            "model": args.model,
            "device": args.device,
            "resolution": list(res),
            "frames_sent": args.count,
            "frames_inferred": n,
            "fps": round(fps, 2),
            "inference_ms": {
                "p50": round(s[int(n * 0.5)], 3),
                "p95": round(s[int(n * 0.95)], 3),
                "avg": round(statistics.mean(s), 3),
            },
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
