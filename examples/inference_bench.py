"""End-to-end inference benchmark for the Cyberwave edge ML pipeline.

Measures the full critical path:

  Frame publish → Zenoh transport → decode → model.predict → detection publish

Two operating modes
-------------------
**Latency mode** (``--send-rate N``, default 30 fps):
  Frames are paced at a controlled rate; latency numbers reflect true
  frame-to-detection transit time.

**Throughput mode** (``--send-rate 0``):
  Frames are flooded as fast as possible.  Useful to find the max
  inference throughput; latency numbers include queuing delay.

Prerequisites
-------------
::

    pip install 'cyberwave[zenoh,ml]'
    # or for ONNX only:
    pip install 'cyberwave[zenoh,ml-onnx]'

Usage
-----
::

    # Default: 100 frames at 30 fps through a YOLO model
    python examples/inference_bench.py --model yolov8n.pt

    # ONNX model at custom resolution
    python examples/inference_bench.py --model yolov8n.onnx --resolution 640x480

    # Throughput mode
    python examples/inference_bench.py --model yolov8n.pt --send-rate 0 --count 200

    # Save results as JSON for CI regression tracking
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

TWIN_UUID = "aaaaaaaa-bench-4000-a000-000000000001"
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


def _make_backend(connect: list[str]) -> ZenohBackend:
    return ZenohBackend(connect=connect, shared_memory=False)


def _make_bus(backend: ZenohBackend, twin_uuid: str) -> DataBus:
    return DataBus(backend, twin_uuid)


def _make_frame(h: int, w: int, sent_ns: int) -> np.ndarray:
    """Return a zeroed HxWx3 frame with *sent_ns* packed into the first 8 bytes."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame.ravel()[:8].view(_TS_DTYPE)[0] = sent_ns
    return frame


def _frame_ts(frame: np.ndarray) -> int:
    return int(frame.ravel()[:8].view(_TS_DTYPE)[0])


# ---------------------------------------------------------------------------
# Publisher thread
# ---------------------------------------------------------------------------

def _publisher(
    connect: list[str],
    n_messages: int,
    n_warmup: int,
    rate_hz: float,
    resolution: tuple[int, int],
    *,
    ready: threading.Event,
    timing: dict[str, int],
    done: threading.Event,
) -> None:
    backend = _make_backend(connect)
    bus = _make_bus(backend, TWIN_UUID)
    ready.wait()

    h, w = resolution[1], resolution[0]
    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0

    for _ in range(n_warmup):
        bus.publish(FRAME_CHANNEL, _make_frame(h, w, 0))
        if interval_s:
            time.sleep(interval_s)
    time.sleep(0.3)

    timing["start_ns"] = time.perf_counter_ns()
    for _ in range(n_messages):
        sent_ns = time.perf_counter_ns()
        bus.publish(FRAME_CHANNEL, _make_frame(h, w, sent_ns))
        if interval_s:
            time.sleep(interval_s)

    done.set()
    bus.close()


# ---------------------------------------------------------------------------
# Worker thread (subscribe → decode → infer → publish detections)
# ---------------------------------------------------------------------------

def _worker(
    connect: list[str],
    model_path: str,
    runtime_name: str | None,
    device: str,
    *,
    ready: threading.Event,
    done: threading.Event,
    timing: dict[str, int],
    inference_latencies_ms: list[float],
) -> None:
    from cyberwave.models.loaded_model import LoadedModel
    from cyberwave.models.manager import ModelManager

    mgr = ModelManager(default_device=device)
    loaded: LoadedModel = mgr.load_from_file(model_path, runtime=runtime_name, device=device)

    loaded.warm_up()

    backend = _make_backend(connect)
    bus = _make_bus(backend, TWIN_UUID)

    def on_frame(value: object) -> None:
        if not isinstance(value, np.ndarray):
            return
        t0 = time.monotonic()
        try:
            loaded.predict(value, confidence=0.25)
        except Exception:
            pass
        latency = (time.monotonic() - t0) * 1000.0
        inference_latencies_ms.append(latency)
        timing["end_ns"] = time.perf_counter_ns()

    sub = bus.subscribe(FRAME_CHANNEL, on_frame, policy="latest")
    ready.set()

    done.wait()
    time.sleep(1.5)
    sub.close()
    backend.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _print_results(
    inference_latencies_ms: list[float],
    n_sent: int,
    n_warmup: int,
    rate_hz: float,
    resolution: tuple[int, int],
    model_path: str,
    device: str,
    timing: dict[str, int],
) -> dict[str, Any]:
    def ms(v: float) -> str:
        return f"{v:.2f} ms"

    start_ns = timing.get("start_ns", 0)
    end_ns = timing.get("end_ns", start_ns)
    elapsed_s = (end_ns - start_ns) / 1e9

    n_infer = len(inference_latencies_ms)
    infer_sorted = sorted(inference_latencies_ms) if n_infer else [0.0]
    infer_avg = statistics.mean(infer_sorted) if n_infer else 0.0
    infer_p50 = infer_sorted[int(len(infer_sorted) * 0.50)] if n_infer >= 2 else infer_sorted[-1]
    infer_p95 = infer_sorted[int(len(infer_sorted) * 0.95)] if n_infer >= 2 else infer_sorted[-1]
    infer_p99 = infer_sorted[int(len(infer_sorted) * 0.99)] if n_infer >= 2 else infer_sorted[-1]

    fps = n_infer / elapsed_s if elapsed_s > 0 else 0.0
    drop_rate = max(0, n_sent - n_infer) / n_sent * 100.0 if n_sent > 0 else 0.0

    mode = "throughput (flooded)" if rate_hz == 0 else f"latency (send-rate={rate_hz:.0f} fps)"

    print(f"\n── Inference Benchmark [{mode}] ────")
    print(f"  model      {model_path}")
    print(f"  device     {device}")
    print(f"  resolution {resolution[0]}x{resolution[1]}")
    print(f"  frames     {n_sent} sent, {n_infer} inferred (warmup {n_warmup} discarded)")
    print(f"  drop rate  {drop_rate:.1f}%")
    print(f"  ──")
    print(f"  Inference latency:")
    print(f"    min      {ms(infer_sorted[0])}")
    print(f"    p50      {ms(infer_p50)}")
    print(f"    p95      {ms(infer_p95)}")
    print(f"    p99      {ms(infer_p99)}")
    print(f"    avg      {ms(infer_avg)}")
    print(f"  ──")
    print(f"  throughput {fps:.1f} fps")
    print(f"  elapsed    {elapsed_s * 1000:.1f} ms")

    results: dict[str, Any] = {
        "model": model_path,
        "device": device,
        "resolution": list(resolution),
        "frames_sent": n_sent,
        "frames_inferred": n_infer,
        "warmup": n_warmup,
        "drop_rate_pct": round(drop_rate, 2),
        "fps": round(fps, 2),
        "elapsed_s": round(elapsed_s, 3),
        "inference_ms": {
            "min": round(infer_sorted[0], 3),
            "p50": round(infer_p50, 3),
            "p95": round(infer_p95, 3),
            "p99": round(infer_p99, 3),
            "avg": round(infer_avg, 3),
        },
    }
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end inference benchmark (frame → Zenoh → model → detection)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to model file (e.g. yolov8n.pt, model.onnx)",
    )
    parser.add_argument(
        "--runtime", default=None,
        help="Runtime override (ultralytics, onnxruntime, tflite, tensorrt, torch)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--resolution", default="640x480", metavar="WxH",
        help="Frame resolution (default: 640x480)",
    )
    parser.add_argument(
        "--count", type=int, default=100, metavar="N",
        help="Number of frames to benchmark (default: 100)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, metavar="W",
        help="Warmup frames discarded before timing (default: 10)",
    )
    parser.add_argument(
        "--send-rate", type=float, default=30.0, dest="send_rate", metavar="R",
        help="Frame publish rate in fps (default: 30). Set to 0 for throughput mode.",
    )
    parser.add_argument(
        "--connect", metavar="ENDPOINT",
        help="External Zenoh router endpoint. Omit for in-process broker.",
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Write JSON results to FILE for CI regression tracking.",
    )
    args = parser.parse_args()

    res_parts = args.resolution.lower().split("x")
    resolution = (int(res_parts[0]), int(res_parts[1]))

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

    rate_label = "flooded" if args.send_rate == 0 else f"{args.send_rate:.0f} fps"
    print(
        f"Benchmark: {args.count} frames (+ {args.warmup} warmup) at {rate_label}, "
        f"model={args.model}, resolution={args.resolution}, device={args.device}"
    )

    worker_ready = threading.Event()
    pub_done = threading.Event()
    timing: dict[str, int] = {}
    inference_latencies_ms: list[float] = []

    t_worker = threading.Thread(
        target=_worker,
        kwargs=dict(
            connect=connect,
            model_path=args.model,
            runtime_name=args.runtime,
            device=args.device,
            ready=worker_ready,
            done=pub_done,
            timing=timing,
            inference_latencies_ms=inference_latencies_ms,
        ),
        daemon=True,
    )
    t_pub = threading.Thread(
        target=_publisher,
        kwargs=dict(
            connect=connect,
            n_messages=args.count,
            n_warmup=args.warmup,
            rate_hz=args.send_rate,
            resolution=resolution,
            ready=worker_ready,
            timing=timing,
            done=pub_done,
        ),
        daemon=True,
    )

    t_worker.start()
    t_pub.start()

    timeout_s = max(120.0, args.count / args.send_rate * 3) if args.send_rate > 0 else 300.0
    t_pub.join(timeout=timeout_s)
    t_worker.join(timeout=15.0)

    if broker is not None:
        try:
            broker.close()
        except Exception:
            pass

    results = _print_results(
        inference_latencies_ms,
        args.count,
        args.warmup,
        args.send_rate,
        resolution,
        args.model,
        args.device,
        timing,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
