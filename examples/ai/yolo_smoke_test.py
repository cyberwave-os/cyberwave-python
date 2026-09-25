"""
YOLO Smoke Test — batch-test YOLO models against a single image.

Works locally without an API key. Ultralytics auto-downloads weights.

Usage:
    python yolo_smoke_test.py
    python yolo_smoke_test.py --image path/to/frame.jpg
    python yolo_smoke_test.py --models yolov8n.pt yolo11n.pt

Requirements:
    pip install cyberwave[ml] Pillow
"""

from __future__ import annotations

import argparse
import pathlib
import time

_BUILTIN_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8n-seg.pt",
    "yolov8n-pose.pt",
    "yolov8n-cls.pt",
    "yolov8n-obb.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11n-seg.pt",
    "yolo11n-pose.pt",
    "yolo11n-cls.pt",
    "yolo11n-obb.pt",
    "yolo26n.pt",
    "yolo26s.pt",
]

_HERE = pathlib.Path(__file__).parent
_DEFAULT_IMAGE = _HERE / "engineers-humans.png"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", default=str(_DEFAULT_IMAGE), metavar="PATH")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--models", nargs="+", default=None, metavar="ID")
    args = ap.parse_args()

    from PIL import Image as PILImage

    image = PILImage.open(args.image).convert("RGB")
    model_ids = args.models or _BUILTIN_MODELS

    print(f"Testing {len(model_ids)} model(s) on {pathlib.Path(args.image).name}\n")
    print(f"{'Model':<28} {'Status':<6} {'N':>4} {'Time':>7}")
    print("-" * 50)

    passed = failed = 0
    for model_id in model_ids:
        t0 = time.perf_counter()
        try:
            from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime

            rt = UltralyticsRuntime()
            handle = rt.load(model_id)
            pred = rt.predict(handle, image, confidence=args.conf)
            n = len(pred.detections) if hasattr(pred, "detections") else len(pred)
            elapsed = time.perf_counter() - t0
            print(f"{model_id:<28} {'PASS':<6} {n:>4} {elapsed:.2f}s")
            passed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"{model_id:<28} {'FAIL':<6} {'—':>4} {elapsed:.2f}s")
            print(f"    {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\nPASSED {passed}/{len(model_ids)}  FAILED {failed}/{len(model_ids)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
