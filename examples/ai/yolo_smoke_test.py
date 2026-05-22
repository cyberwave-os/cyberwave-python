"""Batch smoke-test every YOLO model against a single image.

Loads each model, runs inference, and reports a pass/fail summary.
Works entirely locally — no Cyberwave API key required when using the
built-in model list, because Ultralytics auto-downloads weights on first use.

Usage:
    python yolo_smoke_test.py
    python yolo_smoke_test.py --image path/to/frame.jpg
    python yolo_smoke_test.py --image path/to/frame.jpg --conf 0.25
    python yolo_smoke_test.py --models yolov8n.pt yolo11n.pt yolo11n-seg.pt

If a CYBERWAVE_API_KEY is set the script also pulls YOLO entries from the
workspace catalog and appends them to the test list (deduplicated by load ID).
"""

from __future__ import annotations

import argparse
import pathlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Well-known YOLO model IDs that Ultralytics auto-downloads.
# Covers detect / segment / pose / OBB / classify across v8, v11, v26.
# ---------------------------------------------------------------------------
_BUILTIN_MODELS: list[str] = [
    # YOLOv8 — detect
    "yolov8n.pt",
    "yolov8s.pt",
    # YOLOv8 — segment
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    # YOLOv8 — pose
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    # YOLOv8 — classify
    "yolov8n-cls.pt",
    # YOLOv8 — OBB
    "yolov8n-obb.pt",
    # YOLO11 — detect
    "yolo11n.pt",
    "yolo11s.pt",
    # YOLO11 — segment
    "yolo11n-seg.pt",
    # YOLO11 — pose
    "yolo11n-pose.pt",
    # YOLO11 — classify
    "yolo11n-cls.pt",
    # YOLO11 — OBB
    "yolo11n-obb.pt",
    # YOLO26 — detect
    "yolo26n.pt",
    "yolo26s.pt",
]

_HERE = pathlib.Path(__file__).parent
_DEFAULT_IMAGE = _HERE / "engineers-humans.png"


@dataclass
class ModelResult:
    model_id: str
    passed: bool
    elapsed_s: float
    detections: int = 0
    output_type: str = ""
    error: str = ""
    detail: str = field(default="", repr=False)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _catalog_yolo_ids() -> list[str]:
    """Return sdk_load_ids of YOLO entries from the workspace catalog.

    Returns [] silently when CYBERWAVE_API_KEY is not configured or the
    catalog is unreachable.
    """
    try:
        from cyberwave import Cyberwave
        cw = Cyberwave()
        models = cw.models.list()
        return [
            m.sdk_load_id
            for m in models
            if m.sdk_load_id and "yolo" in (m.sdk_load_id or "").lower()
        ]
    except Exception:
        return []


def _run_one(model_id: str, image: Any, *, conf: float) -> ModelResult:
    t0 = time.perf_counter()
    try:
        from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime

        rt = UltralyticsRuntime()
        if not rt.is_available():
            return ModelResult(
                model_id=model_id,
                passed=False,
                elapsed_s=0.0,
                error="ultralytics not installed — pip install ultralytics",
            )

        handle = rt.load(model_id)
        pred = rt.predict(handle, image, confidence=conf)
        elapsed = time.perf_counter() - t0

        output = pred.output
        output_type = type(output).__name__ if output is not None else "none"
        count = len(output) if output is not None else 0
        if count:
            detail = pred.describe()
        elif pred.classification and pred.classification.top1:
            cls = pred.classification.top1
            detail = f"top1={cls.label!r} conf={cls.confidence:.3f}"
        else:
            detail = "(no results above threshold)"

        return ModelResult(
            model_id=model_id,
            passed=True,
            elapsed_s=elapsed,
            detections=count,
            output_type=output_type,
            detail=detail,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return ModelResult(
            model_id=model_id,
            passed=False,
            elapsed_s=elapsed,
            error=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--image",
        default=str(_DEFAULT_IMAGE),
        metavar="PATH",
        help=f"Image to run inference on (default: {_DEFAULT_IMAGE.name})",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="Confidence threshold (default: 0.25)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="ID",
        help="Override the model list with explicit SDK load IDs",
    )
    ap.add_argument(
        "--no-catalog",
        action="store_true",
        help="Skip catalog lookup even if CYBERWAVE_API_KEY is set",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-detection lines for passing models",
    )
    args = ap.parse_args()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    try:
        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGB")
    except ImportError:
        raise SystemExit("Pillow is required — pip install Pillow")

    if args.models:
        model_ids = list(args.models)
    else:
        model_ids = list(_BUILTIN_MODELS)
        if not args.no_catalog:
            catalog_ids = _catalog_yolo_ids()
            known = set(model_ids)
            extras = [mid for mid in catalog_ids if mid not in known]
            if extras:
                print(f"[catalog] appending {len(extras)} additional model(s): {extras}")
            model_ids.extend(extras)

    total = len(model_ids)
    print(f"\nSmoke-testing {total} YOLO model(s) on: {image_path.name}")
    print(f"Confidence threshold: {args.conf}\n")
    print(f"{'Model':<28} {'Status':<6} {'Type':<20} {'N':>4} {'Time':>7}")
    print("-" * 72)

    results: list[ModelResult] = []
    for model_id in model_ids:
        r = _run_one(model_id, image, conf=args.conf)
        results.append(r)
        n_col = str(r.detections) if r.passed else "—"
        print(
            f"{model_id:<28} {r.status:<6} {(r.output_type or '—'):<20}"
            f" {n_col:>4} {r.elapsed_s:.2f}s"
        )
        if r.passed and args.verbose and r.detail:
            for line in r.detail.splitlines():
                print(f"    {line}")
        elif not r.passed:
            print(f"    ↳ {r.error}")

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    print("\n" + "=" * 72)
    print(f"PASSED  {len(passed):>3} / {total}")
    print(f"FAILED  {len(failed):>3} / {total}")

    if passed:
        print(f"\n✓ Passing ({len(passed)}):")
        for r in passed:
            print(
                f"  {r.model_id:<28}  {r.output_type:<20}"
                f"  {r.detections} result(s)  {r.elapsed_s:.2f}s"
            )

    if failed:
        print(f"\n✗ Failing ({len(failed)}):")
        for r in failed:
            print(f"  {r.model_id:<28}  {r.error}")
        if not args.verbose:
            print("\n  (run with --verbose to see full tracebacks)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()