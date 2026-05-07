"""Webcam pose-aware anonymisation viewer.

Streams from a local webcam, runs a YOLOv8 pose model, and shows the
anonymised frame (mosaic / redact / blur / solid fill) with a
colour-coded skeleton overlay. Useful for sanity-checking the SDK's
vision pipeline end-to-end on a real camera before wiring it into a
worker.

Demonstrates:

- ``cyberwave.models.ModelManager.load`` — auto-downloads YOLOv8 weights.
- ``UltralyticsRuntime.predict`` populating ``Detection.keypoints``.
- ``cyberwave.vision.{anonymize_frame, blank_persons, draw_skeleton}``.

Hotkeys:

  q       quit
  m       cycle anonymisation mode (pixelate -> redact -> blur -> bbox)
  a       toggle anonymisation off/on (off = skeleton on raw frame —
          useful to judge pose quality without the mosaic in the way)
  s       toggle skeleton overlay
  +/-     raise / lower the visibility threshold for the skeleton

Tip: ``yolov8n-pose.pt`` is the smallest variant (fast, lower accuracy).
For visibly better keypoints try ``--model yolov8s-pose.pt`` (~22MB,
~3x slower on CPU but still real-time on a modern Mac) or
``yolov8m-pose.pt`` (~52MB, ~5-6x slower).

Run from the SDK directory:

  python examples/webcam_pose_anonymize.py

Requires ``opencv-python`` and ``ultralytics`` (``pip install
cyberwave[ml]`` plus ``opencv-python``).

Model weights are cached under ``~/.cyberwave/models/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from cyberwave.models.manager import ModelManager
from cyberwave.vision import anonymize_frame, blank_persons, draw_skeleton

MODES = ("pixelate", "redact", "blur", "bbox")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default 0)"
    )
    p.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help=(
            "Model file or catalog id "
            "(default: yolov8n-pose.pt — auto-downloaded on first run; "
            "try yolov8s-pose.pt or yolov8m-pose.pt for better keypoints)"
        ),
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default 0.4)",
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="pixelate",
        help="Initial anonymisation mode (default: pixelate)",
    )
    p.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Start with skeleton overlay off",
    )
    p.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Start with anonymisation off (skeleton drawn on raw frame)",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    mm = ModelManager()
    if Path(args.model).is_file():
        # Explicit on-disk path — bypass the catalog lookup.
        print(f"Loading model from file: {args.model}")
        model = mm.load_from_file(args.model)
    else:
        # Catalog path — for ultralytics this also triggers auto-download
        # via the YOLO() wrapper into ~/.cyberwave/models on first run.
        print(f"Loading model from catalog id: {args.model}")
        model = mm.load(args.model)
    print(f"Loaded with runtime={model.runtime}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}", file=sys.stderr)
        return 1

    mode = args.mode
    show_skel = not args.no_skeleton
    anonymize = not args.no_anonymize
    skel_thresh = 0.3
    last_log, frames, infer_total = time.monotonic(), 0, 0.0

    print(
        "Hotkeys: q=quit  m=cycle mode  a=toggle anonymise  "
        "s=toggle skeleton  +/-=tweak threshold"
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                break

            # Some webcams / v4l2 backends return RGBA or single-channel
            # frames depending on driver quirks; the SDK helpers expect
            # a 3-channel BGR uint8 image. Convert / refuse early so the
            # error message points at the camera rather than at YOLO.
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                print(
                    f"unexpected frame shape {frame.shape} (need HxWx3 or HxWx4)",
                    file=sys.stderr,
                )
                break
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.dtype != "uint8":
                frame = frame.astype("uint8")

            t0 = time.monotonic()
            result = model.predict(
                frame, classes=["person"], confidence=args.confidence
            )
            infer_total += time.monotonic() - t0

            # Compose the output frame depending on the toggles.
            if anonymize and show_skel:
                out = anonymize_frame(
                    frame,
                    result.detections,
                    mode=mode,
                    skeleton_threshold=skel_thresh,
                )
            elif anonymize:
                out = blank_persons(frame, result.detections, mode=mode)
            elif show_skel:
                # Skeleton-only on raw frame — useful for judging pose quality.
                out = frame.copy()
                for det in result.detections:
                    if det.label == "person" and det.keypoints is not None:
                        draw_skeleton(
                            out,
                            det.keypoints,
                            conf_threshold=skel_thresh,
                            inplace=True,
                        )
            else:
                out = frame.copy()

            kp_count = sum(1 for d in result.detections if d.keypoints is not None)
            hud = (
                f"mode={mode}  anon={anonymize}  skel={show_skel}  "
                f"thresh={skel_thresh:.2f}  dets={len(result.detections)}  "
                f"with_kps={kp_count}"
            )
            cv2.putText(
                out,
                hud,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow("cyberwave anonymise (q m a s +/-)", out)

            frames += 1
            now = time.monotonic()
            if now - last_log >= 2.0:
                fps = frames / (now - last_log)
                avg_ms = (infer_total / frames) * 1000 if frames else 0.0
                print(
                    f"fps={fps:5.1f}  inference={avg_ms:5.1f} ms  "
                    f"last-frame dets={len(result.detections)} with_kps={kp_count}"
                )
                last_log, frames, infer_total = now, 0, 0.0

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            if k == ord("a"):
                anonymize = not anonymize
            if k == ord("s"):
                show_skel = not show_skel
            if k in (ord("+"), ord("=")):
                skel_thresh = min(0.95, skel_thresh + 0.05)
            if k in (ord("-"), ord("_")):
                skel_thresh = max(0.0, skel_thresh - 0.05)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
