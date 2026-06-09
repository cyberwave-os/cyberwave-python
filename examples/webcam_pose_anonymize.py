"""
Webcam Pose Anonymisation — live webcam with person detection and anonymisation overlay.

Hotkeys: q=quit, m=cycle mode (pixelate/redact/blur/bbox), s=toggle skeleton.

Requirements:
    pip install cyberwave[ml] opencv-python
"""

from __future__ import annotations

import sys
import time

import cv2

from cyberwave.models.manager import ModelManager
from cyberwave.vision import anonymize_frame

MODES = ("pixelate", "redact", "blur", "bbox")


def main() -> int:
    mm = ModelManager()
    model = mm.load("yolov8n-pose.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera", file=sys.stderr)
        return 1

    mode = "pixelate"
    show_skel = True
    last_log, frames = time.monotonic(), 0

    print("Hotkeys: q=quit  m=cycle mode  s=toggle skeleton")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            result = model.predict(frame, classes=["person"], confidence=0.4)

            out = anonymize_frame(
                frame,
                result.detections,
                mode=mode,
                skeleton_threshold=0.3 if show_skel else 1.1,
            )

            cv2.putText(
                out,
                f"mode={mode} dets={len(result.detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.imshow("cyberwave anonymise", out)

            frames += 1
            now = time.monotonic()
            if now - last_log >= 2.0:
                print(
                    f"fps={frames / (now - last_log):.1f}  "
                    f"dets={len(result.detections)}"
                )
                last_log, frames = now, 0

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            if k == ord("s"):
                show_skel = not show_skel
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
