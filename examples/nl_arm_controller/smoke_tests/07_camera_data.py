"""Smoke test 7 — Mac webcam end-to-end.

Confirms that we can open the USB webcam (or built-in FaceTime camera)
attached to the Mac, read frames at the requested resolution, and encode
them as base64 JPEG for Claude Vision.

No Pi, no network, no SDK — just OpenCV reading from /dev/video* via the
macOS AVFoundation backend.

Run:
    python smoke_tests/07_camera_data.py            # use CW_CAMERA_INDEX from .env
    python smoke_tests/07_camera_data.py --index 1  # try a specific webcam

Expected output:
    → Opening webcam (index 0)…
      ✓ camera ready: 1280x720 @ 30 fps
    [ 1/10] frame 1280x720  rms=92.4   grab+encode=18 ms   b64=42 KB
    [ 2/10] frame 1280x720  rms=93.1   grab+encode=14 ms   b64=43 KB
    ...
    → saved /tmp/nl_arm_camera_smoke.jpg   (open it to verify the picture)
    ✅ webcam OK   (10/10 samples)

If macOS pops up a "Terminal would like to access the camera" prompt, click
Allow and re-run. After the first time you'll never be asked again.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

from vision import open_camera, open_camera_from_env  # noqa: E402

N_SAMPLES = 10
SAMPLE_INTERVAL_S = 0.3
SAVE_PATH = Path("/tmp/nl_arm_camera_smoke.jpg")


def _parse_index() -> int | None:
    """Allow `--index N` override on the CLI."""
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        if i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                print(f"❌ --index must be an integer, got {sys.argv[i + 1]!r}")
                sys.exit(2)
    return None


def main() -> None:
    index_override = _parse_index()
    print(
        f"→ Opening webcam "
        f"(index {index_override if index_override is not None else 'from .env'})…"
    )

    try:
        cam = (
            open_camera(index_override)
            if index_override is not None
            else open_camera_from_env()
        )
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    print(
        f"  ✓ camera ready: {cam.info.width}x{cam.info.height} "
        f"@ {cam.info.fps:.0f} fps  (index {cam.info.index})"
    )

    successes = 0
    last_frame = None
    last_b64_size = 0

    try:
        import numpy as np

        for i in range(1, N_SAMPLES + 1):
            t0 = time.monotonic()
            frame = cam.grab_frame()
            if frame is None:
                print(f"  [{i:>2}/{N_SAMPLES}] frame read failed")
                time.sleep(SAMPLE_INTERVAL_S)
                continue

            b64 = cam.grab_frame_b64(quality=80)  # round-trips through encoder
            dt_ms = (time.monotonic() - t0) * 1000
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            h, w = frame.shape[:2]
            successes += 1
            last_frame = frame
            last_b64_size = len(b64) if b64 else 0

            print(
                f"  [{i:>2}/{N_SAMPLES}] frame {w}x{h}  "
                f"rms={rms:5.1f}  grab+encode={dt_ms:4.0f} ms  "
                f"b64={last_b64_size // 1024} KB"
            )
            time.sleep(SAMPLE_INTERVAL_S)
    finally:
        cam.close()

    if last_frame is None:
        print()
        print("❌ Never read a frame from the webcam.")
        print("   Things to check:")
        print("   • macOS Camera permission for your terminal app")
        print("     (System Settings → Privacy & Security → Camera)")
        print("   • Try a different --index (0, 1, 2 …)")
        print("   • Make sure no other app is currently using the camera")
        sys.exit(1)

    try:
        import cv2

        cv2.imwrite(str(SAVE_PATH), last_frame)
        print(f"\n→ saved {SAVE_PATH}   (open it to verify the picture)")
    except Exception as exc:
        print(f"\n⚠️  could not save debug image: {exc}")

    rate = successes / N_SAMPLES
    if rate < 0.5:
        print(f"\n⚠️  only {successes}/{N_SAMPLES} samples succeeded ({rate:.0%})")
        sys.exit(1)

    print(f"\n✅ webcam OK   ({successes}/{N_SAMPLES} samples)")


if __name__ == "__main__":
    main()
