"""Vision module — grab frames from a webcam attached to the Mac.

Companion to `voice.py` (which handles audio + STT). This module owns the
camera handle for the agent loop: it opens a `cv2.VideoCapture` at startup,
keeps it warm for the whole session, and returns frames as base64 JPEG on
demand for Claude Vision.

We deliberately keep this simple — no MQTT, no data bus, no Zenoh, no HTTP.
The webcam plugs into the Mac, OpenCV reads it directly, and `nl_arm_controller`
calls `grab_frame_b64()` once per turn.

Public API:
  open_camera(index, width, height, fps)   → Camera   (raises if it can't open)
  Camera.grab_frame_b64(quality)           → str | None  (None if read fails)
  Camera.grab_frame()                      → np.ndarray | None  (raw BGR)
  Camera.close()

Environment variables (read by `open_camera_from_env()`):
  CW_CAMERA_INDEX    integer device index, default 0  (0 = first webcam)
  CW_CAMERA_WIDTH    requested capture width,  default 1280
  CW_CAMERA_HEIGHT   requested capture height, default 720
  CW_CAMERA_JPEG_QUALITY  default 80  (1–100)

macOS notes:
  * The first run will trigger the system Camera permission prompt for the
    terminal app. Click Allow, then re-run.
  * Built-in FaceTime camera is index 0 if it's the only camera. With a
    second USB webcam plugged in, the USB cam is usually index 1 (but order
    is not guaranteed — try 0, 1, 2 if you're unsure).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass


DEFAULT_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_QUALITY = 80


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float


class Camera:
    """Thin wrapper around cv2.VideoCapture with base64-JPEG helper."""

    def __init__(self, cap, info: CameraInfo) -> None:
        self._cap = cap
        self.info = info
        self._closed = False

    def grab_frame(self):
        """Grab one frame as a BGR numpy array. Returns None on failure."""
        if self._closed:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def grab_frame_b64(self, quality: int = DEFAULT_QUALITY) -> str | None:
        """Grab one frame and return it as a base64-encoded JPEG string.

        Returns None if the camera can't be read (don't crash the agent —
        the planner falls back to text-only when no frame is available).
        """
        frame = self.grab_frame()
        if frame is None:
            return None
        return _frame_to_base64_jpeg(frame, quality=quality)

    def close(self) -> None:
        if not self._closed:
            try:
                self._cap.release()
            except Exception:
                pass
            self._closed = True

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_camera(
    index: int = DEFAULT_INDEX,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> Camera:
    """Open the webcam at `index` and prime it with one warm-up read.

    Raises RuntimeError if the device can't be opened or doesn't return
    frames — the agent treats this as fatal at startup, but the loop itself
    handles per-frame failures gracefully via `grab_frame_b64() → None`.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required. Install with: pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {index}. "
            f"On macOS, check System Settings → Privacy → Camera and grant your terminal access. "
            f"If you have multiple webcams, try CW_CAMERA_INDEX=1 or 2."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Warm-up read — first frame is sometimes black / discarded by the driver.
    for _ in range(3):
        ok, _ = cap.read()
        if ok:
            break

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    info = CameraInfo(index=index, width=actual_w, height=actual_h, fps=actual_fps)

    cam = Camera(cap, info)

    # Final sanity check — we must be able to actually read a frame.
    if cam.grab_frame() is None:
        cam.close()
        raise RuntimeError(
            f"Opened camera {index} but couldn't read a frame. "
            f"Try a different CW_CAMERA_INDEX (e.g. 1) or check that no other app is using the camera."
        )

    return cam


def open_camera_from_env() -> Camera:
    """Open the camera using `CW_CAMERA_*` environment variables."""
    index = int(os.environ.get("CW_CAMERA_INDEX", DEFAULT_INDEX))
    width = int(os.environ.get("CW_CAMERA_WIDTH", DEFAULT_WIDTH))
    height = int(os.environ.get("CW_CAMERA_HEIGHT", DEFAULT_HEIGHT))
    fps = int(os.environ.get("CW_CAMERA_FPS", DEFAULT_FPS))
    return open_camera(index, width=width, height=height, fps=fps)


def _frame_to_base64_jpeg(frame, *, quality: int = DEFAULT_QUALITY) -> str:
    """Encode a BGR numpy frame as a base64-encoded JPEG string.

    BGR is what `cv2.VideoCapture` returns. Anthropic's vision API accepts
    base64 JPEG directly. Quality 80 keeps payloads ~30–60 KB at 720p.
    """
    import cv2

    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# Back-compat alias for any caller that imported the old name.
def frame_to_base64_jpeg(frame, *, quality: int = DEFAULT_QUALITY) -> str:
    return _frame_to_base64_jpeg(frame, quality=quality)
