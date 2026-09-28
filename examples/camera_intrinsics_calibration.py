"""Estimate real camera intrinsics (fx, fy, cx, cy, distortion) from a ChArUco board.

Hand-eye calibration takes intrinsics as an input and cannot recover from bad
ones: every solvePnP board pose inherits the error, and it surfaces only as an
inflated hand-eye residual with no indication of the cause. handeye_calibration.py
deliberately has no FOV-derived fallback for exactly that reason: deriving fx/fy
from a twin's declared field of view is fine for drawing an FOV cone, not for
metric work. Pass intrinsics=None there and the session solves its own.

Run this once per camera *per resolution* (intrinsics are resolution specific),
then export the printed FX/FY/CX/CY before running the hand-eye calibration.

Usage:
    python examples/camera_intrinsics_calibration.py

Hold the board at varied angles and distances -- tilt it, don't just slide it.
Views that are all fronto-parallel leave the focal length poorly constrained.
"""

import os
import time

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit('OpenCV is required: pip install "cyberwave[camera]"') from exc

from cyberwave import Cyberwave

CAMERA_TWIN_UUID = os.environ.get(
    "CAMERA_TWIN_UUID", "ee531f91-7fc0-4ee6-89b6-3547177e1c2c"
)

# Keep in sync with BOARD in handeye_calibration.py -- a mismatch here produces
# confidently wrong intrinsics rather than an error.
SQUARES_X = int(os.environ.get("SQUARES_X", "11"))
SQUARES_Y = int(os.environ.get("SQUARES_Y", "8"))
SQUARE_SIZE_M = float(os.environ.get("SQUARE_SIZE_M", "0.015"))
MARKER_SIZE_M = float(os.environ.get("MARKER_SIZE_M", "0.011"))
DICTIONARY = os.environ.get("DICTIONARY", "DICT_4X4_50")

VIEWS = int(os.environ.get("VIEWS", "15"))
# calibrateCamera tolerates few points per view, but sparse views make the
# distortion terms unstable; require a solid chunk of the board instead.
MIN_CORNERS = int(os.environ.get("MIN_CORNERS", "8"))
FRAME_RETRIES = 5
FRAME_RETRY_DELAY_S = 1.0


def _grab_frame(camera):
    """One frame from the platform's latest-frame cache, retried through brief gaps."""
    for attempt in range(1, FRAME_RETRIES + 1):
        frame = camera.camera.get_frame(format="numpy", source="cloud")
        if frame is not None:
            return frame
        if attempt < FRAME_RETRIES:
            print(f"  no frame yet (attempt {attempt}/{FRAME_RETRIES}) — retrying...")
            time.sleep(FRAME_RETRY_DELAY_S)
    raise RuntimeError("No frame available from the camera twin.")


cw = Cyberwave()
camera = cw.twins.get(CAMERA_TWIN_UUID)

dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M, dictionary
)
detector = cv2.aruco.CharucoDetector(board)

print(f"Board: {SQUARES_X}x{SQUARES_Y}, square={SQUARE_SIZE_M * 1000:.1f}mm, "
      f"marker={MARKER_SIZE_M * 1000:.1f}mm, dict={DICTIONARY}")
print("Measure the printed squares with calipers -- board size is the only")
print("metric input here, and a 2% error becomes a 2% error in every distance.\n")
print("Vary angle AND distance between views; tilt the board, don't just slide it.\n")

object_points: list[np.ndarray] = []
image_points: list[np.ndarray] = []
image_size = None

while len(object_points) < VIEWS:
    input(f"[{len(object_points) + 1}/{VIEWS}] Position the board, then press Enter...")
    frame = _grab_frame(camera)

    if image_size is None:
        image_size = (frame.shape[1], frame.shape[0])
    elif image_size != (frame.shape[1], frame.shape[0]):
        # Intrinsics are per-resolution; silently mixing sizes would be garbage.
        raise RuntimeError(
            f"Frame size changed mid-run ({image_size} -> "
            f"{(frame.shape[1], frame.shape[0])}). Restart with a stable stream."
        )

    corners, ids, _, _ = detector.detectBoard(frame)
    if ids is None or len(ids) < MIN_CORNERS:
        print(f"  only {0 if ids is None else len(ids)} corners (need {MIN_CORNERS}) — "
              "get more of the board in frame, or reduce glare")
        continue

    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < MIN_CORNERS:
        print("  corner match failed — retry")
        continue

    object_points.append(obj_pts)
    image_points.append(img_pts)
    print(f"  captured {len(ids)} corners ({len(object_points)}/{VIEWS} views)")

print("\nCalibrating...")
rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None
)

fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

print(f"\nRMS reprojection error: {rms:.3f} px  (< 1.0 is good, > 2.0 means redo it)")
print(f"resolution: {image_size[0]}x{image_size[1]}")
print(f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
print(f"distortion: {np.array2string(dist_coeffs.ravel(), precision=5)}")

# cx/cy landing far from the image centre usually means a bad board size or too
# few varied views, not a genuinely offset sensor.
off_centre = abs(cx - image_size[0] / 2) / image_size[0]
if off_centre > 0.15 or abs(cy - image_size[1] / 2) / image_size[1] > 0.15:
    print("\nWARNING: principal point is far from the image centre — suspect the "
          "board dimensions or too few distinct viewpoints.")

print("\nExport these before running the hand-eye calibration:\n")
print(f"export FX={fx:.4f} FY={fy:.4f} CX={cx:.4f} CY={cy:.4f}")
