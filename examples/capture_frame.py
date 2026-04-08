"""
Frame Capture & Manipulation Example

Grab frames from a twin's camera sensor and process them with OpenCV / PIL.

The source of the frame — real camera ("real-world") or simulated 3-D
render ("simulation") — is controlled by ``cw.affect()``.  When no
explicit ``source_type`` is passed to ``capture_frame``, the active
affect mode is used automatically.

Requirements:
    pip install cyberwave numpy opencv-python Pillow
"""

import cv2
import numpy as np
from cyberwave import Cyberwave

cw = Cyberwave()
robot = cw.twin("the-robot-studio/so101")

# ── Affect-based source selection ─────────────────────────────────────

cw.affect("simulation")
sim_frame = robot.capture_frame("numpy")   # rendered frame from the 3-D camera

cw.affect("real-world")
real_frame = robot.capture_frame("numpy")  # live frame from the real camera

# ── Single frame as numpy array (uses the current affect mode) ────────

frame = robot.capture_frame("numpy")  # BGR numpy array

# Draw a timestamp overlay
cv2.putText(
    frame,
    "Cyberwave live",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

cv2.imwrite("annotated_frame.jpg", frame)
print("Saved annotated_frame.jpg")

# ── Edge detection ────────────────────────────────────────────────────

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite("edges.jpg", edges)
print("Saved edges.jpg")

# ── Batch capture → side-by-side composite ────────────────────────────

frames = robot.capture_frames(3, interval_ms=500, format="numpy")
composite = np.hstack(frames)
cv2.imwrite("composite.jpg", composite)
print(f"Saved composite.jpg ({len(frames)} frames stitched)")

# ── Using the twin.camera namespace ───────────────────────────────────

frame2 = robot.camera.read()  # numpy by default, like cv2.VideoCapture
path = robot.camera.snapshot()  # temp JPEG file
print(f"Snapshot saved to {path}")

# ── PIL example: resize + thumbnail ───────────────────────────────────

pil_frame = robot.capture_frame("pil")
pil_frame.thumbnail((320, 240))
pil_frame.save("thumbnail.jpg")
print("Saved thumbnail.jpg (320×240)")

cw.disconnect()
