"""
Frame capture — grab frames from a twin camera via ``twin.get_frame()``.

Sources:
  - cloud (default) — platform REST latest-frame
  - local — streamer cache or USB device
  - zenoh — cw.data frames channel
  - remote_edge — MQTT take_photo on the edge driver

Requirements:
    pip install cyberwave numpy opencv-python Pillow
"""

import cv2

from cyberwave import Cyberwave

cw = Cyberwave()
robot = cw.twin("the-robot-studio/so101")

# Cloud (default) — first imaging sensor when sensor_id omitted
# frame = robot.get_frame("numpy")

# Local streamer / USB
frame = robot.get_frame("numpy", source="local")
cv2.imwrite("frame.jpg", frame)

# Or save to disk (same transport kwargs as get_frame)
# path = robot.get_frame("path", path="frame.jpg", source="local")
# folder = robot.get_frames(5, interval_ms=200)

# Edge driver photo command
# edge_jpeg = robot.get_frame("bytes", source="remote_edge")

# Zenoh (requires cyberwave[zenoh] + publisher)
# frame = robot.get_frame("numpy", source="zenoh")

cw.disconnect()
