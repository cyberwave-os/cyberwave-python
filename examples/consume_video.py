"""
Consume a live WebRTC video stream from a twin camera.

``get_video()`` attaches as an SFU consumer (the browser-equivalent): it
publishes nothing and has no effect on the producer or other viewers.

Requirements:
    pip install cyberwave[camera]   # aiortc, av, opencv-python
"""

from cyberwave import Cyberwave

cw = Cyberwave()
robot = cw.twin(twin_id="twin-uuid")

stream = robot.camera.get_video()

# Grab the latest decoded frame (numpy BGR array, or None before the first frame).
frame = stream.get_frame()

# Or open a live window (blocking; press q/Esc to quit).
stream.show()
