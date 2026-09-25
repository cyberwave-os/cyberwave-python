"""
Depth frame capture — grab a depth frame from a depth camera twin.

get_frame() returns a float32 H×W array of depth in metres by default.
Pass raw=True for the underlying uint16 millimetres (MQTT) / uint8 image (REST).

Requirements:
    pip install cyberwave numpy
"""

import numpy as np

from cyberwave import Cyberwave

cw = Cyberwave()
twin = cw.twin("intel/realsensed455")

depth = twin.camera["depth_camera"].get_frame()  # float32 metres, numpy
print(f"shape: {depth.shape}  dtype: {depth.dtype}")
print(f"depth range: {depth.min():.2f}–{depth.max():.2f} m")
np.save("depth_frame.npy", depth)  # use standard numpy methods
