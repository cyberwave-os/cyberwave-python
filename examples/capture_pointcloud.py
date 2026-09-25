"""
Point cloud capture — grab XYZ points from a depth camera twin.

Returns an (N, 3) float32 array of XYZ coordinates in metres (camera optical frame).

Requirements:
    pip install cyberwave numpy
"""

import numpy as np

from cyberwave import Cyberwave

cw = Cyberwave()
twin = cw.twin("intel/realsensed455")

pc = twin.camera["depth_camera"].get_pointcloud()
print(f"depth pointcloud  shape={pc.shape}  z range=[{pc[:,2].min():.2f}, {pc[:,2].max():.2f}] m")
np.save("pointcloud_depth.npy", pc)

cw.disconnect()
