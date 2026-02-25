"""
Depth Camera Streaming Example

Stream RealSense camera feed to a digital twin. Press Ctrl+C to stop.

Requirements:
    pip install cyberwave[realsense]
"""

import os
from cyberwave import Cyberwave

cw = Cyberwave(token=os.getenv("CYBERWAVE_API_KEY"))
camera = cw.twin("intel/realsensed455")
camera.start_streaming()
