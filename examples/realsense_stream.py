"""
RealSense Streaming — stream an Intel RealSense D455 to a twin.

Press Ctrl+C to stop.

Requirements:
    pip install cyberwave[realsense]
"""

from cyberwave import Cyberwave

cw = Cyberwave()
camera = cw.twin("intel/realsensed455")
camera.start_streaming()
