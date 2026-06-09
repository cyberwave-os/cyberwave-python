"""
Camera Streaming — stream a USB camera to a digital twin. Press Ctrl+C to stop.

Requirements:
    pip install cyberwave[camera]
"""

from cyberwave import Cyberwave

cw = Cyberwave()
camera = cw.twin("cyberwave/standard-cam")
camera.start_streaming()
# After streaming, grab a local frame:
# frame = camera.get_frame("numpy", source="local")
