"""
Camera Streaming Example

Stream camera feed to a digital twin. Press Ctrl+C to stop.

Requirements:
    pip install cyberwave[camera]
"""

import os
from cyberwave import Cyberwave

cw = Cyberwave(api_key=os.getenv("CYBERWAVE_API_KEY"))
camera = cw.twin("cyberwave/standard-cam")
camera.start_streaming()
