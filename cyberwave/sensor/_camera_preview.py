"""Camera preview subprocess entry point.

Spawned by :class:`~cyberwave.sensor.camera_sim.MujocoMultiCameraStreamer` as a
separate process so that the cv2 GUI runs isolated from the simulation's
GLFW/OpenGL context (Qt5 + GLFW in the same process causes fatal X11 errors on
some Linux drivers).

Protocol (stdin binary stream)
-------------------------------
Each frame is sent as a length-prefixed raw RGB chunk::

    4 bytes  big-endian uint32  height  (pixels)
    4 bytes  big-endian uint32  width   (pixels)
    4 bytes  big-endian uint32  nbytes  (= height * width * 3)
    nbytes   bytes              raw RGB data (row-major, 8-bit per channel)

EOF or any read error closes the window and exits.

Usage (invoked automatically by camera_sim.py)::

    python -m cyberwave.sensor._camera_preview <window_title>
"""

from __future__ import annotations

import struct
import sys

import cv2
import numpy as np


def main() -> None:
    """Entry point: open a cv2 window and display incoming frames."""
    title = sys.argv[1] if len(sys.argv) > 1 else "camera"
    buf = sys.stdin.buffer

    while True:
        hdr = buf.read(12)
        if len(hdr) < 12:
            break
        h, w, nbytes = struct.unpack(">III", hdr)
        data = b""
        while len(data) < nbytes:
            chunk = buf.read(nbytes - len(data))
            if not chunk:
                cv2.destroyAllWindows()
                return
            data += chunk
        frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        cv2.imshow(title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
