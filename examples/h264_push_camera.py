"""
Push H.264 Camera — forward video a robot SDK already delivers encoded, without decoding it.

For robots that hand you H.264 from their own session (e.g. a quadruped's
WebRTC video): call ``push_h264()`` for every frame with its capture time. The
frames are muxed into the twin's WebRTC stream, and the capture time lands in
its video sync data. ``RobotVideo`` stands in for the robot SDK here. In a robot
driver, compose ``CameraStreamMixin`` with your ``BaseDriver``/``BaseROS2Driver``
instead: it starts the stream itself once the driver is active.

Requirements:
    pip install 'cyberwave[gstreamer]'   # plus gstreamer1.0-plugins-{base,good,bad,ugly}
    CYBERWAVE_API_KEY, CYBERWAVE_TWIN_UUID
"""

import threading
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from cyberwave.driver import BaseCameraDriver, CameraSourceConfig


class RobotVideo:
    """Stand-in for a robot SDK: emits encoded H.264 frames with their capture time."""

    def __init__(self, on_frame):
        Gst.init(None)
        self.on_frame, self.stop_event = on_frame, threading.Event()
        self.pipeline = Gst.parse_launch(
            "videotestsrc is-live=true ! video/x-raw,width=1280,height=720,framerate=15/1 "
            "! x264enc tune=zerolatency key-int-max=30 ! appsink name=out sync=false"
        )

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        sink = self.pipeline.get_by_name("out")
        while not self.stop_event.is_set():
            sample = sink.emit("try-pull-sample", Gst.SECOND)
            if sample:
                buf = sample.get_buffer()
                self.on_frame(buf.extract_dup(0, buf.get_size()), time.time())

    def stop(self):
        self.stop_event.set()
        self.pipeline.set_state(Gst.State.NULL)


class RobotCamera(BaseCameraDriver):
    CAMERA_SOURCE = CameraSourceConfig.push(camera_name="front_camera")

    async def on_connect_to_device(self):
        # Hand every encoded frame from the robot SDK to Cyberwave, with its capture time.
        self.robot = RobotVideo(on_frame=lambda frame, captured_at: self.push_h264(frame, capture_time=captured_at))
        self.robot.start()

    async def on_shutdown(self):
        self.robot.stop()
        await super().on_shutdown()


RobotCamera.main()
