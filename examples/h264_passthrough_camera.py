"""
H.264 Passthrough Camera — stream a network camera to its twin without decoding or re-encoding.

The camera's own H.264 is muxed into the WebRTC stream, with video sync data
added to every frame. On an edge device the same needs no code: set the twin's
metadata ``video_device`` to the URL and ``h264_passthrough`` to true.

Requirements:
    pip install 'cyberwave[gstreamer]'   # plus gstreamer1.0-plugins-{base,good,bad}
    CYBERWAVE_API_KEY, CYBERWAVE_TWIN_UUID
    CAMERA_URL, one of:
        rtsp://user:pass@192.168.1.10:554/stream   IP camera / NVR
        rtp://0.0.0.0:5600                         RTP/H.264 pushed to this host over UDP
        udp://239.0.0.1:5000                       MPEG-TS over UDP (multicast or unicast)
        tcp://192.168.1.20:8000                    MPEG-TS over TCP
"""

import os

from cyberwave.driver import BaseCameraDriver, CameraSourceConfig


class PassthroughCamera(BaseCameraDriver):
    CAMERA_SOURCE = CameraSourceConfig.from_url(os.environ["CAMERA_URL"], camera_name="color_camera")


PassthroughCamera.main()
