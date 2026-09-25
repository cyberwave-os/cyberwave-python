"""
Multimedia Streaming — stream fake video + audio to a twin over WebRTC. No robot needed.

Env vars:
    CYBERWAVE_API_KEY       API key
    CYBERWAVE_TWIN_UUID     Twin UUID

Requirements:
    pip install cyberwave[camera]
"""

import asyncio
import math
import os
import struct

import numpy as np

from cyberwave import Cyberwave
from cyberwave.sensor.av_streamer import MultimediaStreamer
from cyberwave.sensor.camera_virtual import VirtualVideoTrack
from cyberwave.sensor.microphone import MicrophoneAudioTrack

_SAMPLES = 960


class FakeVideoSource:
    """Colour-cycling stripe pattern."""

    def __init__(self, width=960, height=540):
        self._w, self._h, self._offset = width, height, 0
        import colorsys

        n = (width * 2) // 80 + 1
        row = []
        for i in range(n):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 0.95)
            row.extend([int(r * 255), int(g * 255), int(b * 255)] * 80)
        arr = np.array(row, dtype=np.uint8).reshape(-1, 3)
        self._palette = np.tile(arr[np.newaxis, :, :], (height, 1, 1))

    def get_frame(self):
        off = self._offset % self._palette.shape[1]
        frame = np.roll(self._palette, -off, axis=1)[:, : self._w, :]
        self._offset += 4
        return np.ascontiguousarray(frame)


class SineAudioSource:
    """440 Hz sine wave."""

    def __init__(self):
        self._phase = 0.0
        self._step = 2 * math.pi * 440 / 48000

    def get_audio(self):
        out = []
        for _ in range(_SAMPLES):
            out.append(int(math.sin(self._phase) * 0.2 * 32767))
            self._phase = (self._phase + self._step) % (2 * math.pi)
        return struct.pack(f"<{_SAMPLES}h", *out)


async def main():
    twin_uuid = os.environ["CYBERWAVE_TWIN_UUID"]

    cw = Cyberwave(
        api_key=os.environ["CYBERWAVE_API_KEY"],
        mqtt_host=os.environ.get("CYBERWAVE_MQTT_HOST", "localhost"),
        source_type="edge",
    )
    cw.mqtt.connect()

    video = FakeVideoSource()
    audio = SineAudioSource()

    streamer = MultimediaStreamer(
        client=cw.mqtt,
        create_video_track=lambda: VirtualVideoTrack(
            video.get_frame, width=960, height=540, fps=15
        ),
        create_audio_track=lambda: MicrophoneAudioTrack(audio.get_audio),
        twin_uuid=twin_uuid,
        camera_name="rgb",
        mic_name="audio",
        auto_reconnect=True,
    )
    await streamer.start()
    print(f"Streaming to {twin_uuid} — press Ctrl+C to stop")

    try:
        await asyncio.sleep(float("inf"))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await streamer.stop()
        cw.mqtt.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
