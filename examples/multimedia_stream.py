"""
Multimedia Streaming Example (Video + Audio)

Stream a colour-cycling fake video frame and an Ode to Joy melody to a
Cyberwave digital twin over WebRTC. No physical robot required.

Requirements:
    pip install cyberwave[camera]

Usage:
    source /path/to/.env.local
    python multimedia_stream.py
    python multimedia_stream.py --video          # video only
    python multimedia_stream.py --audio          # audio only
    python multimedia_stream.py --video --audio  # both (same as no flags)

Required env vars (see .env.local):
    CYBERWAVE_API_KEY         Knox API key
    CYBERWAVE_BASE_URL        Backend URL         (default: http://localhost:8000)
    CYBERWAVE_MQTT_HOST       MQTT broker         (default: localhost)
    CYBERWAVE_TWIN_UUID       Digital twin UUID for video stream
    CYBERWAVE_AUDIO_TWIN_UUID Digital twin UUID for audio stream (defaults to CYBERWAVE_TWIN_UUID)
    CYBERWAVE_TURN_SERVERS    JSON TURN server list (default: [] = no TURN)
"""

import argparse
import asyncio
import logging
import math
import os
import struct

import numpy as np

from cyberwave import Cyberwave
from cyberwave.sensor.audio_microphone import MicrophoneAudioStreamer
from cyberwave.sensor.camera_virtual import VirtualCameraStreamer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────────
_AUDIO_SAMPLES = 960   # 20 ms at 48 kHz mono


# ── Fake video source ────────────────────────────────────────────────────────

class FakeVideoSource:
    """Generates an animated diagonal stripe pattern (no robot required).

    Vertical rainbow stripes scroll horizontally across the frame so it's easy
    to confirm the stream is both live and updating.
    """

    _STRIPE_WIDTH = 80  # pixels per colour stripe

    def __init__(self, width: int = 960, height: int = 540) -> None:
        self._w = width
        self._h = height
        self._offset = 0  # scroll offset in pixels

        # Pre-build a wide palette row (one full stripe cycle) and tile it
        import colorsys
        n_stripes = (width * 2) // self._STRIPE_WIDTH + 1
        palette = []
        for i in range(n_stripes):
            hue = i / n_stripes
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            palette.extend([int(r * 255), int(g * 255), int(b * 255)] * self._STRIPE_WIDTH)
        row = np.array(palette, dtype=np.uint8).reshape(-1, 3)
        # Tile vertically to full frame height
        self._palette = np.tile(row[np.newaxis, :, :], (height, 1, 1))

    def get_frame(self) -> np.ndarray:
        offset = self._offset % self._palette.shape[1]
        # Roll the palette row to produce the scroll effect
        frame = np.roll(self._palette, -offset, axis=1)[:, :self._w, :]
        self._offset = (self._offset + 4) % self._palette.shape[1]
        return np.ascontiguousarray(frame)


# ── Sine audio source ─────────────────────────────────────────────────────────

class SineAudioSource:
    """440 Hz sine wave at 20% amplitude."""

    _FREQ = 440
    _SAMPLE_RATE = 48000
    _AMPLITUDE = 0.20

    def __init__(self) -> None:
        self._phase = 0.0
        self._step = 2 * math.pi * self._FREQ / self._SAMPLE_RATE

    def get_audio(self) -> bytes:
        out = []
        for _ in range(_AUDIO_SAMPLES):
            out.append(int(math.sin(self._phase) * self._AMPLITUDE * 32767))
            self._phase = (self._phase + self._step) % (2 * math.pi)
        return struct.pack(f"<{_AUDIO_SAMPLES}h", *out)


class MelodyAudioSource:
    """Plays Ode to Joy using sine wave synthesis at 120 BPM.

    Notes have a short silence gap at the start for articulation.
    """

    _SAMPLE_RATE = 48000
    _AMPLITUDE = 0.25
    _TEMPO_BPS = 2.0          # beats per second (120 BPM)
    _GAP_SAMPLES = int(0.018 * 48000)  # 18 ms silence gap between notes

    # (frequency_hz, duration_in_beats)
    _NOTES = [
        (329.63, 1), (329.63, 1), (349.23, 1), (392.00, 1),
        (392.00, 1), (349.23, 1), (329.63, 1), (293.66, 1),
        (261.63, 1), (261.63, 1), (293.66, 1), (329.63, 1),
        (329.63, 1.5), (293.66, 0.5), (293.66, 2),
        (329.63, 1), (329.63, 1), (349.23, 1), (392.00, 1),
        (392.00, 1), (349.23, 1), (329.63, 1), (293.66, 1),
        (261.63, 1), (261.63, 1), (293.66, 1), (329.63, 1),
        (293.66, 1.5), (261.63, 0.5), (261.63, 2),
    ]

    def __init__(self) -> None:
        self._note_idx = 0
        self._samples_in_note = 0
        self._note_total = 0
        self._freq = 0.0
        self._phase = 0.0
        self._advance_note()

    def _advance_note(self) -> None:
        freq, beats = self._NOTES[self._note_idx]
        self._note_idx = (self._note_idx + 1) % len(self._NOTES)
        self._freq = freq
        self._note_total = int(beats / self._TEMPO_BPS * self._SAMPLE_RATE)
        self._samples_in_note = 0
        self._phase = 0.0

    def get_audio(self) -> bytes:
        out = []
        for _ in range(_AUDIO_SAMPLES):
            if self._samples_in_note >= self._note_total:
                self._advance_note()
            if self._samples_in_note < self._GAP_SAMPLES:
                sample = 0
            else:
                sample = int(math.sin(self._phase) * self._AMPLITUDE * 32767)
                self._phase = (self._phase + 2 * math.pi * self._freq / self._SAMPLE_RATE) % (2 * math.pi)
            self._samples_in_note += 1
            out.append(sample)
        return struct.pack(f"<{_AUDIO_SAMPLES}h", *out)


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream fake video and/or synthetic audio to a Cyberwave digital twin.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Stream video (default: on if neither --audio nor --video is given)",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Stream audio (default: on if neither --audio nor --video is given)",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    neither = not args.audio and not args.video
    stream_video = args.video or neither
    stream_audio = args.audio or neither

    twin_uuid = os.environ.get("CYBERWAVE_TWIN_UUID", "")
    audio_twin = os.environ.get("CYBERWAVE_AUDIO_TWIN_UUID", "") or twin_uuid

    if stream_video and not twin_uuid:
        raise SystemExit("Set CYBERWAVE_TWIN_UUID to the UUID of your digital twin.")
    if stream_audio and not audio_twin:
        raise SystemExit(
            "Set CYBERWAVE_AUDIO_TWIN_UUID or CYBERWAVE_TWIN_UUID for audio streaming."
        )

    cw = Cyberwave(
        api_key=os.environ["CYBERWAVE_API_KEY"],
        base_url=os.environ.get("CYBERWAVE_BASE_URL", "http://localhost:8000"),
        mqtt_host=os.environ.get("CYBERWAVE_MQTT_HOST", "localhost"),
        mqtt_port=int(os.environ.get("CYBERWAVE_MQTT_PORT", "1883")),
        mqtt_username=os.environ.get("CYBERWAVE_MQTT_USER", "test"),
        topic_prefix=os.environ.get("CYBERWAVE_ENVIRONMENT", "local"),
        source_type="edge",
    )

    get_frame = FakeVideoSource().get_frame
    get_audio = MelodyAudioSource().get_audio
    if stream_video and stream_audio:
        logger.info("Streaming: fake video (colour-cycling) + Ode to Joy melody")
    elif stream_video:
        logger.info("Streaming: fake video (colour-cycling)")
    else:
        logger.info("Streaming: Ode to Joy melody")

    cw.mqtt.connect()
    logger.info("MQTT connected")

    # ── Start streamers ────────────────────────────────────────────────────────
    cam_streamer: VirtualCameraStreamer | None = None
    if stream_video:
        cam_streamer = VirtualCameraStreamer(
            client=cw.mqtt,
            get_frame=get_frame,
            width=960,
            height=540,
            fps=15,
            twin_uuid=twin_uuid,
            camera_name="rgb",
            auto_reconnect=True,
        )

    audio_streamer: MicrophoneAudioStreamer | None = None
    if stream_audio:
        audio_streamer = MicrophoneAudioStreamer(
            client=cw.mqtt,
            get_audio=get_audio,
            twin_uuid=audio_twin,
            sensor_name="audio",
            auto_reconnect=True,
        )

    if stream_video and stream_audio:
        await cam_streamer.start()
        await audio_streamer.start()
    elif stream_video:
        await cam_streamer.start()
    else:
        await audio_streamer.start()

    if stream_video and stream_audio:
        logger.info(
            "Streaming video+audio to twin %s (audio twin: %s) — press Ctrl+C to stop",
            twin_uuid,
            audio_twin,
        )
    elif stream_video:
        logger.info("Streaming video to twin %s — press Ctrl+C to stop", twin_uuid)
    else:
        logger.info("Streaming audio to twin %s — press Ctrl+C to stop", audio_twin)

    try:
        await asyncio.sleep(float("inf"))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Stopping...")
        if stream_video and stream_audio:
            await asyncio.gather(
                cam_streamer.stop(),
                audio_streamer.stop(),
                return_exceptions=True,
            )
        elif stream_video:
            await cam_streamer.stop()
        else:
            await audio_streamer.stop()
        cw.mqtt.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    asyncio.run(main())
