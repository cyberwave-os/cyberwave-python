"""
Go2 Microphone Audio Streaming Example

Stream the Unitree Go2 robot microphone to a Cyberwave digital twin over WebRTC.
Press Ctrl+C to stop.

Requirements:
    pip install cyberwave unitree_webrtc_connect

Usage:
    source /path/to/.env.local
    python audio_stream_go2.py                  # robot mic (auto-fallback to sine)
    python audio_stream_go2.py --source sine    # 440 Hz test tone, no robot needed
    python audio_stream_go2.py --source robot   # robot mic only (error if unreachable)

Required env vars (see .env.local):
    CYBERWAVE_API_KEY         Knox API key
    CYBERWAVE_BASE_URL        Backend URL  (default: http://localhost:8000)
    CYBERWAVE_MQTT_HOST       MQTT broker  (default: localhost)
    CYBERWAVE_TWIN_UUID       Go2 digital twin UUID
    CYBERWAVE_GO2_IP_ADDR     Robot IP     (default: 192.168.123.161)
    CYBERWAVE_TURN_SERVERS    JSON TURN server list (default: [] = no TURN)
"""

import argparse
import asyncio
import logging
import math
import os
import struct

from cyberwave import Cyberwave
from cyberwave.sensor.audio_microphone import MicrophoneAudioStreamer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_SAMPLES = 960   # 20 ms at 48 kHz mono
_BYTES   = _SAMPLES * 2  # s16


class Go2AudioBridge:
    """Receives av.AudioFrame from unitree_webrtc_connect, exposes get() for MicrophoneAudioStreamer.

    For the full implementation (with resampling, stereo downmix, and queue
    eviction) see cyberwave-edge-runtime/.../go2/src/go2_audio_bridge.py.
    """

    def __init__(self) -> None:
        import queue
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=16)
        self._buf = bytearray()

    async def on_audio_frame(self, frame) -> None:
        import numpy as np
        try:
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            s16 = (arr.astype(np.float32).clip(-1.0, 1.0) * 32767).astype("int16")
            self._buf.extend(s16.tobytes())
            while len(self._buf) >= _BYTES:
                chunk = bytes(self._buf[:_BYTES])
                del self._buf[:_BYTES]
                try:
                    self._queue.put_nowait(chunk)
                except Exception:
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                    self._queue.put_nowait(chunk)
        except Exception as e:
            logger.debug("Audio frame skipped: %s", e)

    def get(self, timeout: float = 0.02) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    def clear(self) -> None:
        self._buf.clear()
        while True:
            try:
                self._queue.get_nowait()
            except Exception:
                break


class SineAudioSource:
    """440 Hz sine wave at 20% amplitude — used when the robot is not connected."""

    _FREQ = 440
    _SAMPLE_RATE = 48000
    _AMPLITUDE = 0.20  # 20% — audible but not annoying

    def __init__(self) -> None:
        self._phase = 0.0
        self._step = 2 * math.pi * self._FREQ / self._SAMPLE_RATE

    def get_audio(self) -> bytes:
        out = []
        for _ in range(_SAMPLES):
            out.append(int(math.sin(self._phase) * self._AMPLITUDE * 32767))
            self._phase = (self._phase + self._step) % (2 * math.pi)
        return struct.pack(f"<{_SAMPLES}h", *out)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Go2 mic (or a sine wave) to Cyberwave.")
    parser.add_argument(
        "--source",
        choices=["robot", "sine"],
        default="robot",
        help="Audio source: 'robot' = Go2 mic with sine fallback (default), 'sine' = 440 Hz test tone",
    )
    args = parser.parse_args()

    robot_ip = os.environ.get("CYBERWAVE_GO2_IP_ADDR", "192.168.123.161")
    twin_uuid = os.environ.get("CYBERWAVE_TWIN_UUID", "")
    if not twin_uuid:
        raise SystemExit("Set CYBERWAVE_TWIN_UUID to the UUID of your Go2 twin in Cyberwave.")

    # Connect to Cyberwave
    cw = Cyberwave(
        api_key=os.environ["CYBERWAVE_API_KEY"],
        base_url=os.environ.get("CYBERWAVE_BASE_URL", "http://localhost:8000"),
        mqtt_host=os.environ.get("CYBERWAVE_MQTT_HOST", "localhost"),
        mqtt_port=int(os.environ.get("CYBERWAVE_MQTT_PORT", "1883")),
        mqtt_username=os.environ.get("CYBERWAVE_MQTT_USER", "test"),
        topic_prefix=os.environ.get("CYBERWAVE_ENVIRONMENT", "local"),
        source_type="edge",
    )

    # Resolve audio source
    bridge: Go2AudioBridge | None = None
    conn = None

    if args.source == "sine":
        logger.info("Audio source: 440 Hz sine wave (test tone)")
        get_audio = SineAudioSource().get_audio
    else:
        # Try to connect to the Go2 robot; fall back to sine wave if unreachable
        try:
            from unitree_webrtc_connect.webrtc_driver import (
                UnitreeWebRTCConnection,
                WebRTCConnectionMethod,
            )
            logger.info("Connecting to Go2 at %s ...", robot_ip)
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
            await conn.connect()
            conn.pc.on("error", lambda e: None)  # suppress pyee disconnect noise
            bridge = Go2AudioBridge()
            conn.audio.add_track_callback(bridge.on_audio_frame)
            conn.audio.switchAudioChannel(True)
            logger.info("Connected to Go2 — streaming robot mic")
            get_audio = bridge.get
        except Exception as exc:
            if args.source == "robot":
                raise SystemExit(f"Could not connect to Go2: {exc}") from exc
            logger.warning("Could not connect to Go2 (%s) — falling back to sine wave", exc)
            conn = None
            get_audio = SineAudioSource().get_audio

    # Connect MQTT (needed for WebRTC signaling via webrtc-offer / webrtc-answer)
    cw.mqtt.connect()
    logger.info("MQTT connected")

    # Stream microphone audio to Cyberwave over WebRTC
    # TURN servers: default to none for local/LAN setups (direct ICE works without TURN).
    # Set CYBERWAVE_TURN_SERVERS to a JSON array to override, e.g.:
    #   export CYBERWAVE_TURN_SERVERS='[{"urls":"turn:192.168.10.101:3478","username":"u","credential":"p"}]'
    import json as _json
    _turn_env = os.environ.get("CYBERWAVE_TURN_SERVERS", "")
    turn_servers: list = _json.loads(_turn_env) if _turn_env.strip() else []

    streamer = MicrophoneAudioStreamer(
        client=cw.mqtt,
        get_audio=get_audio,
        twin_uuid=twin_uuid,
        sensor_name="mic",
        auto_reconnect=True,
        turn_servers=turn_servers,
    )
    await streamer.start()
    logger.info("Streaming to twin %s — press Ctrl+C to stop", twin_uuid)

    try:
        await asyncio.sleep(float("inf"))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Stopping...")
        await streamer.stop()
        if bridge is not None:
            bridge.clear()
        if conn is not None:
            await conn.disconnect()
        cw.mqtt.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    asyncio.run(main())
