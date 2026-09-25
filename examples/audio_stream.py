"""
Audio Streaming — stream Go2 microphone to a twin over WebRTC.

Press Ctrl+C to stop.

Env vars:
    CYBERWAVE_API_KEY       API key
    CYBERWAVE_TWIN_UUID     Go2 twin UUID
    CYBERWAVE_GO2_IP_ADDR   Robot IP (default: 192.168.123.161)

Requirements:
    pip install cyberwave unitree_webrtc_connect
"""

import asyncio
import os

from cyberwave import Cyberwave
from cyberwave.sensor.microphone import MicrophoneAudioStreamer

cw = Cyberwave(
    api_key=os.environ["CYBERWAVE_API_KEY"],
    mqtt_host=os.environ.get("CYBERWAVE_MQTT_HOST", "localhost"),
    source_type="edge",
)
twin_uuid = os.environ["CYBERWAVE_TWIN_UUID"]
robot_ip = os.environ.get("CYBERWAVE_GO2_IP_ADDR", "192.168.123.161")


async def main():
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    await conn.connect()

    import queue

    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=16)

    async def on_audio(frame):
        arr = frame.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        s16 = (arr.astype("float32").clip(-1, 1) * 32767).astype("int16")
        try:
            audio_queue.put_nowait(s16.tobytes())
        except queue.Full:
            audio_queue.get_nowait()
            audio_queue.put_nowait(s16.tobytes())

    conn.audio.add_track_callback(on_audio)
    conn.audio.switchAudioChannel(True)

    cw.mqtt.connect()

    streamer = MicrophoneAudioStreamer(
        client=cw.mqtt,
        get_audio=lambda: (
            audio_queue.get(timeout=0.02) if not audio_queue.empty() else None
        ),
        twin_uuid=twin_uuid,
        mic_name="mic",
        auto_reconnect=True,
    )
    await streamer.start()
    print(f"Streaming audio to {twin_uuid} — press Ctrl+C to stop")

    try:
        await asyncio.sleep(float("inf"))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await streamer.stop()
        await conn.disconnect()
        cw.mqtt.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
