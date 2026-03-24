"""Microphone audio streaming (e.g. Go2 robot mic).

Mirrors the camera_<source> pattern: use MicrophoneAudioTrack / MicrophoneAudioStreamer
for audio from a microphone via a get_audio() callback (e.g. PyAudio, sounddevice).
Also defines BaseAudioTrack and BaseAudioStreamer for custom audio sources.

Example:
    >>> def get_audio():
    ...     # Return 20ms of s16 mono 48kHz (960 samples = 1920 bytes), or None for silence
    ...     return bytes(1920)  # or read from device
    >>>
    >>> streamer = MicrophoneAudioStreamer(
    ...     client.mqtt, get_audio, twin_uuid="twin-id", sensor_name="mic"
    ... )
    >>> await streamer.start()
"""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from av import AudioFrame

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient

logger = logging.getLogger(__name__)


def _strip_non_opus_audio(sdp: str) -> str:
    """Remove PCMU/PCMA codecs from a WebRTC SDP offer, keeping only Opus.

    aiortc includes static codecs PCMU (PT=0) and PCMA (PT=8) in every audio
    offer. Mediasoup only registers Opus, so if PCMU/PCMA are listed first in
    the SDP the SFU picks them and fails with an "Unsupported codec" error.
    This munging step strips those codecs before signaling.
    """
    lines = sdp.splitlines()

    # Static RTP PTs that are always PCMU (0) and PCMA (8)
    non_opus_pts: set[str] = {"0", "8"}

    # Also discover any dynamically-assigned PCMU/PCMA PTs via rtpmap
    for line in lines:
        if line.startswith("a=rtpmap:"):
            rest = line[len("a=rtpmap:"):]
            pt, _, codec_clock = rest.partition(" ")
            codec = codec_clock.split("/")[0].upper()
            if codec in ("PCMU", "PCMA"):
                non_opus_pts.add(pt)

    result: list[str] = []
    for line in lines:
        # Drop per-codec attribute lines for non-Opus codecs
        if any(
            line.startswith(f"a=rtpmap:{pt} ")
            or line.startswith(f"a=fmtp:{pt} ")
            or line.startswith(f"a=rtcp-fb:{pt} ")
            for pt in non_opus_pts
        ):
            continue
        # Strip non-Opus PTs from the m=audio payload list
        if line.startswith("m=audio"):
            parts = line.split()
            # m=audio <port> <proto> PT1 PT2 ...
            if len(parts) >= 4:
                filtered = [pt for pt in parts[3:] if pt not in non_opus_pts]
                if filtered:
                    line = " ".join(parts[:3] + filtered)
        result.append(line)

    sep = "\r\n" if "\r\n" in sdp else "\n"
    return sep.join(result)


# 20ms packetization for Opus (standard for WebRTC)
AUDIO_PTIME = 0.020
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_LAYOUT = "mono"

# Reused from sensor package to avoid circular import
_AUDIO_TURN_SERVERS = [
    {"urls": ["stun:turn.cyberwave.com:3478"]},
    {
        "urls": "turn:turn.cyberwave.com:3478",
        "username": "cyberwave-user",
        "credential": "cyberwave-admin",
    },
]
_CONNECTION_LOSS_CONFIRMATION_CHECKS = 3


def _notify(callback: Callable[..., None] | None, *args: Any) -> None:
    """Invoke callback(*args) if provided."""
    if callback is not None:
        callback(*args)


# =============================================================================
# Base audio track
# =============================================================================


class BaseAudioTrack(AudioStreamTrack):
    """Abstract base class for audio stream tracks.

    Subclasses must implement:
        - recv: Return the next AudioFrame (e.g. s16, 48kHz, 20ms)
        - close: Release resources
    """

    def __init__(self) -> None:
        super().__init__()
        self._closed = False

    def get_stream_attributes(self) -> dict[str, Any]:
        """Stream attributes included in the WebRTC offer payload."""
        return {}

    def close(self) -> None:
        """Release audio resources. Override in subclasses."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


# =============================================================================
# Microphone audio track
# =============================================================================


class MicrophoneAudioTrack(BaseAudioTrack):
    """Audio track that pulls frames from a microphone callback.

    The callback should return exactly 20ms of audio as bytes (s16, mono),
    or None to send silence. Sample rate is fixed at 48kHz for WebRTC/Opus.

    Args:
        get_audio: Callable[[], bytes | None]. Return 960 * 2 = 1920 bytes
            (s16 mono 48kHz 20ms), or None for silence.
        sample_rate: Must be 48000 for Opus (default).
        layout: "mono" or "stereo" (stereo = 3840 bytes per 20ms).
    """

    def __init__(
        self,
        get_audio: Callable[[], bytes | None],
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        layout: str = DEFAULT_LAYOUT,
    ) -> None:
        super().__init__()
        self.get_audio = get_audio
        self.sample_rate = sample_rate
        self.layout = layout
        self._samples_per_frame = int(AUDIO_PTIME * sample_rate)
        self._bytes_per_frame = self._samples_per_frame * 2 * (2 if layout == "stereo" else 1)
        self._start: float | None = None
        self._pts: int = 0

    def get_stream_attributes(self) -> dict[str, Any]:
        return {
            "audio_type": "microphone",
            "sample_rate": self.sample_rate,
            "layout": self.layout,
            "ptime_ms": int(AUDIO_PTIME * 1000),
        }

    async def recv(self) -> AudioFrame:
        if self._closed:
            raise MediaStreamError("Track is closed")

        now = time.monotonic()
        if self._start is None:
            self._start = now

        # Throttle to 20ms per frame
        next_pts_time = self._start + (self._pts / self.sample_rate)
        wait = next_pts_time - now
        if wait > 0:
            await asyncio.sleep(wait)

        raw = await asyncio.to_thread(self.get_audio)
        if raw is None:
            raw = bytes(self._bytes_per_frame)
        elif len(raw) < self._bytes_per_frame:
            raw = raw + bytes(self._bytes_per_frame - len(raw))

        frame = AudioFrame(format="s16", layout=self.layout, samples=self._samples_per_frame)
        frame.pts = self._pts
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        frame.planes[0].update(raw[: self._bytes_per_frame])
        self._pts += self._samples_per_frame

        return frame


# =============================================================================
# Base audio streamer (WebRTC + MQTT signaling)
# =============================================================================


class BaseAudioStreamer:
    """Abstract base class for WebRTC audio streaming to Cyberwave.

    Manages peer connection, MQTT signaling (webrtc-offer / webrtc-answer),
    and optional reconnection. Subclasses must implement initialize_track().
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        turn_servers: list | None = None,
        twin_uuid: str | None = None,
        auto_reconnect: bool = True,
        sensor_name: Optional[str] = None,
    ) -> None:
        self.client = client
        self.twin_uuid: str | None = twin_uuid
        self.sensor_name: Optional[str] = sensor_name  # e.g. "mic", "audio" (for multi-stream routing)
        self.auto_reconnect = auto_reconnect
        self.turn_servers = turn_servers if turn_servers is not None else _AUDIO_TURN_SERVERS

        self.pc: RTCPeerConnection | None = None
        self.streamer: BaseAudioTrack | None = None

        self._answer_received = False
        self._answer_data: dict[str, Any] | None = None
        self._should_reconnect = False
        self._is_running = False
        self._monitor_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._bad_connection_checks = 0
        self._session_id: int = 0  # incremented each _start_webrtc
        self._current_offer_session: int = 0  # session ID of the pending offer; gates stale answer callbacks
        self._subscribed_to_answer: bool = False  # answer topic is subscribed only once

        # Owned background task for run_with_auto_reconnect (created by run())
        self._run_task: asyncio.Task | None = None
        self._run_stop_event: asyncio.Event | None = None

    # -------------------------------------------------------------------------
    # Abstract
    # -------------------------------------------------------------------------

    def initialize_track(self) -> BaseAudioTrack:
        """Create and return the audio track. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement initialize_track()")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._answer_received = False
        self._answer_data = None
        self._bad_connection_checks = 0
        self._session_id += 1  # invalidate any in-flight on_answer callbacks

    async def _start_webrtc(self, twin_uuid: str | None = None) -> None:
        """Set up the WebRTC peer connection and perform MQTT signaling."""
        self._reset_state()
        if twin_uuid is not None:
            self.twin_uuid = twin_uuid
        elif self.twin_uuid is None:
            raise ValueError("twin_uuid must be set at init or when calling start()")

        logger.info("Starting WebRTC audio stream for twin %s", self.twin_uuid)
        self._current_offer_session = self._session_id
        await asyncio.sleep(0.1)
        await self._setup_webrtc()
        try:
            await self._perform_signaling()
        except Exception:
            if self.pc is not None:
                try:
                    await self.pc.close()
                    await asyncio.sleep(0.5)
                except Exception:
                    pass
                self.pc = None
            if self.streamer is not None:
                try:
                    self.streamer.close()
                except Exception:
                    pass
                self.streamer = None
            raise
        logger.debug("WebRTC audio connection established")

    async def start(self) -> None:
        """Start streaming with auto-reconnect in a background task.

        The task is owned by this streamer instance. Call stop() to signal
        it, await its completion, and release all WebRTC resources.
        Analogous to VirtualCameraStreamer.start().
        """
        self._run_stop_event = asyncio.Event()
        self._run_task = asyncio.create_task(
            self.run_with_auto_reconnect(stop_event=self._run_stop_event)
        )

    async def stop(self) -> None:
        """Stop streaming and release resources.

        Signals the background task started by start(), awaits its
        completion, then closes the WebRTC peer connection.
        """
        # Tear down the owned background task first
        if self._run_stop_event is not None:
            self._run_stop_event.set()
        if self._run_task is not None:
            try:
                await asyncio.wait_for(self._run_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                if not self._run_task.done():
                    self._run_task.cancel()
                    try:
                        await self._run_task
                    except asyncio.CancelledError:
                        pass
            self._run_task = None
            self._run_stop_event = None

        # Close the WebRTC peer connection and audio track
        if self.pc:
            try:
                await self.pc.close()
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error("Error closing peer connection: %s", e)
            finally:
                self.pc = None
        if self.streamer:
            try:
                self.streamer.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error("Error closing audio track: %s", e)
            finally:
                self.streamer = None
        self._reset_state()
        logger.info("Audio streaming stopped")

    async def run_with_auto_reconnect(
        self,
        stop_event: asyncio.Event | None = None,
        command_callback: Callable[..., None] | None = None,
    ) -> None:
        """Run audio streaming with auto-reconnect and optional start_audio/stop_audio commands."""
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before running")

        self._is_running = True
        self._event_loop = asyncio.get_running_loop()
        stop = stop_event or asyncio.Event()
        self._subscribe_to_commands(command_callback)
        if not self._subscribed_to_answer:
            self._subscribe_to_answer()
            self._subscribed_to_answer = True

        if self.pc is None:
            try:
                _notify(command_callback, "connecting", "Starting audio stream")
                await self._start_webrtc()
                self._should_reconnect = self.auto_reconnect
                _notify(command_callback, "ok", "Audio streaming started")
            except Exception as e:
                logger.error("Auto-start audio stream failed: %s", e, exc_info=True)
                _notify(command_callback, "error", str(e))

        if self.auto_reconnect:
            self._monitor_task = asyncio.create_task(self._monitor_connection(stop))

        _next_retry_at = time.monotonic() + 15.0
        _retry_backoff = 30.0
        _initial_connected = False

        try:
            while not stop.is_set() and self._is_running:
                if (
                    self.pc is None
                    and self.auto_reconnect
                    and not _initial_connected
                ):
                    if time.monotonic() >= _next_retry_at:
                        try:
                            logger.info("No active audio stream — retrying offer...")
                            await self._start_webrtc()
                            self._should_reconnect = self.auto_reconnect
                            _initial_connected = True
                        except Exception as exc:
                            logger.info(
                                "Audio stream retry failed (%s). Will retry in %.0fs.",
                                exc,
                                _retry_backoff,
                            )
                            _next_retry_at = time.monotonic() + _retry_backoff
                            _retry_backoff = min(_retry_backoff * 2, 120.0)
                await asyncio.sleep(0.5)
        finally:
            await self._cleanup_run()

    # -------------------------------------------------------------------------
    # WebRTC setup
    # -------------------------------------------------------------------------

    async def _setup_webrtc(self) -> None:
        self.streamer = self.initialize_track()
        ice_servers = [RTCIceServer(**s) for s in self.turn_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        self._setup_pc_handlers()
        self.pc.addTrack(self.streamer)

    def _setup_pc_handlers(self) -> None:
        @self.pc.on("connectionstatechange")
        def _on_connectionstatechange():
            logger.info("WebRTC connection state: %s", self.pc.connectionState)

        @self.pc.on("iceconnectionstatechange")
        def _on_iceconnectionstatechange():
            logger.info("WebRTC ICE connection state: %s", self.pc.iceConnectionState)

    # -------------------------------------------------------------------------
    # Signaling
    # -------------------------------------------------------------------------

    async def _perform_signaling(self) -> None:
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        _ice_deadline = time.monotonic() + 30.0
        while self.pc.iceGatheringState != "complete":
            if time.monotonic() > _ice_deadline:
                raise TimeoutError("ICE gathering timed out after 30s")
            await asyncio.sleep(0.1)
        sdp = self.pc.localDescription.sdp
        sdp = _strip_non_opus_audio(sdp)
        self._send_offer(sdp)
        await self._wait_for_answer()

    def _send_offer(self, sdp: str) -> None:
        prefix = self.client.topic_prefix
        offer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-offer"
        stream_attributes = {}
        if self.streamer:
            stream_attributes = self.streamer.get_stream_attributes()
        offer_payload = {
            "target": "backend",
            "sender": "edge",
            "type": self.pc.localDescription.type,
            "sdp": sdp,
            "timestamp": time.time(),
            "recording": False,
            "stream_attributes": stream_attributes,
            "sensor": self.sensor_name,
            "track_id": self.streamer.id if self.streamer else None,
            "frontend_type": "audio",
            "session_id": f"{self.client.client_id}_{self.sensor_name}",
        }
        self._publish_message(offer_topic, offer_payload)

    async def _wait_for_answer(self, timeout: float = 60.0) -> None:
        start = time.monotonic()
        while not self._answer_received:
            if time.monotonic() - start > timeout:
                raise TimeoutError("Timeout waiting for WebRTC answer")
            await asyncio.sleep(0.1)
        if self._answer_data is None:
            raise RuntimeError("Answer received but answer data is None")
        answer = (
            json.loads(self._answer_data)
            if isinstance(self._answer_data, str)
            else self._answer_data
        )
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    def _subscribe_to_answer(self) -> None:
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before subscribing")
        prefix = self.client.topic_prefix
        answer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-answer"
        logger.info("Subscribing to WebRTC answer topic: %s", answer_topic)

        def on_answer(data):
            try:
                if self._session_id != self._current_offer_session:
                    logger.debug(
                        "Discarding stale WebRTC answer (current session %d, offer session %d)",
                        self._session_id, self._current_offer_session,
                    )
                    return
                payload = data if isinstance(data, dict) else json.loads(data)
                if payload.get("type") == "offer":
                    return
                if payload.get("type") == "answer" and payload.get("target") == "edge":
                    # Reject answers whose SDP doesn't contain an audio track —
                    # this is a backwards-compatible way to distinguish video
                    # answers from audio answers without relying on the sensor field.
                    if "m=audio" not in payload.get("sdp", ""):
                        logger.debug("Ignoring answer with no m=audio (likely video stream)")
                        return
                    answer_sensor = payload.get("sensor") or payload.get("camera")
                    expected = self.sensor_name if self.sensor_name is not None else "default"
                    if answer_sensor is None or answer_sensor == expected:
                        self._answer_data = payload
                        self._answer_received = True
                elif payload.get("type") == "candidate" and payload.get("target") == "edge":
                    self._handle_candidate(payload)
            except Exception as e:
                logger.error("Error in on_answer: %s", e)

        self.client.subscribe(answer_topic, on_answer)
        candidate_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-candidate"
        self.client.subscribe(candidate_topic, on_answer)

    def _handle_candidate(self, payload: dict[str, Any]) -> None:
        if not self.pc or not payload.get("candidate") or not self._event_loop:
            return
        try:
            from aiortc import RTCIceCandidate
            c = payload["candidate"]
            candidate = RTCIceCandidate(
                candidate=c["candidate"],
                sdpMid=c.get("sdpMid"),
                sdpMLineIndex=c.get("sdpMLineIndex"),
            )
            asyncio.run_coroutine_threadsafe(
                self.pc.addIceCandidate(candidate), self._event_loop
            )
        except Exception as e:
            logger.warning("Failed to add ICE candidate: %s", e)

    def _subscribe_to_commands(self, command_callback: Callable[..., None] | None = None) -> None:
        prefix = self.client.topic_prefix
        command_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/command"
        logger.info("Subscribing to command topic: %s", command_topic)

        def on_command(data):
            try:
                payload = data if isinstance(data, dict) else json.loads(data)
                if "status" in payload:
                    return
                cmd = payload.get("command")
                if cmd == "start_audio":
                    asyncio.run_coroutine_threadsafe(
                        self._handle_start_command(command_callback),
                        self._event_loop,
                    )
                elif cmd == "stop_audio":
                    asyncio.run_coroutine_threadsafe(
                        self._handle_stop_command(command_callback),
                        self._event_loop,
                    )
            except Exception as e:
                logger.error("Error processing command: %s", e, exc_info=True)

        self.client.subscribe(command_topic, on_command)

    async def _handle_start_command(self, callback: Callable[..., None] | None = None) -> None:
        try:
            if self.pc is not None:
                _notify(callback, "ok", "Audio stream already running")
                return
            await self._start_webrtc()
            self._should_reconnect = self.auto_reconnect
            _notify(callback, "ok", "Audio streaming started")
        except Exception as e:
            logger.error("Error starting audio stream: %s", e, exc_info=True)
            _notify(callback, "error", str(e))

    async def _handle_stop_command(self, callback: Callable[..., None] | None = None) -> None:
        try:
            if self.pc is None:
                _notify(callback, "ok", "Audio stream not running")
                return
            self._should_reconnect = False
            await self.stop()
            _notify(callback, "ok", "Audio stream stopped")
        except Exception as e:
            logger.error("Error stopping audio stream: %s", e, exc_info=True)
            _notify(callback, "error", str(e))

    def _publish_message(self, topic: str, payload: dict[str, Any]) -> None:
        self.client.publish(topic, payload, qos=2)
        logger.info("Published to %s", topic)

    # -------------------------------------------------------------------------
    # Connection monitoring
    # -------------------------------------------------------------------------

    async def _monitor_connection(self, stop_event: asyncio.Event) -> None:
        reconnect_delay = 2.0
        max_attempts = 10
        attempt = 0
        while not stop_event.is_set() and self._is_running:
            if not self._should_reconnect or self.pc is None:
                await asyncio.sleep(1.0)
                continue
            if self._is_connection_lost():
                attempt = await self._attempt_reconnect(
                    stop_event, attempt, reconnect_delay, max_attempts
                )
                if attempt < 0:
                    break
            await asyncio.sleep(1.0)

    def _is_connection_lost(self) -> bool:
        state = getattr(self.pc, "connectionState", None)
        ice = getattr(self.pc, "iceConnectionState", None)
        bad = state in ("disconnected", "failed", "closed") or ice in (
            "disconnected",
            "failed",
            "closed",
        )
        if bad:
            self._bad_connection_checks += 1
            if self._bad_connection_checks < _CONNECTION_LOSS_CONFIRMATION_CHECKS:
                return False
            logger.warning("WebRTC audio connection lost")
            return True
        if self._bad_connection_checks > 0:
            self._bad_connection_checks = 0
        return False

    async def _close_peer_connection(self) -> None:
        """Close only the WebRTC peer connection and audio track.

        Unlike stop(), this does NOT touch _run_task / _run_stop_event, so it
        is safe to call from within _run_task (e.g. from _cleanup_run or
        _attempt_reconnect).
        """
        if self.pc is not None:
            try:
                await self.pc.close()
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error("Error closing peer connection: %s", e)
            finally:
                self.pc = None
        if self.streamer is not None:
            try:
                self.streamer.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error("Error closing audio track: %s", e)
            finally:
                self.streamer = None
        self._reset_state()

    async def _attempt_reconnect(
        self,
        stop_event: asyncio.Event,
        attempt: int,
        base_delay: float,
        max_attempts: int,
    ) -> int:
        try:
            try:
                # Close only the peer connection; do NOT call stop() here because
                # _attempt_reconnect runs inside _monitor_task, and stop() would
                # tear down _run_task and cancel _monitor_task itself.
                await self._close_peer_connection()
            except Exception as e:
                logger.warning("Error closing connection during reconnect: %s", e)
            await asyncio.sleep(base_delay)
            if not self._should_reconnect or stop_event.is_set():
                return -1
            logger.info("Reconnecting audio stream (attempt %s)...", attempt + 1)
            await self._start_webrtc()
            return 0
        except Exception as e:
            attempt += 1
            logger.error("Reconnect attempt failed: %s", e, exc_info=True)
            if attempt >= max_attempts:
                self._should_reconnect = False
                return -1
            await asyncio.sleep(min(base_delay * (2 ** attempt), 30.0))
            return attempt

    async def _cleanup_run(self) -> None:
        self._is_running = False
        self._should_reconnect = False
        self._event_loop = None
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        if self.pc is not None:
            try:
                # Do NOT call self.stop() here: _cleanup_run is invoked from
                # run_with_auto_reconnect's finally block, meaning we are
                # executing inside _run_task.  stop() waits on _run_task →
                # 5-second timeout + self-cancel on every clean shutdown.
                await self._close_peer_connection()
            except Exception as e:
                logger.error("Error closing connection during cleanup: %s", e)


# =============================================================================
# Microphone audio streamer
# =============================================================================


class MicrophoneAudioStreamer(BaseAudioStreamer):
    """Audio streamer for microphone capture via a get_audio() callback.

    Pass a get_audio callable that returns 20ms of s16 mono 48kHz (1920 bytes)
    or None for silence (e.g. from PyAudio or sounddevice).

    Example:
        >>> streamer = MicrophoneAudioStreamer(
        ...     client.mqtt,
        ...     twin_uuid="twin-id",
        ...     get_audio=my_capture_callback,
        ...     sensor_name="mic",
        ... )
        >>> await streamer.start()
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        get_audio: Callable[[], bytes | None],
        *,
        twin_uuid: str | None = None,
        turn_servers: list | None = None,
        auto_reconnect: bool = True,
        sensor_name: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        layout: str = DEFAULT_LAYOUT,
    ) -> None:
        super().__init__(
            client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            auto_reconnect=auto_reconnect,
            sensor_name=sensor_name,
        )
        self._get_audio = get_audio
        self._sample_rate = sample_rate
        self._layout = layout

    def initialize_track(self) -> BaseAudioTrack:
        return MicrophoneAudioTrack(
            self._get_audio,
            sample_rate=self._sample_rate,
            layout=self._layout,
        )
