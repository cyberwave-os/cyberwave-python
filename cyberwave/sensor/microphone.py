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
import platform
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
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
DEFAULT_AUDIO_RECORDING = False
DEFAULT_AUTO_RECONNECT = True
DEFAULT_FRONTEND_TYPE = "audio"
DEFAULT_STREAM_SOURCE = "live"
DEFAULT_STREAM_INSTANCE_ID = "default"

# Shared WebRTC routing key for mic + speaker legs (``sensor`` field in offers).
# Must match ``DEFAULT_AUDIO_SENSOR_ID`` in media-service ``audio_sensor.rs``.
DEFAULT_AUDIO_SENSOR_ID = "audio"
DEFAULT_MIC_NAME = DEFAULT_AUDIO_SENSOR_ID

# Active microphone twin sensor *types* — edge producers only.
# Keep in sync with ``MICROPHONE_SENSOR_TYPES`` in ``audio_sensor.rs``.
MICROPHONE_SENSOR_TYPES = frozenset(
    {"mic", "microphone", "audio_in", "audio", "audio_mono", "audio_stereo"}
)

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


def _get_sounddevice_module() -> Any | None:
    try:
        import sounddevice as sd  # type: ignore[import]

        return sd
    except Exception:
        return None


def list_host_microphone_devices() -> tuple[list[dict[str, Any]], int | None]:
    """List host audio input devices using ``sounddevice``.

    Install ``cyberwave[microphone]`` to include the host capture dependencies.
    """
    sd = _get_sounddevice_module()
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed; install with: pip install 'cyberwave[microphone]'"
        )

    raw_devices = sd.query_devices()
    default_device = sd.default.device
    candidate = default_device[0] if isinstance(default_device, tuple) else default_device
    default_input_index = candidate if isinstance(candidate, int) and candidate >= 0 else None

    devices: list[dict[str, Any]] = []
    for index, device in enumerate(raw_devices):
        max_input = int(device.get("max_input_channels", 0) or 0)
        if max_input <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": str(device.get("name", f"input-{index}")),
                "max_input_channels": max_input,
                "default_samplerate": float(device.get("default_samplerate", 0.0) or 0.0),
                "hostapi": int(device.get("hostapi", -1) or -1),
            }
        )
    return devices, default_input_index


def check_host_microphone_settings(
    *,
    device: int | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
) -> None:
    """Validate host microphone settings with ``sounddevice``."""
    sd = _get_sounddevice_module()
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed; install with: pip install 'cyberwave[microphone]'"
        )
    sd.check_input_settings(
        device=device,
        channels=channels,
        samplerate=sample_rate,
        dtype="int16",
    )


def create_linux_microphone_monitor() -> Any | None:
    """Create a Linux ``pyudev`` monitor for sound-device hotplug events."""
    if platform.system().lower() != "linux":
        return None
    try:
        import pyudev  # type: ignore[import]
    except Exception:
        return None

    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem="sound")
    return monitor


class HostMicrophoneCapture:
    """Capture fixed-size int16 chunks from a host microphone.

    Use :meth:`get_audio` as the callback for :class:`MicrophoneAudioStreamer`.
    The default format is 20 ms of s16 mono 48 kHz audio.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        frames_per_chunk: int | None = None,
        device_index: int | None = None,
        queue_chunks: int = 32,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_chunk = frames_per_chunk or int(AUDIO_PTIME * sample_rate)
        self.device_index = device_index
        self.bytes_per_chunk = self.frames_per_chunk * channels * 2
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max(1, queue_chunks))
        self._on_chunk = on_chunk
        self._stream: Any | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Start host microphone capture. This method is idempotent."""
        with self._lock:
            if self._stream is not None:
                return
            sd = _get_sounddevice_module()
            if sd is None:
                raise RuntimeError(
                    "sounddevice is not installed; install with: pip install 'cyberwave[microphone]'"
                )

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.frames_per_chunk,
                device=self.device_index,
                callback=self._on_audio_callback,
                latency="low",
            )
            try:
                stream.start()
            except Exception:
                try:
                    stream.close()
                except Exception:
                    logger.exception("Error while closing audio stream after start failure")
                self.clear()
                raise
            self._stream = stream

    def stop(self) -> None:
        """Stop host microphone capture and clear buffered chunks."""
        with self._lock:
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Error while stopping sounddevice input stream")

        self.clear()

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def get_audio(self, timeout: float = AUDIO_PTIME) -> bytes | None:
        """Return one captured chunk or ``None`` when no chunk is available."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _queue_chunk(self, chunk: bytes) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)

    def _on_audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.debug("sounddevice callback status: %s", status)

        chunk = np.asarray(indata, dtype=np.int16).copy()
        if chunk.size == 0:
            return
        self._queue_chunk(chunk.tobytes())
        if self._on_chunk is not None:
            try:
                self._on_chunk(chunk)
            except Exception:
                logger.exception("Host microphone chunk callback failed")


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

    The ``frame_count`` counter exists for the streamer's health-check
    poller to detect liveness — concrete subclasses are responsible for
    bumping it once per emitted frame.  Without that, an audio twin
    would always look stale on the dashboard even when the track is
    happily streaming over WebRTC (the pre-CYB-2005 bug).
    """

    def __init__(self) -> None:
        super().__init__()
        self._closed = False
        self.frame_count: int = 0

    def get_stream_attributes(self) -> dict[str, Any]:
        """Stream attributes included in the WebRTC offer payload."""
        return {}

    def get_stream_config(self) -> dict[str, Any] | None:
        """Return a typed ``stream_config`` block for ``edge_health``, or ``None``.

        Mirrors :meth:`cyberwave.sensor.base_video.BaseVideoStreamer._build_stream_config`
        for the audio side.  Default implementation returns ``None`` so a
        generic ``BaseAudioTrack`` stays on the no-``stream_config`` wire
        shape; concrete subclasses (``MicrophoneAudioTrack``, future
        codec-specific tracks) override to declare their kind, source,
        and runtime parameters.

        Called on every heartbeat via the ``stream_config_provider`` the
        streamer wires into ``EdgeHealthCheck``, so subclasses can return
        post-negotiation values (Opus FEC state, ALSA-reopened device
        path, ...) without registering at startup.  Credentials in
        ``source`` must already be masked by the override; the publisher
        does not redact.
        """
        return None

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

    def get_stream_config(self) -> dict[str, Any] | None:
        """Advertise the microphone config in every ``edge_health`` heartbeat.

        The dashboard otherwise has to guess "is this twin an audio
        source?" from the asset spec, which is fragile when the twin
        carries multiple sensors or was provisioned without a sensor
        kind.  Wiring the ``audio`` discriminator on the wire moves
        that decision into the producer where the truth lives.

        Intentionally omits ``source``.  Across the rest of the SDK
        ``source`` is a device path, URL, or ROS topic (camera publishes
        ``/dev/video0``, lidar publishes ``/point_cloud2``); a WebRTC
        microphone has no equivalent — the host ALSA / CoreAudio device
        path is a security leak (it would describe the operator's host
        filesystem), and publishing the codec instead would overload
        the field's semantics.  The audio-kind validator accepts a
        missing ``source`` for exactly this reason.  Drivers that DO
        have a meaningful identifier (e.g. a JACK port name on a
        multi-mic edge) can override and attach it.
        """
        return {
            "kind": "audio",
            "sample_rate_hz": self.sample_rate,
            "channels": 2 if self.layout == "stereo" else 1,
            "codec": "opus",
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
        # Liveness signal for the streamer's health-check poller — bump
        # AFTER the wait/encode so a track that's still blocking on the
        # get_audio callable doesn't look "fresh" forever.
        self.frame_count += 1

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
        auto_reconnect: bool = DEFAULT_AUTO_RECONNECT,
        mic_name: Optional[str] = None,
        recording: bool = DEFAULT_AUDIO_RECORDING,
        frontend_type: str = DEFAULT_FRONTEND_TYPE,
        stream_source: Optional[str] = None,
        stream_instance_id: Optional[str] = None,
        enable_health_check: bool = True,
    ) -> None:
        self.client = client
        self.twin_uuid: str | None = twin_uuid
        self.mic_name: Optional[str] = mic_name  # e.g. "mic", "audio" (for multi-stream routing)
        self.auto_reconnect = auto_reconnect
        self.turn_servers = turn_servers if turn_servers is not None else _AUDIO_TURN_SERVERS
        self.recording = bool(recording)
        self.frontend_type = frontend_type
        self.stream_source = stream_source
        self.stream_instance_id = stream_instance_id
        self.enable_health_check = enable_health_check

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

        # ``edge_health`` plumbing — pre-CYB-2005 ``BaseAudioStreamer``
        # didn't publish a heartbeat at all, which made paired
        # microphone twins always show "Edge service not running" in
        # the dashboard even when audio was streaming fine over WebRTC.
        # The lifecycle mirrors ``av_streamer.MultimediaStreamer``: start
        # after a successful ``_start_webrtc``, stop on disconnect /
        # ``stop()``.
        self._health_check: Any = None
        self._health_monitor_task: asyncio.Task | None = None
        self._last_frame_count: int = 0

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
        if self.enable_health_check:
            self._start_health_check()

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
        # Tear down the health publisher first so we don't keep
        # emitting heartbeats after the WebRTC track is gone — would
        # otherwise look like a phantom-live audio stream in the
        # dashboard for ``stale_timeout`` seconds.
        self._stop_health_check()

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
            "recording": self.recording,
            "stream_attributes": stream_attributes,
            "sensor": self.mic_name,
            "track_id": self.streamer.id if self.streamer else None,
            "frontend_type": self.frontend_type,
            "session_id": f"{self.client.client_id}_{self.mic_name}",
            # Active twin — produces audio into the SFU (passive speakers consume).
            # ``sensor_type`` matches catalog microphone metadata (``type: "audio"``).
            "sensor_type": "audio",
            "role": "producer",
        }
        if self.stream_source:
            offer_payload["stream_source"] = self.stream_source
        if self.stream_instance_id:
            offer_payload["stream_instance_id"] = self.stream_instance_id
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
                logger.debug(
                    "Audio on_answer: type=%s target=%s sensor=%s stream_source=%s stream_instance_id=%s has_m_audio=%s",
                    payload.get("type"),
                    payload.get("target"),
                    payload.get("sensor") or payload.get("camera"),
                    payload.get("stream_source"),
                    payload.get("stream_instance_id"),
                    "m=audio" in payload.get("sdp", ""),
                )
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
                    expected = self.mic_name if self.mic_name is not None else "default"
                    answer_stream_source = payload.get("stream_source") or DEFAULT_STREAM_SOURCE
                    expected_stream_source = self.stream_source or DEFAULT_STREAM_SOURCE
                    answer_stream_instance_id = (
                        payload.get("stream_instance_id") or DEFAULT_STREAM_INSTANCE_ID
                    )
                    expected_stream_instance_id = (
                        self.stream_instance_id or DEFAULT_STREAM_INSTANCE_ID
                    )
                    if (
                        (answer_sensor is None or answer_sensor == expected)
                        and answer_stream_source == expected_stream_source
                        and answer_stream_instance_id == expected_stream_instance_id
                    ):
                        self._answer_data = payload
                        self._answer_received = True
                    else:
                        logger.warning(
                            "Audio answer rejected: stream identity mismatch "
                            "(expected_sensor=%r, got_sensor=%r, "
                            "expected_stream_source=%r, got_stream_source=%r, "
                            "expected_stream_instance_id=%r, got_stream_instance_id=%r)",
                            expected,
                            answer_sensor,
                            expected_stream_source,
                            answer_stream_source,
                            expected_stream_instance_id,
                            answer_stream_instance_id,
                        )
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
                self._publish_webrtc_recording_command("start_recording")
                _notify(callback, "ok", "Audio recording started")
                return
            await self._start_webrtc()
            self._should_reconnect = self.auto_reconnect
            self._publish_webrtc_recording_command("start_recording")
            _notify(callback, "ok", "Audio streaming started")
        except Exception as e:
            logger.error("Error starting audio stream: %s", e, exc_info=True)
            _notify(callback, "error", str(e))

    async def _handle_stop_command(self, callback: Callable[..., None] | None = None) -> None:
        try:
            if self.pc is None:
                _notify(callback, "ok", "Audio stream not running")
                return
            self._publish_webrtc_recording_command("stop_recording")
            _notify(callback, "ok", "Audio recording stopped")
        except Exception as e:
            logger.error("Error stopping audio stream: %s", e, exc_info=True)
            _notify(callback, "error", str(e))

    def _publish_webrtc_recording_command(self, command: str) -> None:
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before publishing recording commands")
        prefix = self.client.topic_prefix
        command_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-command"
        self._publish_message(
            command_topic,
            {
                "command": command,
                "source_type": "edge",
                "sensor": self.mic_name,
            },
        )

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
        # Stop the heartbeat first so the dashboard doesn't see a brief
        # "stale stream" flash between cleanup and final teardown.
        self._stop_health_check()
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

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Spin up an ``EdgeHealthCheck`` for this audio track.

        Mirrors the pattern in ``av_streamer.MultimediaStreamer._start_health_check``
        and ``base_video.BaseVideoStreamer._start_health_check``: construct
        with a ``stream_config_provider`` so the wire reflects current
        track state on every heartbeat (sample rate / channels / codec
        post-negotiation), start the publisher, and kick off a frame
        monitor that forwards the track's frame counter to the
        publisher so ``is_stale`` reacts to real liveness rather than
        a static "we're connected" flag.

        Pre-CYB-2005, ``BaseAudioStreamer`` skipped this entirely and
        paired microphone twins always rendered "Edge service not
        running" in the dashboard even when audio was streaming fine
        over WebRTC.
        """
        if not self.enable_health_check or not self.twin_uuid:
            return
        try:
            from ..edge.health import EdgeHealthCheck
            from .base_video import (
                SDK_EDGE_HEALTH_INTERVAL_SECONDS,
                SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
            )

            self._health_check = EdgeHealthCheck(
                mqtt_client=self.client,
                twin_uuids=[self.twin_uuid],
                edge_id=self.twin_uuid,
                stale_timeout=SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
                interval=SDK_EDGE_HEALTH_INTERVAL_SECONDS,
                stream_config_provider=self._collect_stream_configs,
            )
            self._health_check.start()
            self._last_frame_count = 0
            # ``_start_health_check`` is normally called from inside
            # the async ``_start_webrtc`` so an event loop is always
            # running.  Guard for sync callers (test harness, manual
            # repls) by closing the coroutine cleanly when no loop is
            # available — without this, the unawaited coroutine
            # triggers ``RuntimeWarning`` on GC.
            monitor_coro = self._monitor_frame_count()
            try:
                self._health_monitor_task = asyncio.create_task(monitor_coro)
            except RuntimeError:
                monitor_coro.close()
                self._health_monitor_task = None
            logger.debug("Audio health check started")
        except Exception as e:
            # Never let a health-check failure tank the audio stream
            # itself — the WebRTC track is the load-bearing surface.
            logger.warning("Failed to start audio health check: %s", e)

    def _stop_health_check(self) -> None:
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            self._health_monitor_task = None
        if self._health_check:
            try:
                self._health_check.stop()
            except Exception as e:
                logger.warning("Error stopping audio health check: %s", e)
            self._health_check = None
        self._last_frame_count = 0

    def _collect_stream_configs(self) -> dict[str, dict[str, Any]]:
        """Bridge from the track's ``get_stream_config()`` to the provider shape.

        ``EdgeHealthCheck.stream_config_provider`` returns
        ``{stream_id: config}``; ``BaseAudioStreamer`` is single-stream
        by design, so the dict has at most one entry under the
        canonical ``"stream"`` key.  Returns ``{}`` when the track
        hasn't been initialised yet, the track is a generic
        ``BaseAudioTrack`` that hasn't overridden the hook, or the
        override raises — the heartbeat must keep flowing in all of
        those.
        """
        track = self.streamer
        if track is None:
            return {}
        try:
            cfg = track.get_stream_config()
        except Exception as exc:
            logger.debug("Audio track get_stream_config raised: %s", exc)
            return {}
        if cfg is None:
            return {}
        return {"stream": cfg}

    async def _monitor_frame_count(self) -> None:
        """Forward audio-track liveness to the health publisher.

        Polls every 100 ms.  When the track's ``frame_count`` has
        advanced since the previous poll, calls
        :meth:`EdgeHealthCheck.mark_alive` **once** — not once per
        emitted audio frame.

        Per-frame forwarding (the pattern the video side uses via
        ``update_frame_count``) would put ``fps: 50.0`` and
        ``frames_sent: <packet count>`` on the wire for a microphone,
        because aiortc emits a 20 ms Opus frame at 50 Hz.  Those
        numbers are correct WebRTC terminology but operationally
        meaningless — sample rate / channels live in
        ``stream_config`` and that's what the dashboard renders.  See
        ``EdgeHealthCheck.mark_alive`` for the full rationale.

        The track-level ``frame_count`` counter is still incremented
        in ``MicrophoneAudioTrack.recv`` because we need a monotone
        signal to detect "new frames since last poll" — we just
        don't forward each increment.
        """
        while self._is_running or self.pc is not None:
            try:
                track = self.streamer
                if track is not None and self._health_check is not None:
                    current = getattr(track, "frame_count", 0)
                    if current < self._last_frame_count:
                        # Reset on track replacement / reconnect.
                        self._last_frame_count = current
                    if current > self._last_frame_count:
                        self._health_check.mark_alive()
                        self._last_frame_count = current
            except Exception as e:
                logger.debug("Audio health monitor error: %s", e)
            await asyncio.sleep(0.1)


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
        ...     mic_name="mic",
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
        auto_reconnect: bool = DEFAULT_AUTO_RECONNECT,
        mic_name: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        layout: str = DEFAULT_LAYOUT,
        recording: bool = DEFAULT_AUDIO_RECORDING,
        frontend_type: str = DEFAULT_FRONTEND_TYPE,
        stream_source: Optional[str] = None,
        stream_instance_id: Optional[str] = None,
        enable_health_check: bool = True,
    ) -> None:
        super().__init__(
            client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            auto_reconnect=auto_reconnect,
            mic_name=mic_name,
            recording=recording,
            frontend_type=frontend_type,
            stream_source=stream_source,
            stream_instance_id=stream_instance_id,
            enable_health_check=enable_health_check,
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
