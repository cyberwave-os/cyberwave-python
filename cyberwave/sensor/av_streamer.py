"""Multimedia streaming (video + audio) over a single WebRTC connection.

MultimediaStreamer sends both video and audio tracks through one
RTCPeerConnection, providing A/V synchronisation, lower overhead,
and a single TURN allocation compared to separate streamers.

When video and audio target different twin UUIDs, use separate
VirtualCameraStreamer / MicrophoneAudioStreamer instead.

Example:
    >>> from cyberwave.sensor.camera_virtual import VirtualVideoTrack
    >>> from cyberwave.sensor.microphone import MicrophoneAudioTrack
    >>>
    >>> streamer = MultimediaStreamer(
    ...     client=cw.mqtt,
    ...     create_video_track=lambda: VirtualVideoTrack(get_frame, width=640, height=480, fps=30),
    ...     create_audio_track=lambda: MicrophoneAudioTrack(get_audio),
    ...     twin_uuid="twin-id",
    ...     camera_name="rgb",
    ...     mic_name="audio",
    ... )
    >>> await streamer.start()
"""

from __future__ import annotations

import asyncio
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

from . import (
    BaseVideoTrack,
    CONNECTION_LOSS_CONFIRMATION_CHECKS,
    DEFAULT_TURN_SERVERS,
    SDK_EDGE_HEALTH_INTERVAL_SECONDS,
    SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
)
from .microphone import BaseAudioTrack, _strip_non_opus_audio

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient

logger = logging.getLogger(__name__)

# VP8 codec lines that aiortc inserts by default.  We strip them so only
# H264 remains, matching what the mediasoup SFU expects.
_VP8_PREFIXES = (
    "a=rtpmap:97",
    "a=rtpmap:98",
    "a=rtcp-fb:97 nack",
    "a=rtcp-fb:97 nack pli",
    "a=rtcp-fb:97 goog-remb",
    "a=rtcp-fb:98 nack",
    "a=rtcp-fb:98 nack pli",
    "a=rtcp-fb:98 goog-remb",
    "a=fmtp:98",
)


def _filter_multimedia_sdp(sdp: str) -> str:
    """Remove VP8 video codecs and non-Opus audio codecs from the SDP."""
    lines = sdp.split("\r\n")
    filtered: list[str] = []
    for line in lines:
        if line.startswith("m=video"):
            parts = line.split()
            filtered.append(" ".join(p for p in parts if p not in ("97", "98")))
        elif line.startswith(_VP8_PREFIXES):
            continue
        else:
            filtered.append(line)
    return _strip_non_opus_audio("\r\n".join(filtered))


class MultimediaStreamer:
    """Stream video **and** audio over a single WebRTC peer connection.

    Both tracks share one ICE/DTLS session, giving natural A/V
    synchronisation and halving the network setup cost.

    Args:
        client: Cyberwave MQTT client instance.
        create_video_track: Zero-arg factory that returns a fresh
            :class:`BaseVideoTrack`.  Called on every (re)connect.
        create_audio_track: Zero-arg factory that returns a fresh
            :class:`BaseAudioTrack`.  Called on every (re)connect.
        twin_uuid: UUID of the digital twin.
        camera_name: Sensor identifier echoed in the offer for the video
            track (used by the SFU to route answers and recordings).
        mic_name: Sensor identifier for the audio track.
        turn_servers: TURN/STUN server list (default: Cyberwave servers).
        auto_reconnect: Re-establish the connection on drops.
        recording: Whether the SFU should record the video track.
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        *,
        create_video_track: Callable[[], BaseVideoTrack],
        create_audio_track: Callable[[], BaseAudioTrack],
        twin_uuid: str,
        camera_name: Optional[str] = None,
        mic_name: Optional[str] = None,
        turn_servers: Optional[list] = None,
        auto_reconnect: bool = True,
        recording: bool = False,
        enable_health_check: bool = True,
    ) -> None:
        self.client = client
        self._create_video_track = create_video_track
        self._create_audio_track = create_audio_track
        self.twin_uuid = twin_uuid
        self.camera_name = camera_name
        self.mic_name = mic_name
        self.auto_reconnect = auto_reconnect
        self._should_record = recording
        self.enable_health_check = enable_health_check
        self.turn_servers = turn_servers if turn_servers is not None else DEFAULT_TURN_SERVERS

        # WebRTC state
        self.pc: Optional[RTCPeerConnection] = None
        self.video_track: Optional[BaseVideoTrack] = None
        self.audio_track: Optional[BaseAudioTrack] = None

        # Answer handling
        self._answer_received = False
        self._answer_data: Optional[dict[str, Any]] = None

        # Reconnect state
        self._should_reconnect = False
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bad_connection_checks = 0

        # Session tracking
        self._session_id: int = 0
        self._current_offer_session: int = 0
        self._subscribed_to_answer: bool = False

        # Background task (owned by start/stop)
        self._run_task: Optional[asyncio.Task] = None
        self._run_stop_event: Optional[asyncio.Event] = None

        # Health check state
        self._health_check: Optional[Any] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._last_frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start streaming with auto-reconnect in a background task.

        Call :meth:`stop` to tear everything down.
        """
        self._run_stop_event = asyncio.Event()
        self._run_task = asyncio.create_task(
            self._run_with_auto_reconnect(stop_event=self._run_stop_event)
        )

    async def stop(self) -> None:
        """Stop streaming and release all resources."""
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

        await self._close_peer_connection()
        logger.info("Multimedia streaming stopped")

    # ------------------------------------------------------------------
    # Internal lifecycle
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._answer_received = False
        self._answer_data = None
        self._bad_connection_checks = 0
        self._session_id += 1

    async def _start_webrtc(self) -> None:
        self._reset_state()
        self._current_offer_session = self._session_id

        logger.info("Starting multimedia WebRTC stream for twin %s", self.twin_uuid)
        await asyncio.sleep(0.1)
        await self._setup_webrtc()
        try:
            await self._perform_signaling()
        except Exception:
            await self._teardown_tracks_and_pc()
            raise
        logger.debug("Multimedia WebRTC connection established")
        asyncio.create_task(self._wait_and_publish_camera_sync_frame())
        if self.enable_health_check:
            self._start_health_check()

    async def _run_with_auto_reconnect(
        self,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        self._is_running = True
        self._event_loop = asyncio.get_running_loop()
        stop = stop_event or asyncio.Event()

        self._subscribe_to_commands()
        if not self._subscribed_to_answer:
            self._subscribe_to_answer()
            self._subscribed_to_answer = True

        if self.pc is None:
            try:
                await self._start_webrtc()
                self._should_reconnect = self.auto_reconnect
            except Exception as e:
                logger.error("Auto-start multimedia stream failed: %s", e, exc_info=True)

        if self.auto_reconnect:
            self._monitor_task = asyncio.create_task(self._monitor_connection(stop))

        _next_retry_at = time.monotonic() + 15.0
        _retry_backoff = 30.0
        _initial_connected = False

        try:
            while not stop.is_set() and self._is_running:
                if self.pc is None and self.auto_reconnect and not _initial_connected:
                    if time.monotonic() >= _next_retry_at:
                        try:
                            logger.info("No active multimedia stream — retrying offer...")
                            await self._start_webrtc()
                            self._should_reconnect = self.auto_reconnect
                            _initial_connected = True
                        except Exception as exc:
                            logger.info(
                                "Multimedia stream retry failed (%s). Will retry in %.0fs.",
                                exc,
                                _retry_backoff,
                            )
                            _next_retry_at = time.monotonic() + _retry_backoff
                            _retry_backoff = min(_retry_backoff * 2, 120.0)
                await asyncio.sleep(0.5)
        finally:
            await self._cleanup_run()

    # ------------------------------------------------------------------
    # WebRTC setup
    # ------------------------------------------------------------------

    async def _setup_webrtc(self) -> None:
        self.video_track = self._create_video_track()
        self.audio_track = self._create_audio_track()

        ice_servers = [RTCIceServer(**s) for s in self.turn_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        self._setup_pc_handlers()
        self.pc.addTrack(self.video_track)
        self.pc.addTrack(self.audio_track)

    def _setup_pc_handlers(self) -> None:
        @self.pc.on("connectionstatechange")
        def _on_connectionstatechange() -> None:
            logger.info("WebRTC connection state: %s", self.pc.connectionState)

        @self.pc.on("iceconnectionstatechange")
        def _on_iceconnectionstatechange() -> None:
            logger.info("WebRTC ICE connection state: %s", self.pc.iceConnectionState)

    # ------------------------------------------------------------------
    # Signaling
    # ------------------------------------------------------------------

    async def _perform_signaling(self) -> None:
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        deadline = time.monotonic() + 30.0
        while self.pc.iceGatheringState != "complete":
            if time.monotonic() > deadline:
                raise TimeoutError("ICE gathering timed out after 30s")
            await asyncio.sleep(0.1)

        sdp = _filter_multimedia_sdp(self.pc.localDescription.sdp)
        self._send_offer(sdp)
        await self._wait_for_answer()

    def _send_offer(self, sdp: str) -> None:
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-offer"

        video_attrs = self.video_track.get_stream_attributes() if self.video_track else {}
        audio_attrs = self.audio_track.get_stream_attributes() if self.audio_track else {}

        offer_payload: dict[str, Any] = {
            "target": "backend",
            "sender": "edge",
            "type": self.pc.localDescription.type,
            "sdp": sdp,
            "timestamp": time.time(),
            "recording": self._should_record,
            "frontend_type": "multimedia",
            "stream_attributes": {
                "video": video_attrs,
                "audio": audio_attrs,
            },
            "sensor": self.camera_name,
            "track_id": self.video_track.id if self.video_track else None,
            "session_id": f"{self.client.client_id}_multimedia",
        }
        self.client.publish(topic, offer_payload, qos=2)
        logger.info("Published multimedia offer to %s", topic)

    async def _wait_for_answer(self, timeout: float = 60.0) -> None:
        start = time.monotonic()
        while not self._answer_received:
            if time.monotonic() - start > timeout:
                raise TimeoutError("Timeout waiting for WebRTC answer")
            await asyncio.sleep(0.1)

        if self._answer_data is None:
            raise RuntimeError("Answer received but data is None")

        answer = (
            json.loads(self._answer_data)
            if isinstance(self._answer_data, str)
            else self._answer_data
        )
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    def _subscribe_to_answer(self) -> None:
        prefix = self.client.topic_prefix
        answer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-answer"
        logger.info("Subscribing to WebRTC answer topic: %s", answer_topic)

        def on_answer(data: Any) -> None:
            try:
                if self._session_id != self._current_offer_session:
                    return
                payload = data if isinstance(data, dict) else json.loads(data)
                if payload.get("type") == "offer":
                    return
                if payload.get("type") == "answer" and payload.get("target") == "edge":
                    sdp = payload.get("sdp", "")
                    # A multimedia answer must contain both media sections.
                    # Single-track answers (video-only or audio-only) are left
                    # for the standalone streamer that sent the original offer.
                    if "m=video" not in sdp or "m=audio" not in sdp:
                        logger.debug(
                            "Ignoring answer without both m=video and m=audio"
                        )
                        return
                    self._answer_data = payload
                    self._answer_received = True
                elif payload.get("type") == "candidate" and payload.get("target") == "edge":
                    self._handle_candidate(payload)
            except Exception as e:
                logger.error("Error in multimedia on_answer: %s", e)

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

    def _subscribe_to_commands(self) -> None:
        prefix = self.client.topic_prefix
        command_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/command"

        def on_command(data: Any) -> None:
            try:
                payload = data if isinstance(data, dict) else json.loads(data)
                if "status" in payload:
                    return
                cmd = payload.get("command")
                if cmd in ("start_video", "start_audio"):
                    asyncio.run_coroutine_threadsafe(
                        self._handle_start_command(), self._event_loop
                    )
                elif cmd in ("stop_video", "stop_audio"):
                    asyncio.run_coroutine_threadsafe(
                        self._handle_stop_command(), self._event_loop
                    )
            except Exception as e:
                logger.error("Error processing multimedia command: %s", e, exc_info=True)

        self.client.subscribe(command_topic, on_command)

    async def _handle_start_command(self) -> None:
        if self.pc is not None:
            return
        try:
            await self._start_webrtc()
            self._should_reconnect = self.auto_reconnect
        except Exception as e:
            logger.error("Error starting multimedia stream: %s", e, exc_info=True)

    async def _handle_stop_command(self) -> None:
        if self.pc is None:
            return
        try:
            self._should_reconnect = False
            # Only close the PC; don't call stop() which tears down _run_task
            await self._close_peer_connection()
        except Exception as e:
            logger.error("Error stopping multimedia stream: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Connection monitoring
    # ------------------------------------------------------------------

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
            if self._bad_connection_checks < CONNECTION_LOSS_CONFIRMATION_CHECKS:
                return False
            logger.warning("Multimedia WebRTC connection lost")
            return True
        if self._bad_connection_checks > 0:
            self._bad_connection_checks = 0
        return False

    async def _attempt_reconnect(
        self,
        stop_event: asyncio.Event,
        attempt: int,
        base_delay: float,
        max_attempts: int,
    ) -> int:
        try:
            await self._close_peer_connection()
            await asyncio.sleep(base_delay)
            if not self._should_reconnect or stop_event.is_set():
                return -1
            logger.info("Reconnecting multimedia stream (attempt %s)...", attempt + 1)
            await self._start_webrtc()
            return 0
        except Exception as e:
            attempt += 1
            logger.error("Reconnect attempt failed: %s", e, exc_info=True)
            if attempt >= max_attempts:
                self._should_reconnect = False
                return -1
            await asyncio.sleep(min(base_delay * (2**attempt), 30.0))
            return attempt

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        if not self.enable_health_check or not self.twin_uuid:
            return
        try:
            from ..edge.health import EdgeHealthCheck

            self._health_check = EdgeHealthCheck(
                mqtt_client=self.client,
                twin_uuids=[self.twin_uuid],
                edge_id=self.twin_uuid,
                stale_timeout=SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
                interval=SDK_EDGE_HEALTH_INTERVAL_SECONDS,
            )
            self._health_check.start()
            self._last_frame_count = 0
            self._health_monitor_task = asyncio.create_task(
                self._monitor_frame_count()
            )
            logger.debug("Multimedia health check started")
        except Exception as e:
            logger.warning("Failed to start multimedia health check: %s", e)

    def _stop_health_check(self) -> None:
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            self._health_monitor_task = None
        if self._health_check:
            try:
                self._health_check.stop()
            except Exception as e:
                logger.warning("Error stopping multimedia health check: %s", e)
            self._health_check = None
        self._last_frame_count = 0

    async def _monitor_frame_count(self) -> None:
        """Poll the video track's frame counter and forward deltas to health check."""
        while self._is_running or self.pc is not None:
            try:
                if self.video_track and self._health_check:
                    current = getattr(self.video_track, "frame_count", 0)
                    if current < self._last_frame_count:
                        self._last_frame_count = current
                    if current > self._last_frame_count:
                        for _ in range(current - self._last_frame_count):
                            self._health_check.update_frame_count()
                        self._last_frame_count = current
            except Exception as e:
                logger.debug("Multimedia health monitor error: %s", e)
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Sync-frame telemetry
    # ------------------------------------------------------------------

    def _publish_camera_sync_frame(
        self, pts: int, timestamp: float, timestamp_monotonic: float
    ) -> None:
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/telemetry"
        payload = {
            "type": "camera_sync_frame",
            "sender": "edge",
            "pts": pts,
            "timestamp": timestamp,
            "timestamp_monotonic": timestamp_monotonic,
            "track_id": self.video_track.id if self.video_track else None,
            "twin_uuid": self.twin_uuid,
            "sensor": self.camera_name,
        }
        self.client.publish(topic, payload, qos=2)
        logger.info(
            "Published camera_sync_frame: pts=%s, timestamp=%.3f", pts, timestamp
        )

    async def _wait_and_publish_camera_sync_frame(
        self, sync_frame: int = 30, timeout: float = 10.0
    ) -> None:
        """Wait for the video track to reach *sync_frame* and publish the anchor."""
        if self.video_track:
            self.video_track.sync_frame_target = sync_frame

        start_time = time.time()
        while self.video_track and self.video_track.sync_frame_pts is None:
            if time.time() - start_time > timeout:
                logger.warning(
                    "Timeout waiting for multimedia sync frame %s, "
                    "current frame: %s",
                    sync_frame,
                    self.video_track.frame_count if self.video_track else 0,
                )
                return
            await asyncio.sleep(0.05)

        if self.video_track and self.video_track.sync_frame_pts is not None:
            pts = self.video_track.sync_frame_pts
            timestamp = self.video_track.sync_frame_timestamp
            timestamp_monotonic = self.video_track.sync_frame_timestamp_monotonic
            if timestamp is not None:
                self._publish_camera_sync_frame(
                    pts, timestamp, timestamp_monotonic or 0.0
                )

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    async def _close_peer_connection(self) -> None:
        """Close the PC and both tracks.  Safe to call from within _run_task."""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        self._stop_health_check()
        if self.pc is not None:
            try:
                await self.pc.close()
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error("Error closing peer connection: %s", e)
            finally:
                self.pc = None
        if self.video_track is not None:
            try:
                self.video_track.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error("Error closing video track: %s", e)
            finally:
                self.video_track = None
        if self.audio_track is not None:
            try:
                self.audio_track.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error("Error closing audio track: %s", e)
            finally:
                self.audio_track = None
        self._reset_state()

    async def _teardown_tracks_and_pc(self) -> None:
        """Quick cleanup after failed signaling (no sleeps)."""
        if self.pc is not None:
            try:
                await self.pc.close()
                await asyncio.sleep(0.5)
            except Exception:
                pass
            self.pc = None
        for track in (self.video_track, self.audio_track):
            if track is not None:
                try:
                    track.close()
                except Exception:
                    pass
        self.video_track = None
        self.audio_track = None

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
        await self._close_peer_connection()
