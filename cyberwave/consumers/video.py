"""Consume an ongoing WebRTC video stream from the backend's SFU.

The SDK acts as an SFU consumer (the browser-equivalent of a ``<video>`` tag):
it publishes a ``recvonly`` offer with ``sender="client_python_sdk"`` and
decodes the inbound track into the latest frame.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Optional

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from ..exceptions import NoOngoingVideoStreamAvailable
from ..sensor.base_video import DEFAULT_TURN_SERVERS

logger = logging.getLogger(__name__)

_ICE_GATHERING_TIMEOUT_S = 15.0


class IncomingVideoStream:
    """A live WebRTC video consumer. Construct via ``twin.camera.get_video()``."""

    SENDER = "client_python_sdk"

    def __init__(
        self,
        mqtt_client: Any,
        twin_uuid: str,
        *,
        sensor_id: Optional[str] = None,
        stream_source: Optional[str] = None,
        stream_instance_id: Optional[str] = None,
        frontend_type: str = "rgb",
        timeout: float = 30.0,
        turn_servers: Optional[list] = None,
        sender: Optional[str] = None,
    ) -> None:
        self._mqtt = mqtt_client
        self._twin_uuid = twin_uuid
        self._sensor_id = sensor_id
        self._stream_source = stream_source
        self._stream_instance_id = stream_instance_id
        self._frontend_type = frontend_type
        self._timeout = timeout
        self._turn_servers = turn_servers
        # Fanout identity. Defaults to "client_python_sdk"; pass "frontend" for
        # compatibility with older backend deployments that predate SDK-specific
        # routing.
        self._sender = sender or self.SENDER

        prefix = getattr(mqtt_client, "topic_prefix", "") or ""
        client_id = getattr(mqtt_client, "client_id", "sdk")
        self._session_id = f"{client_id}-{uuid.uuid4().hex[:6]}"
        self._offer_topic = f"{prefix}cyberwave/twin/{twin_uuid}/webrtc-offer"
        self._answer_topic = f"{prefix}cyberwave/twin/{twin_uuid}/webrtc-answer"
        self._subscriber_key = f"video_consumer_{self._session_id}"

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._pc: Optional[RTCPeerConnection] = None
        self._answer_future: Optional[asyncio.Future] = None
        self._decode_task: Optional[asyncio.Task] = None
        self._reply_handler = None  # set once the answer subscription is live
        self._current_frame = None
        self._frame_lock = threading.Lock()
        self._ended = False
        self._stopped = False

    def _build_offer_payload(self, sdp: str) -> dict:
        payload = {
            # "backend" is the required target value for this offer/answer
            # exchange; this matches exactly what the browser frontend publishes.
            "target": "backend",
            "sender": self._sender,
            "type": "offer",
            "sdp": sdp,
            "frontend_type": self._frontend_type,
            "sensor": self._sensor_id,
            "session_id": self._session_id,
            "timestamp": time.time(),
        }
        if self._stream_source:
            payload["stream_source"] = self._stream_source
        if self._stream_instance_id:
            payload["stream_instance_id"] = self._stream_instance_id
        return payload

    @staticmethod
    def _classify_reply(
        data: Any, session_id: str
    ) -> Optional[tuple[str, Optional[str]]]:
        """Classify an inbound webrtc-answer-topic message for *our* session.

        Returns ``("answer", sdp)``, ``("unavailable", message)``, or ``None``
        when the message is not actionable for this consumer.
        """
        if not isinstance(data, dict) or data.get("session_id") != session_id:
            return None
        kind = data.get("type")
        if kind == "answer":
            sdp = data.get("sdp")
            return ("answer", sdp) if sdp else None
        if kind in ("wait", "error"):
            message = (
                data.get("message")
                or data.get("error")
                or "No ongoing video stream available"
            )
            return ("unavailable", message)
        return None

    def _format_frame(self, arr: Any, format: str) -> Any:
        if format == "numpy":
            return arr.copy()
        import cv2

        ok, encoded = cv2.imencode(".jpg", arr)
        if not ok:
            return None
        from ..twin._helpers import _decode_frame

        return _decode_frame(encoded.tobytes(), format)

    # --- lifecycle -------------------------------------------------------

    def start(self) -> "IncomingVideoStream":
        """Negotiate the consumer connection. Blocks until connected, or raises
        ``NoOngoingVideoStreamAvailable`` (wait/error reply) or ``TimeoutError``
        (no reply within ``timeout``)."""
        self._thread = threading.Thread(
            target=self._run_loop, name=f"video-consumer-{self._session_id}", daemon=True
        )
        self._thread.start()
        self._loop_ready.wait()
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        try:
            future.result()
        except BaseException:
            self.stop()
            raise
        return self

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def _ice_servers(self) -> list:
        turn = self._turn_servers if self._turn_servers is not None else DEFAULT_TURN_SERVERS
        servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        for entry in turn:
            servers.append(
                RTCIceServer(
                    urls=entry["urls"],
                    username=entry.get("username"),
                    credential=entry.get("credential"),
                )
            )
        return servers

    async def _connect(self) -> None:
        self._pc = RTCPeerConnection(RTCConfiguration(iceServers=self._ice_servers()))
        self._answer_future = self._loop.create_future()

        @self._pc.on("track")
        def _on_track(track):  # noqa: ANN001
            logger.info("Video consumer: received %s track", track.kind)
            if track.kind == "video":
                self._start_decode(track)

        @self._pc.on("connectionstatechange")
        async def _on_state():
            logger.info(
                "Video consumer: connection state -> %s", self._pc.connectionState
            )
            if self._pc.connectionState in ("failed", "closed", "disconnected"):
                with self._frame_lock:
                    self._current_frame = None
                self._ended = True

        self._pc.addTransceiver("video", direction="recvonly")
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        await self._wait_ice_complete()

        # Subscribe BEFORE publishing so we never miss the answer.
        self._reply_handler = self._on_reply
        self._mqtt.subscribe(
            self._answer_topic, self._reply_handler, subscriber_key=self._subscriber_key
        )
        logger.info(
            "Video consumer: published offer to %s (session=%s), awaiting answer",
            self._offer_topic,
            self._session_id,
        )
        self._mqtt.publish(
            self._offer_topic,
            self._build_offer_payload(self._pc.localDescription.sdp),
            qos=0,
        )

        try:
            kind, data = await asyncio.wait_for(self._answer_future, timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"No WebRTC answer for twin {self._twin_uuid} within {self._timeout}s"
            ) from exc

        if kind == "unavailable":
            raise NoOngoingVideoStreamAvailable(
                data or "No ongoing video stream available"
            )
        logger.info("Video consumer: answer received, setting remote description")
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=data, type="answer")
        )
        # Belt-and-suspenders: the "track" event is the primary path, but for an
        # offerer with a recvonly transceiver we also start decoding directly
        # from the receiver's track so a missed/late event never leaves us
        # connected-but-frameless.
        for receiver in self._pc.getReceivers():
            track = getattr(receiver, "track", None)
            if track is not None and track.kind == "video":
                self._start_decode(track)
        logger.info("Video consumer: connected; decode loop running")

    def _start_decode(self, track) -> None:  # noqa: ANN001
        """Start the decode loop for *track* exactly once (idempotent across the
        ``on_track`` event and the post-answer receiver sweep)."""
        if self._decode_task is not None and not self._decode_task.done():
            return
        self._decode_task = asyncio.ensure_future(self._decode_loop(track))

    async def _wait_ice_complete(self) -> None:
        if self._pc.iceGatheringState == "complete":
            return
        done = self._loop.create_future()

        @self._pc.on("icegatheringstatechange")
        def _on_ice():
            if self._pc.iceGatheringState == "complete" and not done.done():
                done.set_result(True)

        try:
            await asyncio.wait_for(done, timeout=_ICE_GATHERING_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.debug("ICE gathering timed out; sending offer with partial candidates")

    # --- inbound answer (called on the paho network thread) --------------

    def _on_reply(self, data: Any) -> None:
        result = self._classify_reply(data, self._session_id)
        if result is None:
            return
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._resolve_answer, result)

    def _resolve_answer(self, result: tuple) -> None:
        if self._answer_future is not None and not self._answer_future.done():
            self._answer_future.set_result(result)

    # --- decode loop -----------------------------------------------------

    async def _decode_loop(self, track) -> None:  # noqa: ANN001
        first = True
        while not self._stopped:
            try:
                frame = await track.recv()
            except Exception as exc:
                # aiortc raises MediaStreamError on normal track end/teardown.
                # Any other exception type is unexpected — log at WARNING so it's visible.
                is_expected = type(exc).__name__ in ("MediaStreamError", "ConnectionError")
                logger.log(
                    logging.INFO if is_expected else logging.WARNING,
                    "Video consumer: track ended (%s)",
                    type(exc).__name__,
                    exc_info=not is_expected,
                )
                with self._frame_lock:
                    self._current_frame = None
                self._ended = True
                return
            arr = frame.to_ndarray(format="bgr24")
            if first:
                logger.info(
                    "Video consumer: first frame decoded (%dx%d)", arr.shape[1], arr.shape[0]
                )
                first = False
            with self._frame_lock:
                self._current_frame = arr

    # --- public read API -------------------------------------------------

    def get_frame(self, format: str = "numpy") -> Any:
        """Return the latest decoded frame, or ``None`` if none is available.

        ``format`` is ``"numpy"`` (BGR uint8, default), ``"bytes"`` (JPEG), or
        ``"pil"``. Returns ``None`` before the first frame and after the
        producer/track ends."""
        with self._frame_lock:
            arr = self._current_frame
        if arr is None:
            return None
        return self._format_frame(arr, format)

    def show(self, window_name: Optional[str] = None) -> None:
        """Open an OpenCV window and render frames until closed (blocking).

        Press ``q`` or ``Esc``, or close the window, to return. This is a
        convenience viewer for scripts and interactive sessions; it does not
        stop the stream (call :meth:`stop` when done). Not suitable for
        notebooks — use ``get_frame()`` with matplotlib there instead.

        Requires ``cyberwave[camera]`` (opencv-python).
        """
        import cv2

        window = window_name or f"cyberwave: {self._twin_uuid}"
        window_open = False
        try:
            while not self._ended:
                frame = self.get_frame()
                if frame is not None:
                    cv2.imshow(window, frame)
                    window_open = True
                # ~30 Hz UI pump; also catches the quit key.
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                # getWindowProperty on a never-created window raises "NULL
                # window" on macOS/Cocoa, so only query once it exists.
                if window_open and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if window_open:
                cv2.destroyWindow(window)
                # On macOS/Cocoa destroyWindow() only schedules teardown — the
                # native window is actually removed by the HighGUI event loop,
                # which advances on waitKey(). Without these pumps the window
                # lingers as a ghost after q/Esc/Ctrl-C and looks "unclosable".
                for _ in range(4):
                    cv2.waitKey(1)

    # --- teardown --------------------------------------------------------

    def stop(self) -> None:
        """Idempotently close this consumer. Affects only this peer connection;
        the producer and other consumers are untouched."""
        if self._stopped:
            return
        self._stopped = True
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._teardown(), loop).result(timeout=5)
            except Exception:
                logger.debug("teardown error (ignored)", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _teardown(self) -> None:
        try:
            self._mqtt.unsubscribe(self._answer_topic, subscriber_key=self._subscriber_key)
        except Exception:
            logger.debug("unsubscribe error (ignored)", exc_info=True)
        if self._decode_task is not None:
            self._decode_task.cancel()
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                logger.debug("pc.close error (ignored)", exc_info=True)
        # Let aiortc/aioice process the cancellations they just scheduled so we
        # don't stop the loop out from under pending internal tasks ("Task was
        # destroyed but it is pending").
        pending = [
            t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()
        ]
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def __enter__(self) -> "IncomingVideoStream":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except BaseException:
            pass
