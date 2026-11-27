"""Pre-encoded H.264 from a GStreamer pipeline, streamed over WebRTC without re-encoding.

Use this for sources that already deliver H.264, so nothing is decoded or
re-encoded on the way to WebRTC:

- a stream already on the network: RTSP camera or NVR, RTP over UDP, MPEG-TS
  over UDP/TCP. :func:`~cyberwave.sensor.gst_sources.h264_source_for_url`
  builds the fragment from a URL.
- H.264 the caller already holds, e.g. the frames a robot's own WebRTC session
  delivers: pass ``source="appsrc"`` and call :meth:`GstH264Source.push`.
- any ``gst-launch`` fragment that produces H.264: a UVC camera with an H.264
  mode (``v4l2src device=/dev/video0 ! video/x-h264``), a hardware encoder
  (``... ! nvv4l2h264enc``).

The SDK appends the fixed tail::

    <source> ! h264parse config-interval=-1
             ! video/x-h264,stream-format=byte-stream,alignment=au
             ! appsink

so every buffer that reaches :class:`~cyberwave.sensor.camera_h264.H264PacketVideoTrack`
is one Annex-B access unit with SPS/PPS repeated before each keyframe.

**Frame sync.** aiortc only packetizes pre-encoded frames, so the video sync
data the SDK writes into frames it encodes itself never reaches them. When the
``cyberwave_video_sync`` extension is installed, a buffer probe on
``h264parse``'s src pad writes the same video sync data into each frame,
stamped from the buffer's capture time. Pass ``require_sync=True`` to fail
instead of streaming without it.

Requires PyGObject and the system GStreamer libraries (``cyberwave[gstreamer]``
plus ``gstreamer1.0-plugins-{base,good,bad}`` from the OS). ``gi`` is imported
only when a pipeline starts.

Example::

    streamer = GStreamerH264CameraStreamer(
        client.mqtt,
        source="rtspsrc location=rtsp://192.168.1.10/stream latency=0 ! rtph264depay",
        twin_uuid=twin.uuid,
        camera_name="color_camera",
    )
    await streamer.run_with_auto_reconnect()
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .camera_h264 import (
    FRAMING_ANNEXB,
    FRAMING_AUTO,
    H264PacketCameraStreamer,
    H264PacketVideoTrack,
    to_annex_b,
)
from .gst_sources import APPSRC_FRAGMENT, PUSH_SOURCE, redact_credentials

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient

logger = logging.getLogger(__name__)

#: Elements appended to the caller's source. ``alignment=au`` gives one buffer,
#: and therefore one video sync record, per frame.
PIPELINE_TAIL = (
    "h264parse name=cw_h264parse config-interval=-1 "
    "! video/x-h264,stream-format=byte-stream,alignment=au "
    "! appsink name=cw_appsink emit-signals=true sync=false max-buffers=1 drop=false"
)

#: With a decoded-frame callback the access units are teed: one branch to the
#: WebRTC track untouched, one decoded to BGR (latest frame only, never
#: back-pressuring the encoded branch).
PIPELINE_TAIL_WITH_DECODE = (
    "h264parse name=cw_h264parse config-interval=-1 "
    "! video/x-h264,stream-format=byte-stream,alignment=au "
    "! tee name=cw_tee "
    "cw_tee. ! queue "
    "! appsink name=cw_appsink emit-signals=true sync=false max-buffers=1 drop=false "
    "cw_tee. ! queue leaky=downstream max-size-buffers=2 "
    "! decodebin ! videoconvert ! video/x-raw,format=BGR "
    "! appsink name=cw_raw_appsink emit-signals=true sync=false max-buffers=1 drop=true"
)

_RESTART_BACKOFF_S = (1.0, 2.0, 5.0, 10.0)


class VideoSyncUnavailableError(RuntimeError):
    """``require_sync=True`` but the ``cyberwave_video_sync`` probe is not installed."""


def build_pipeline_description(source: str, *, decode: bool = False) -> str:
    """Full ``gst-launch`` description for an H.264 *source* fragment."""
    source = source.strip().rstrip("!").strip()
    if not source:
        raise ValueError("GStreamer source must not be empty")
    if source == PUSH_SOURCE:
        source = APPSRC_FRAGMENT
    return f"{source} ! {PIPELINE_TAIL_WITH_DECODE if decode else PIPELINE_TAIL}"


def _load_gst() -> Any:
    try:
        import gi  # type: ignore[import-untyped,import-not-found,unused-ignore]

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore[import-untyped,import-not-found,unused-ignore]
    except (ImportError, ValueError) as exc:
        raise ImportError(
            "GStreamer sources need PyGObject and the system GStreamer libraries. "
            "Install them with: pip install 'cyberwave[gstreamer]' "
            "(plus gstreamer1.0-plugins-base/-good/-bad from your OS packages)"
        ) from exc
    if not Gst.is_initialized():
        Gst.init(None)
    return Gst


class GstH264Source:
    """Run a GStreamer H.264 pipeline and hand out its access units one by one.

    :meth:`get_packet` is the ``get_packet`` callable that
    :class:`~cyberwave.sensor.camera_h264.H264PacketVideoTrack` polls: it
    returns each access unit exactly once, oldest first. A pipeline error or
    end-of-stream (an RTSP camera rebooting, a cable pulled) restarts the
    pipeline with backoff until :meth:`stop`.

    Args:
        source: ``gst-launch`` fragment producing H.264, without the tail
            (see :data:`PIPELINE_TAIL`), or ``"appsrc"`` to feed it with
            :meth:`push`.
        fps: Frame rate stamped into the video sync data's time base.
        track_id: Callable returning the current WebRTC track id, for the video sync data.
        queue_size: Access units buffered between the pipeline and the track.
        require_sync: Raise :class:`VideoSyncUnavailableError` from :meth:`start`
            when the ``cyberwave_video_sync`` probe is not installed.
        frame_callback: Optional ``callback(bgr_frame, frame_count)`` fed from
            a decoded branch (e.g. to publish frames to local workers). The
            WebRTC stream still forwards the original H.264.
    """

    def __init__(
        self,
        source: str,
        *,
        fps: int = 30,
        track_id: Callable[[], str | None] | None = None,
        queue_size: int = 30,
        require_sync: bool = False,
        frame_callback: Callable[[Any, int], Any] | None = None,
    ) -> None:
        self.description = build_pipeline_description(source, decode=frame_callback is not None)
        self.frame_callback = frame_callback
        self.decoded_frame_count = 0
        self.fps = fps
        self.track_id = track_id
        self.require_sync = require_sync
        self.width: int | None = None
        self.height: int | None = None
        self.keyframe_seen = False
        self.sync_attached = False
        self.restarts = 0
        self._queue: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=max(1, queue_size)
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline: Any = None
        self._appsrc: Any = None
        self._last_push_pts = -1
        # Per pipeline run: drop delta frames until that run's first keyframe.
        self._gate_open = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Build and play the pipeline; a supervisor thread keeps it running."""
        if self.running:
            return
        gst = _load_gst()
        self._stop.clear()
        self._build(gst)
        self._thread = threading.Thread(
            target=self._supervise, args=(gst,), daemon=True, name="gst-h264-source"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the pipeline and the supervisor. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._teardown()

    @property
    def accepts_push(self) -> bool:
        """True for an ``appsrc`` pipeline fed with :meth:`push`."""
        return "name=cw_appsrc" in self.description

    def push(
        self,
        data: bytes,
        *,
        capture_time: float | None = None,
        framing: str = FRAMING_AUTO,
    ) -> bool:
        """Feed H.264 into an ``appsrc`` pipeline. Safe from any thread, never blocks.

        Args:
            data: Annex-B or 4-byte AVCC H.264: a whole frame or any part of
                one (``h264parse`` re-aligns them).
            capture_time: When the frame was captured, as ``time.time()`` seconds
                (wall clock). It becomes the buffer timestamp, so the video sync
                data carries the camera's capture time instead of the arrival time.
                ``None`` stamps the frame now.
            framing: See :func:`~cyberwave.sensor.camera_h264.to_annex_b`.

        Returns:
            ``False`` when the frame was dropped: the pipeline is not running
            (not started yet, or restarting) or it refused the buffer.
        """
        if not self.accepts_push:
            raise RuntimeError("push() needs a GstH264Source built with source='appsrc'")
        appsrc, pipeline = self._appsrc, self._pipeline
        if appsrc is None or pipeline is None or not data:
            return False
        gst = _load_gst()
        buf = gst.Buffer.new_wrapped(to_annex_b(bytes(data), framing))
        buf.pts = self._push_pts(pipeline, capture_time, gst)
        return appsrc.emit("push-buffer", buf) == gst.FlowReturn.OK

    def _push_pts(self, pipeline: Any, capture_time: float | None, gst: Any) -> int:
        """Running time of *capture_time* on the pipeline clock (monotonic per run)."""
        clock = pipeline.get_clock()
        if clock is None:
            return gst.CLOCK_TIME_NONE
        now_running = clock.get_time() - pipeline.get_base_time()
        age_ns = 0 if capture_time is None else int((time.time() - capture_time) * 1e9)
        pts = max(0, now_running - max(0, age_ns), self._last_push_pts + 1)
        self._last_push_pts = pts
        return pts

    def get_packet(self) -> tuple[bytes, bool] | None:
        """Oldest unread ``(access_unit, is_keyframe)``, or ``None``."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    # -- pipeline -------------------------------------------------------------

    def _build(self, gst: Any) -> None:
        self._gate_open = False
        self._last_push_pts = -1
        pipeline = gst.parse_launch(self.description)
        parse = pipeline.get_by_name("cw_h264parse")
        self._attach_sync_probe(parse.get_static_pad("src"))
        sink = pipeline.get_by_name("cw_appsink")
        sink.connect("new-sample", self._on_sample, gst)
        raw_sink = pipeline.get_by_name("cw_raw_appsink")
        if raw_sink is not None:
            raw_sink.connect("new-sample", self._on_raw_sample, gst)
        if pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            pipeline.set_state(gst.State.NULL)
            raise RuntimeError(
                f"GStreamer pipeline failed to start: {redact_credentials(self.description)}"
            )
        self._pipeline = pipeline
        self._appsrc = pipeline.get_by_name("cw_appsrc")
        logger.info(
            "GStreamer H.264 source playing: %s", redact_credentials(self.description)
        )

    def _attach_sync_probe(self, pad: Any) -> None:
        sync_module: Any
        try:
            import cyberwave_video_sync as sync_module
        except ImportError:
            sync_module = None
        attach = getattr(sync_module, "attach_gst_probe", None)
        if attach is None:
            if self.require_sync:
                raise VideoSyncUnavailableError(
                    "cyberwave_video_sync with attach_gst_probe is required for "
                    "pre-encoded H.264 sources but is not installed"
                )
            logger.warning(
                "cyberwave_video_sync probe not installed; pre-encoded H.264 frames "
                "will stream without video sync data"
            )
            return
        attach(pad, track_id=self.track_id, fps=self.fps)
        self.sync_attached = True

    def _teardown(self) -> None:
        self._appsrc = None
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            gst = _load_gst()
            pipeline.set_state(gst.State.NULL)

    def _on_sample(self, sink: Any, gst: Any) -> Any:
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, info = buf.map(gst.MapFlags.READ)
        if not ok:
            return gst.FlowReturn.OK
        try:
            data = bytes(info.data)
        finally:
            buf.unmap(info)
        is_keyframe = not buf.has_flags(gst.BufferFlags.DELTA_UNIT)
        if self.width is None:
            self._read_caps(sample.get_caps())
        with self._lock:
            if is_keyframe:
                self._gate_open = True
                self.keyframe_seen = True
            elif not self._gate_open:
                # Nothing decodes before the first keyframe of this run.
                return gst.FlowReturn.OK
            self._queue.append((data, is_keyframe))
        return gst.FlowReturn.OK

    def _on_raw_sample(self, sink: Any, gst: Any) -> Any:
        sample = sink.emit("pull-sample")
        callback = self.frame_callback
        if sample is None or callback is None:
            return gst.FlowReturn.OK
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        if structure is None:
            return gst.FlowReturn.OK
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not (ok_w and ok_h):
            return gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, info = buf.map(gst.MapFlags.READ)
        if not ok:
            return gst.FlowReturn.OK
        try:
            import numpy as np

            # Rows may be padded to a 4-byte stride.
            stride = len(info.data) // height
            frame = (
                np.frombuffer(info.data, dtype=np.uint8)
                .reshape(height, stride)[:, : width * 3]
                .reshape(height, width, 3)
                .copy()
            )
        finally:
            buf.unmap(info)
        try:
            callback(frame, self.decoded_frame_count)
        except Exception:
            logger.debug("GStreamer frame callback failed", exc_info=True)
        self.decoded_frame_count += 1
        return gst.FlowReturn.OK

    def _read_caps(self, caps: Any) -> None:
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if ok_w and ok_h:
            self.width, self.height = width, height
        ok_f, num, den = structure.get_fraction("framerate")
        if ok_f and num > 0 and den > 0:
            self.fps = max(1, round(num / den))

    def _supervise(self, gst: Any) -> None:
        """Restart the pipeline on error or end-of-stream until stopped."""
        attempt = 0
        while not self._stop.is_set():
            pipeline = self._pipeline
            if pipeline is None:
                delay = _RESTART_BACKOFF_S[min(attempt, len(_RESTART_BACKOFF_S) - 1)]
                if self._stop.wait(delay):
                    return
                attempt += 1
                try:
                    self._build(gst)
                    self.restarts += 1
                except Exception as exc:
                    logger.warning("GStreamer pipeline restart failed: %s", exc)
                continue
            msg = pipeline.get_bus().timed_pop_filtered(
                200 * gst.MSECOND, gst.MessageType.ERROR | gst.MessageType.EOS
            )
            if msg is None:
                if self.keyframe_seen:
                    attempt = 0
                continue
            if msg.type == gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.warning("GStreamer pipeline error: %s (%s); restarting", err, debug)
            else:
                logger.warning("GStreamer pipeline reached end of stream; restarting")
            self._teardown()


class GStreamerH264CameraStreamer(H264PacketCameraStreamer):
    """Stream a GStreamer H.264 pipeline to Cyberwave over WebRTC, no re-encode.

    :meth:`start` starts the pipeline (once), waits for its first keyframe so
    the offer carries the real resolution, then connects. :meth:`close` stops
    the pipeline; :meth:`stop` only closes the WebRTC session, so a
    ``stop_video``/``start_video`` cycle does not reopen the camera.

    Args:
        client: The SDK MQTT client used for signalling.
        source: ``gst-launch`` fragment producing H.264 (see :class:`GstH264Source`).
        width, height: Fallback resolution until the pipeline reports caps.
        fps: Expected frame rate; replaced by the caps framerate when known.
        require_sync: See :class:`GstH264Source`.
        keyframe_timeout: Seconds :meth:`start` waits for the first keyframe.
        frame_callback: See :class:`GstH264Source`.
        **kwargs: Passed to :class:`~cyberwave.sensor.camera_h264.H264PacketCameraStreamer`.
    """

    def __init__(
        self,
        client: CyberwaveMQTTClient,
        *,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        require_sync: bool = False,
        keyframe_timeout: float = 15.0,
        queue_size: int = 30,
        frame_callback: Callable[[Any, int], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.source = GstH264Source(
            source,
            fps=fps,
            track_id=self._current_track_id,
            queue_size=queue_size,
            require_sync=require_sync,
            frame_callback=frame_callback,
        )
        self.keyframe_timeout = keyframe_timeout
        kwargs.setdefault("framing", FRAMING_ANNEXB)
        super().__init__(
            client=client,
            get_packet=self.source.get_packet,
            width=width,
            height=height,
            fps=fps,
            **kwargs,
        )

    def _current_track_id(self) -> str | None:
        track = getattr(self, "streamer", None)
        return getattr(track, "id", None) if track is not None else None

    def initialize_track(self) -> H264PacketVideoTrack:
        """Create the track with the resolution and rate the pipeline reports."""
        if self.source.width and self.source.height:
            self.width, self.height = self.source.width, self.source.height
        self.fps = self.source.fps or self.fps
        return super().initialize_track()

    async def wait_for_keyframe(self, timeout: float | None = None) -> bool:
        """Wait for the pipeline's first keyframe; ``False`` on timeout."""
        deadline = time.monotonic() + (self.keyframe_timeout if timeout is None else timeout)
        while not self.source.keyframe_seen:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True

    async def start(self, twin_uuid: str | None = None):
        """Start the pipeline if needed, wait for a keyframe, then connect."""
        if not self.source.running:
            await asyncio.to_thread(self.source.start)
        if not await self.wait_for_keyframe():
            raise RuntimeError(
                f"No H.264 keyframe from the GStreamer source within "
                f"{self.keyframe_timeout:.0f}s: {redact_credentials(self.source.description)}"
            )
        await super().start(twin_uuid)

    def push(self, data: bytes, *, capture_time: float | None = None) -> bool:
        """Feed H.264 to a ``source="appsrc"`` streamer; see :meth:`GstH264Source.push`."""
        return self.source.push(data, capture_time=capture_time)

    async def close(self) -> None:
        """Stop streaming and the pipeline."""
        await self.stop()
        await asyncio.to_thread(self.source.stop)
