"""Simulation (MuJoCo) camera implementation for Cyberwave SDK.

Provides multi-camera WebRTC streaming from MuJoCo simulations.  Cameras are
auto-discovered from the MuJoCo model, rendered offscreen each simulation step,
and streamed per-camera via WebRTC to the Cyberwave frontend.

Each camera gets its own :class:`SimCameraStreamer` (a :class:`BaseVideoStreamer`
subclass) running in a background thread.  The simulation loop only needs to
call :meth:`MujocoMultiCameraStreamer.capture` after every ``mujoco.mj_step``.

Usage::

    from cyberwave.sensor.camera_sim import MujocoMultiCameraStreamer
    from cyberwave import Cyberwave
    import mujoco

    client = Cyberwave(api_key="...", base_url="...")
    client.mqtt.connect()

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data  = mujoco.MjData(model)

    mgr = MujocoMultiCameraStreamer()
    mgr.start(twin_uuid="...", client=client.mqtt, model=model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            mgr.capture(model, data)   # renders + streams + optional preview
            viewer.sync()

    mgr.stop()
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import os
import queue as _queue_module
import struct
import subprocess
import sys
import threading
import time
import uuid as _uuid_module
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from av import VideoFrame as _AvVideoFrame
import mujoco

# Path to the preview subprocess entry point.
# Spawning a separate process keeps cv2's Qt5 GUI isolated from the simulation's
# GLFW/OpenGL context (same-process Qt5 + GLFW causes fatal X11 errors on some
# Linux drivers).  The script is a proper module, not an embedded string, so it
# can be linted, type-checked, and tested independently.
_PREVIEW_SCRIPT_PATH: Path = Path(__file__).parent / "_camera_preview.py"


from . import BaseVideoStreamer, BaseVideoTrack

if TYPE_CHECKING:
    from cyberwave.mqtt import CyberwaveMQTTClient
    from cyberwave.utils import TimeReference

logger = logging.getLogger(__name__)


# =============================================================================
# Thread-safe frame buffer
# =============================================================================


class ThreadSafeFrameBuffer:
    """Thread-safe single-slot frame buffer with FPS throttling.

    Holds only the **latest** frame so the consumer (WebRTC thread) always
    gets the most recent image without queue build-up.  FPS throttling
    prevents the simulation from writing faster than the stream target rate.

    Args:
        fps: Target frame rate used for write throttling.
    """

    def __init__(self, fps: float = 15.0) -> None:
        self.fps = fps
        self._frame_interval = 1.0 / max(fps, 1.0)
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._last_write_time: float = 0.0

    def add_frame(self, frame: np.ndarray) -> bool:
        """Store *frame* (RGB uint8 H×W×3) if enough time has elapsed.

        Uses ``np.copyto`` into a pre-allocated buffer to avoid allocation
        in the simulation hot-path after the first frame.

        Returns:
            ``True`` if the frame was stored, ``False`` if throttled.
        """
        now = time.monotonic()
        if now - self._last_write_time < self._frame_interval:
            return False
        with self._lock:
            if self._latest is None or self._latest.shape != frame.shape:
                self._latest = frame.copy()
            else:
                np.copyto(self._latest, frame)
            self._last_write_time = now
        return True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, or *None* if no frame yet."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()


# =============================================================================
# aiortc VideoStreamTrack backed by a frame buffer
# =============================================================================


class SimVideoTrack(BaseVideoTrack):
    """aiortc ``VideoStreamTrack`` that streams frames from a :class:`ThreadSafeFrameBuffer`.

    The simulation thread writes RGB uint8 frames into the buffer; this track
    reads the latest frame each :meth:`recv` call and encodes it as YUV420p for
    H.264 transport via WebRTC.

    Args:
        frame_buffer: Shared buffer populated by the simulation thread.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Target streaming FPS (controls ``time_base``).
        time_reference: Optional Cyberwave time reference for sync frames.
    """

    def __init__(
        self,
        frame_buffer: ThreadSafeFrameBuffer,
        width: int = 320,
        height: int = 240,
        fps: int = 15,
        time_reference: Optional["TimeReference"] = None,
    ) -> None:
        super().__init__()
        self.frame_buffer = frame_buffer
        self.width = width
        self.height = height
        self.fps = fps
        self.time_reference = time_reference
        self._last_recv_time: Optional[float] = None
        # Solid-blue placeholder emitted before the sim produces any frames
        self._placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        self._placeholder[..., 2] = 128

    # --- BaseVideoTrack interface ---

    def get_stream_attributes(self) -> dict:
        return {
            "camera_type": "mujoco_sim",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    async def recv(self):
        """Produce the next ``VideoFrame`` for WebRTC transport."""
        # Rate-limit to target FPS
        now = time.monotonic()
        if self._last_recv_time is not None:
            wait = max(0.0, (1.0 / self.fps) - (now - self._last_recv_time))
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_recv_time = time.monotonic()

        # Read latest rendered frame
        frame = self.frame_buffer.get_latest_frame()
        if frame is None:
            frame = self._placeholder

        # Capture timestamps for sync frame
        if self.time_reference is not None:
            timestamp, timestamp_monotonic = self.time_reference.update()
        else:
            timestamp = time.time()
            timestamp_monotonic = time.monotonic()

        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
            self.frame_0_timestamp_monotonic = timestamp_monotonic

        # MuJoCo renders RGB; encode to YUV420p for H.264
        arr = np.ascontiguousarray(frame)
        video_frame = _AvVideoFrame.from_ndarray(arr, format="rgb24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, self.fps)

        self._capture_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
        self.frame_count += 1

        return video_frame

    def close(self) -> None:
        """No external resources to release."""
        pass


# =============================================================================
# Single-camera WebRTC streamer
# =============================================================================


class SimCameraStreamer(BaseVideoStreamer):
    """WebRTC streamer for a single MuJoCo simulation camera.

    Subclasses :class:`BaseVideoStreamer` — all WebRTC signaling, MQTT
    negotiation, auto-reconnect, and health-check logic is inherited.
    Only :meth:`initialize_track` is overridden to return a
    :class:`SimVideoTrack` backed by the shared frame buffer.

    When a twin has multiple cameras, pass a distinct *camera_name* for each
    instance.  This value is forwarded in the WebRTC offer ``sensor`` field so
    the backend and frontend can route each stream to the correct viewer.

    Args:
        client: Connected :class:`CyberwaveMQTTClient` instance.
        frame_buffer: Buffer populated by the simulation thread.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Streaming FPS.
        twin_uuid: UUID of the digital twin.
        camera_name: Sensor identifier used for multi-camera routing.
        turn_servers: Optional TURN server list.
        time_reference: Optional shared time reference.
        auto_reconnect: Restart the stream on disconnection (default: True).
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        frame_buffer: ThreadSafeFrameBuffer,
        width: int = 320,
        height: int = 240,
        fps: int = 15,
        twin_uuid: Optional[str] = None,
        camera_name: Optional[str] = None,
        turn_servers: Optional[list] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
        enable_health_check: bool = True,
    ) -> None:
        super().__init__(
            client=client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
            camera_name=camera_name,
            enable_health_check=enable_health_check,
        )
        self.frame_buffer = frame_buffer
        self.width = width
        self.height = height
        self.fps = fps

    def initialize_track(self) -> SimVideoTrack:
        """Create and return a :class:`SimVideoTrack` backed by this streamer's buffer."""
        track = SimVideoTrack(
            frame_buffer=self.frame_buffer,
            width=self.width,
            height=self.height,
            fps=self.fps,
            time_reference=self.time_reference,
        )
        self.streamer = track
        return track


# =============================================================================
# Internal per-camera slot
# =============================================================================


class _CameraSlot:
    """Internal: per-camera state managed by :class:`MujocoMultiCameraStreamer`."""

    __slots__ = ("cam_id", "cam_name", "sensor_id", "renderer", "buffer",
                 "streamer", "thread", "stop_event",
                 "_preview_proc", "_preview_queue", "_preview_sender_thread")

    def __init__(self, cam_id: int, cam_name: str, sensor_id: str) -> None:
        self.cam_id: int = cam_id
        """MuJoCo camera index (``mujoco.mj_name2id`` result)."""
        self.cam_name: str = cam_name
        """Full camera name as stored in the MJCF model."""
        self.sensor_id: str = sensor_id
        """Short sensor identifier used as the WebRTC camera name and preview window title."""
        self.renderer = None
        """``mujoco.Renderer`` instance for offscreen rendering; created in :meth:`MujocoMultiCameraStreamer.start`."""
        self.buffer: Optional[ThreadSafeFrameBuffer] = None
        """Shared frame buffer: the sim thread writes frames, the WebRTC thread reads them."""
        self.streamer: Optional[SimCameraStreamer] = None
        """WebRTC streamer for this camera; runs in its own background thread."""
        self.thread: Optional[threading.Thread] = None
        """Background daemon thread running the async WebRTC event loop."""
        self.stop_event: Optional[threading.Event] = None
        """Set by :meth:`MujocoMultiCameraStreamer.stop` to signal the streaming thread to exit."""
        self._preview_proc = None
        """``subprocess.Popen`` handle for the cv2 preview child process (stdin=PIPE)."""
        self._preview_queue: Optional[_queue_module.Queue] = None
        """Single-slot queue for passing raw RGB frames from the sim loop to the preview sender thread."""
        self._preview_sender_thread: Optional[threading.Thread] = None
        """Daemon thread that reads from ``_preview_queue`` and writes frames to ``_preview_proc.stdin``."""


# =============================================================================
# Multi-camera manager
# =============================================================================


class MujocoMultiCameraStreamer:
    """Manages WebRTC streaming of **all** cameras in a MuJoCo model.

    Cameras are auto-discovered from ``model.ncam`` via :meth:`discover`.
    For each camera one :class:`SimCameraStreamer` is started in a dedicated
    background thread.  The simulation loop must call :meth:`capture` after
    every ``mujoco.mj_step`` to push rendered frames into each camera's
    :class:`ThreadSafeFrameBuffer`.

    By default an OpenCV preview window is opened per camera (using
    ``cv2.imshow`` on the calling/main thread, exactly as the ur7 demo does).
    Set ``CAMERA_NO_PREVIEW=1`` or pass ``show_previews=False`` to suppress.

    Args:
        show_previews: Show OpenCV preview windows for each camera.
            Overridable by the ``CAMERA_NO_PREVIEW=1`` environment variable.

    Environment variables::

        CAMERA_NO_PREVIEW   Disable OpenCV windows (1=disable, default: 0)
    """

    def __init__(self, show_previews: bool = True) -> None:
        self._show_previews: bool = show_previews and not bool(
            int(os.environ.get("CAMERA_NO_PREVIEW", "0"))
        )
        self._slots: List[_CameraSlot] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, model) -> List[Dict]:
        """Return metadata for every named camera in *model*.

        Args:
            model: ``mujoco.MjModel`` instance.

        Returns:
            List of dicts, one per camera::

                {
                    "cam_id":       int,          # MuJoCo camera index
                    "cam_name":     str,          # full camera name from XML
                    "sensor_id":    str,          # fragment after last "__" separator
                                                  # used as WebRTC ``sensor`` / ``camera_name``
                    "twin_uuid_prefix": str|None, # UUID prefix before "__", or None
                }

        The ``sensor_id`` extraction follows the Cyberwave naming convention
        ``{twin_uuid_prefix}__{sensor_id}`` produced by the universal schema
        exporter.  When no ``__`` is present the full name is used and
        ``twin_uuid_prefix`` is ``None``.
        """
        cameras: List[Dict] = []
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if not name:
                name = f"camera_{i}"
            if "__" in name:
                twin_uuid_prefix, sensor_id = name.split("__", 1)
                # Camera names encode UUIDs without dashes (e.g. "982a0751dbaf43ee…").
                # Normalise to standard 8-4-4-4-12 format so MQTT topics are correct.
                if len(twin_uuid_prefix) == 32 and "-" not in twin_uuid_prefix:
                    try:
                        twin_uuid_prefix = str(_uuid_module.UUID(twin_uuid_prefix))
                    except ValueError:
                        pass
            else:
                twin_uuid_prefix, sensor_id = None, name
            cameras.append({
                "cam_id": i,
                "cam_name": name,
                "sensor_id": sensor_id,
                "twin_uuid_prefix": twin_uuid_prefix,
            })
            logger.info(
                "Discovered camera %d: name=%r  sensor_id=%r  twin_uuid_prefix=%r",
                i, name, sensor_id, twin_uuid_prefix,
            )
        return cameras

    def start(
        self,
        twin_uuid: Optional[str],
        client: "CyberwaveMQTTClient",
        model,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        turn_servers: Optional[list] = None,
        time_reference: Optional["TimeReference"] = None,
        schema_cameras: Optional[List[Dict]] = None,
    ) -> None:
        """Discover cameras, create renderers, and launch one streaming thread per camera.

        This method **must** be called from the simulation thread (or at least
        the same thread that will call :meth:`capture`), because
        ``mujoco.Renderer`` objects are created here and used in :meth:`capture`.

        Args:
            twin_uuid: Fallback twin UUID used for cameras whose name does not
                encode a twin UUID prefix (i.e. cameras that don't follow the
                ``{twin_uuid}__{sensor_id}`` convention).  For multi-twin scenes
                each camera's name already contains the owning twin's UUID, so
                this argument acts purely as a fallback and may be ``None``.
            client: Connected :class:`CyberwaveMQTTClient` instance.
            model: ``mujoco.MjModel`` (cameras discovered + renderers created here).
            width: Global fallback render width in pixels (default: 320).  Per-camera
                values from *schema_cameras* take precedence.
            height: Global fallback render height in pixels (default: 240).
            fps: Global fallback streaming FPS (default: 15).
            turn_servers: Optional TURN server list forwarded to each streamer.
            time_reference: Optional shared time reference.
            schema_cameras: Per-camera metadata from
                :func:`~cyberwave.sensor.config.cameras_from_schema`.  When
                provided, each camera's ``width``, ``height``, and ``fps`` are
                looked up here by name; the global fallbacks are used for any
                camera not in the list.
        """
        if self._slots:
            raise RuntimeError(
                "MujocoMultiCameraStreamer.start() was called twice without stop(). "
                "Call stop() first to release existing cameras."
            )

        global_width = width or 320
        global_height = height or 240
        global_fps = fps or 15

        # Build a lookup from full camera name → schema metadata.
        schema_map: Dict[str, Dict] = {sc["name"]: sc for sc in (schema_cameras or [])}

        cameras = self.discover(model)
        if not cameras:
            logger.warning("No cameras found in MuJoCo model — nothing to stream")
            return

        from .manager import run_streamer_in_background

        for cam_info in cameras:
            # Use the twin UUID embedded in the camera name when available;
            # fall back to the caller-supplied twin_uuid for cameras without one.
            cam_twin_uuid = cam_info["twin_uuid_prefix"] or twin_uuid
            if not cam_twin_uuid:
                logger.warning(
                    "Camera %r has no twin UUID — skipping (pass fallback_twin_uuid "
                    "or use Cyberwave-exported camera names with UUID prefix)",
                    cam_info["cam_name"],
                )
                continue

            # Per-camera dimensions from schema, falling back to global values.
            sc = schema_map.get(cam_info["cam_name"], {})
            cam_width = sc.get("width") or global_width
            cam_height = sc.get("height") or global_height
            cam_fps = sc.get("fps") or global_fps

            slot = _CameraSlot(
                cam_id=cam_info["cam_id"],
                cam_name=cam_info["cam_name"],
                sensor_id=cam_info["sensor_id"],
            )

            # Offscreen renderer — must stay on the sim thread
            slot.renderer = mujoco.Renderer(model, height=cam_height, width=cam_width)

            # Shared buffer: sim thread writes, WebRTC thread reads
            slot.buffer = ThreadSafeFrameBuffer(fps=cam_fps)

            # One WebRTC session per camera.
            # Health-check is owned by the caller (CyberwaveSimStreaming creates a
            # single EdgeHealthCheck) — disable it here to avoid N+1 publishers
            # writing to the same edge_health MQTT topic.
            slot.streamer = SimCameraStreamer(
                client=client,
                frame_buffer=slot.buffer,
                width=cam_width,
                height=cam_height,
                fps=cam_fps,
                twin_uuid=cam_twin_uuid,
                camera_name=slot.sensor_id,  # sensor_id only, not full cam_name with UUID prefix
                turn_servers=turn_servers,
                time_reference=time_reference,
                auto_reconnect=True,
                enable_health_check=False,
            )
            logger.info(
                "Camera %r → twin_uuid=%r (%dx%d@%dfps, from %s)",
                slot.cam_name,
                cam_twin_uuid,
                cam_width, cam_height, cam_fps,
                "name prefix" if cam_info["twin_uuid_prefix"] else "fallback",
            )

            slot.stop_event = threading.Event()
            slot.thread = run_streamer_in_background(
                streamer=slot.streamer,
                stop_event=slot.stop_event,
                thread_name=f"sim-cam-{slot.sensor_id}",
            )
            self._slots.append(slot)
            logger.info("Started streaming thread for camera %r", slot.cam_name)

        # Launch a preview subprocess per camera via subprocess.Popen.
        # We exec a fresh Python interpreter (close_fds=True by default) so the
        # child has NO inherited X11 connections and Qt5/tkinter works cleanly.
        # Frames are sent over stdin as length-prefixed raw RGB bytes.
        #
        # Architecture: the sim loop puts raw frames into a Queue(maxsize=1).
        # A dedicated sender thread blocks on the queue and writes complete
        # frames to proc.stdin (blocking IO).  This decouples the sim loop
        # from the pipe write — the sim never blocks, and the preview always
        # receives intact frames (no partial writes / MemoryError).
        if self._show_previews:
            for slot in self._slots:
                env = os.environ.copy()
                proc = subprocess.Popen(
                    [sys.executable, str(_PREVIEW_SCRIPT_PATH), slot.sensor_id],
                    stdin=subprocess.PIPE,
                    close_fds=True,
                    env=env,
                )
                slot._preview_proc = proc
                slot._preview_queue = _queue_module.Queue(maxsize=1)

                t = threading.Thread(
                    target=self._preview_sender,
                    args=(slot,),
                    name=f"preview-sender-{slot.sensor_id}",
                    daemon=True,
                )
                t.start()
                slot._preview_sender_thread = t
                logger.info("Spawned preview pid=%d for camera %r", proc.pid, slot.sensor_id)

    def capture(self, model, data) -> None:
        """Render each camera offscreen and push frames into their buffers.

        Must be called from the **simulation thread** (same thread as :meth:`start`)
        after every ``mujoco.mj_step`` call.

        Preview windows are shown via ``cv2.imshow`` + ``cv2.waitKey(1)`` on
        the calling thread — the same pattern used by the ur7 conveyor demo.

        Args:
            model: ``mujoco.MjModel`` instance.
            data: ``mujoco.MjData`` instance.
        """
        for slot in self._slots:
            if slot.renderer is None or slot.buffer is None:
                continue
            try:
                slot.renderer.update_scene(data, camera=slot.cam_id)
                frame_rgb: np.ndarray = slot.renderer.render()  # H×W×3 uint8 RGB
                slot.buffer.add_frame(frame_rgb)

                # Hand frame to the dedicated sender thread via a single-slot
                # queue.  put_nowait drops the frame if the sender is still
                # writing the previous one — the sim loop is never blocked.
                if self._show_previews and slot._preview_queue is not None:
                    try:
                        slot._preview_queue.put_nowait(frame_rgb)
                    except _queue_module.Full:
                        pass  # sender busy — drop this frame
            except Exception:
                logger.exception("Error capturing camera %r", slot.cam_name)

    # ------------------------------------------------------------------
    # Preview sender thread
    # ------------------------------------------------------------------

    @staticmethod
    def _preview_sender(slot: "_CameraSlot") -> None:  # noqa: F821
        """Dedicated thread: reads frames from *slot._preview_queue* and writes
        them to *slot._preview_proc.stdin* using **blocking** IO.

        Each frame is sent as::

            4 bytes big-endian uint32  height
            4 bytes big-endian uint32  width
            4 bytes big-endian uint32  nbytes (= h*w*3)
            nbytes bytes               raw RGB

        Using a thread (not non-blocking writes) guarantees every frame
        arrives complete — no partial writes / MemoryError in the child.
        """
        q = slot._preview_queue
        proc = slot._preview_proc
        assert q is not None and proc is not None

        while True:
            try:
                item = q.get(timeout=1.0)
            except _queue_module.Empty:
                # Check if sim stopped or process died
                if proc.poll() is not None:
                    break
                continue

            if item is None:
                break  # stop sentinel

            if proc.poll() is not None:
                break  # subprocess exited

            try:
                frame_rgb = item
                frame_height, frame_width = frame_rgb.shape[:2]
                raw = frame_rgb.tobytes()
                hdr = struct.pack(">III", frame_height, frame_width, len(raw))
                if proc.stdin is not None:
                    proc.stdin.write(hdr + raw)
                    proc.stdin.flush()
            except (BrokenPipeError, OSError):
                break  # preview window closed — exit quietly

    def stop(self) -> None:
        """Signal all streaming threads to stop and release resources.

        Closes renderers (on the calling thread = sim thread) and joins the
        background streaming threads.
        """
        # Signal each streaming thread to stop via its per-slot event.
        for slot in self._slots:
            if slot.stop_event is not None:
                slot.stop_event.set()

        # Shut down preview sender threads and subprocesses
        for slot in self._slots:
            if slot._preview_queue is not None:
                try:
                    slot._preview_queue.put_nowait(None)  # stop sentinel
                except _queue_module.Full:
                    pass
            if slot._preview_sender_thread is not None:
                slot._preview_sender_thread.join(timeout=3.0)
            if slot._preview_proc is not None:
                proc = slot._preview_proc
                if proc.stdin:
                    proc.stdin.close()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()

        # Close offscreen renderers (must be on sim thread)
        for slot in self._slots:
            if slot.renderer is not None:
                slot.renderer.close()
                slot.renderer = None

        # Join streaming daemon threads
        for slot in self._slots:
            if slot.thread is not None and slot.thread.is_alive():
                slot.thread.join(timeout=5.0)

        self._slots.clear()
        logger.info("MujocoMultiCameraStreamer stopped")



# =============================================================================
# High-level opt-in streaming helper (demo / simulation service entrypoint)
# =============================================================================


class CyberwaveSimStreaming:
    """Opt-in, env-var-driven MuJoCo camera streaming for Cyberwave-exported scenes.

    This class wraps the Cyberwave MQTT client, :class:`MujocoMultiCameraStreamer`,
    and :class:`~cyberwave.edge.health.EdgeHealthCheck` behind a single lifecycle
    object.  Demos and the simulation service call :meth:`from_env` once; the
    returned object (or ``None``) is passed to the simulation loop.

    Streaming is **opt-in**: :meth:`from_env` returns ``None`` when
    ``CYBERWAVE_STREAM_CAMERAS`` is not ``"1"``, so callers need only a single
    ``if streaming:`` guard.

    Typical demo usage::

        from cyberwave.sensor.camera_sim import CyberwaveSimStreaming

        streaming = CyberwaveSimStreaming.from_env("out/universal_schema.json")
        # streaming is None when CYBERWAVE_STREAM_CAMERAS != 1

        load_model_and_run(xml_path=xml_path, ..., camera_manager=streaming)

    Camera-to-twin routing uses the ``{twin_uuid_hex}__{sensor_id}`` naming
    convention embedded in every Cyberwave-exported camera name.  The optional
    ``twin_uuid`` is only needed as a fallback for non-standard MJCF scenes
    whose camera names do not embed a UUID prefix.

    Per-camera metadata (dimensions, fps) comes from ``universal_schema.json``
    sensors.  All environment variable parsing is delegated to
    :class:`~cyberwave.sensor.config.SimStreamingConfig`.

    Args:
        client: Connected :class:`CyberwaveMQTTClient` instance.
        twin_uuid: Fallback twin UUID for cameras without an embedded UUID prefix.
        schema_cameras: Per-camera metadata from
            :func:`~cyberwave.sensor.config.cameras_from_schema`.
        show_previews: Open a cv2 preview subprocess per camera.
        width: Global fallback render width in pixels.
        height: Global fallback render height in pixels.
        fps: Global fallback target streaming frame rate.
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        twin_uuid: Optional[str] = None,
        schema_cameras: Optional[List[Dict]] = None,
        show_previews: bool = True,
        width: int = 320,
        height: int = 240,
        fps: int = 15,
    ) -> None:
        self._client = client
        self._twin_uuid = twin_uuid
        self._schema_cameras: List[Dict] = schema_cameras or []
        self._show_previews = show_previews
        self._width = width
        self._height = height
        self._fps = fps
        self._multi_streamer: Optional[MujocoMultiCameraStreamer] = None
        self._health_check = None  # EdgeHealthCheck, created in start()

        # Derive all unique twin UUIDs from schema camera names for health-check
        # heartbeats (each camera may belong to a different twin).
        seen: set = set()
        all_twins: List[str] = []
        for cam in self._schema_cameras:
            name = cam.get("name", "")
            if "__" in name:
                hex_id = name.split("__", 1)[0]
                try:
                    normalized = str(_uuid_module.UUID(hex_id))
                    if normalized not in seen:
                        seen.add(normalized)
                        all_twins.append(normalized)
                except (ValueError, AttributeError):
                    pass
        if twin_uuid and twin_uuid not in seen:
            all_twins.insert(0, twin_uuid)
        self._all_twin_uuids: List[str] = all_twins or ([twin_uuid] if twin_uuid else [])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        universal_schema_path=None,
        fallback_twin_uuid: Optional[str] = None,
    ) -> Optional["CyberwaveSimStreaming"]:
        """Build an instance from environment variables and ``universal_schema.json``.

        Returns ``None`` when streaming is disabled (``CYBERWAVE_STREAM_CAMERAS``
        not set to ``"1"``) or when the Cyberwave API key is missing.

        Camera metadata (per-camera width/height/fps and twin-UUID routing) is
        read from *universal_schema_path*.  For scenes where camera names
        already embed the owning twin's UUID, *fallback_twin_uuid* is optional.

        Args:
            universal_schema_path: Path to the ``out/universal_schema.json``
                written by the scene exporter.
            fallback_twin_uuid: Twin UUID for cameras that do not embed a UUID
                prefix in their name.
        """
        from .config import SimStreamingConfig

        cfg = SimStreamingConfig.from_env(
            universal_schema_path=universal_schema_path,
            fallback_twin_uuid=fallback_twin_uuid,
        )

        if not cfg.enabled:
            return None

        # Cyberwave() reads CYBERWAVE_API_KEY and CYBERWAVE_BASE_URL from env
        # automatically — no need to re-parse them here.
        try:
            from cyberwave import Cyberwave  # type: ignore[import]
            client_obj = Cyberwave()
        except ValueError as exc:
            # Cyberwave() raises ValueError when CYBERWAVE_API_KEY is missing.
            logger.warning("Streaming disabled: %s", exc)
            return None
        except ImportError as exc:
            logger.warning("cyberwave SDK not available (%s) — streaming disabled.", exc)
            return None

        client_obj.mqtt.connect()

        logger.info(
            "CyberwaveSimStreaming: cameras=%d  base_url=%s  preview=%s  fallback=%dx%d@%dfps",
            len(cfg.cameras),
            client_obj.config.base_url,
            cfg.show_previews,
            cfg.width,
            cfg.height,
            cfg.fps,
        )

        return cls(
            client=client_obj.mqtt,
            twin_uuid=cfg.twin_uuid,
            schema_cameras=cfg.cameras or None,
            show_previews=cfg.show_previews,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        )

    # ------------------------------------------------------------------
    # Lifecycle (mirrors MujocoMultiCameraStreamer API so it can be passed
    # directly as `camera_manager` to `load_model_and_run`)
    # ------------------------------------------------------------------

    def start(self, model, **kwargs) -> None:
        """Discover cameras, start streaming, and start the health check.

        Must be called from the simulation thread (same thread as
        :meth:`capture`) because ``mujoco.Renderer`` objects are created here.

        Args:
            model: ``mujoco.MjModel`` instance.
            **kwargs: Extra kwargs forwarded to :meth:`MujocoMultiCameraStreamer.start`
                (override ``width``, ``height``, or ``fps`` if needed).
        """
        # Start EdgeHealthCheck first so the frontend sees the edge as connected
        # before the WebRTC offer arrives.
        try:
            from ..edge.health import EdgeHealthCheck  # type: ignore[import]
            self._health_check = EdgeHealthCheck(
                mqtt_client=self._client,
                twin_uuids=self._all_twin_uuids,
                edge_id=self._twin_uuid,
            )
            self._health_check.start()
            logger.info("EdgeHealthCheck started for twin_uuids=%s", self._all_twin_uuids)
        except Exception as exc:
            logger.warning("Could not start EdgeHealthCheck: %s", exc)

        # Build and start the per-camera streaming manager.
        # Explicit width/height/fps values from config take precedence;
        # callers can still override via **kwargs.
        stream_kwargs = dict(width=self._width, height=self._height, fps=self._fps)
        stream_kwargs.update(kwargs)

        self._multi_streamer = MujocoMultiCameraStreamer(show_previews=self._show_previews)
        self._multi_streamer.start(
            twin_uuid=self._twin_uuid,
            client=self._client,
            model=model,
            schema_cameras=self._schema_cameras or None,
            **stream_kwargs,
        )

    def capture(self, model, data) -> None:
        """Render all cameras and push frames into their WebRTC buffers.

        Must be called from the **simulation thread** after every ``mujoco.mj_step``.
        """
        if self._multi_streamer is not None:
            self._multi_streamer.capture(model, data)

    def stop(self) -> None:
        """Stop streaming and release all resources."""
        if self._multi_streamer is not None:
            try:
                self._multi_streamer.stop()
            except Exception as exc:
                logger.warning("MujocoMultiCameraStreamer.stop() failed: %s", exc)
            self._multi_streamer = None

        if self._health_check is not None:
            try:
                self._health_check.stop()
            except Exception as exc:
                logger.warning("EdgeHealthCheck.stop() failed: %s", exc)
            self._health_check = None

