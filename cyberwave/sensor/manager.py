"""Camera stream manager for running multiple camera streams with threading support.

Provides CameraStreamManager for teleoperation and edge scripts that need to run
camera streaming in background threads with run_with_auto_reconnect (command handling).

Also exposes :func:`run_streamer_in_background`, a low-level helper that runs any
pre-built :class:`~cyberwave.sensor.BaseVideoStreamer` in a daemon thread with its
own asyncio event loop — used by MuJoCo multi-camera streaming and other callers that
create their streamers before threading.
"""

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..client import Cyberwave
    from ..twin import CameraTwin, DepthCameraTwin
    from ..utils import TimeReference

logger = logging.getLogger(__name__)


# =============================================================================
# Shared low-level thread runner for pre-built streamers
# =============================================================================


def run_streamer_in_background(
    streamer: Any,
    stop_event: threading.Event,
    thread_name: str = "cam-streamer",
) -> threading.Thread:
    """Run *streamer* in a daemon thread with its own asyncio event loop.

    Handles MQTT readiness and bridges the threading *stop_event* to the
    asyncio stop event expected by
    :meth:`~cyberwave.sensor.BaseVideoStreamer.run_with_auto_reconnect`.

    This is the canonical thread runner for scenarios where the streamer is
    already built before the thread starts (e.g. MuJoCo simulation cameras).
    For scenarios where the streamer must be constructed inside the thread
    (e.g. :class:`CameraStreamManager`) use :func:`_run_streamer_in_thread`.

    Args:
        streamer: A pre-built :class:`~cyberwave.sensor.BaseVideoStreamer`.
        stop_event: ``threading.Event`` — set it to stop the stream.
        thread_name: Thread name for logging and debugging.

    Returns:
        The already-started daemon thread.
    """

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Stash loop on the thread object so external code can schedule
        # coroutines on it if needed (e.g. forceful stop via run_coroutine_threadsafe).
        threading.current_thread()._event_loop = loop  # type: ignore[attr-defined]
        try:
            loop.run_until_complete(_run_async())
        except Exception:
            logger.exception("Streamer thread error (%r)", thread_name)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _run_async() -> None:
        # 1. Ensure MQTT is connected
        if not streamer.client.connected:
            streamer.client.connect()
        deadline = time.time() + 10.0
        while not streamer.client.connected and time.time() < deadline:
            await asyncio.sleep(0.2)
        if not streamer.client.connected:
            logger.error(
                "MQTT connection timeout for streamer %r — aborting", thread_name
            )
            return

        # 2. Bridge threading stop_event → asyncio Event
        async_stop = asyncio.Event()

        async def _watch_stop() -> None:
            while not stop_event.is_set():
                await asyncio.sleep(0.3)
            async_stop.set()

        asyncio.create_task(_watch_stop())

        # 3. Run with auto-reconnect until stopped
        try:
            await streamer.run_with_auto_reconnect(stop_event=async_stop)
        except Exception:
            logger.exception("run_with_auto_reconnect error (%r)", thread_name)

    t = threading.Thread(target=_target, name=thread_name, daemon=True)
    t.start()
    return t


def _infer_config_from_twin(
    twin: Any,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer stream config from twin capabilities and type. Overrides merge on top."""
    from ..twin import DepthCameraTwin
    from .config import Resolution

    # Infer camera type from twin class (DepthCameraTwin -> realsense, else cv2)
    is_depth = isinstance(twin, DepthCameraTwin)
    camera_type = "realsense" if is_depth else "cv2"

    # Infer camera_name from capabilities.sensors (CameraTwin always has sensors)
    camera_name = None
    sensors = twin.capabilities.get("sensors", [])
    if sensors and isinstance(sensors[0], dict):
        camera_name = sensors[0].get("id", "default")

    config: Dict[str, Any] = {
        "twin": twin,
        "camera_id": 0,
        "camera_type": camera_type,
        "camera_resolution": Resolution.VGA,
        "camera_name": camera_name,
        "fps": 30,
        "enable_depth": is_depth,
        "depth_fps": 30,
        "depth_resolution": None,
        "depth_publish_interval": 30,
        "keyframe_interval": None,
    }
    if overrides:
        config.update(overrides)
    return config


def _run_streamer_in_thread(
    client: "Cyberwave",
    config: Dict[str, Any],
    stop_event: threading.Event,
    time_reference: "TimeReference",
    command_callback: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Run a single camera streamer in a thread with run_with_auto_reconnect."""
    from . import CV2CameraStreamer, RealSenseStreamer
    from .config import Resolution

    twin = config["twin"]
    twin_uuid = twin.uuid
    camera_id = config.get("camera_id", 0)
    camera_type = config.get("camera_type", "cv2")
    camera_resolution = config.get("camera_resolution")
    if camera_resolution is None:
        camera_resolution = Resolution.VGA
    elif isinstance(camera_resolution, (list, tuple)):
        camera_resolution = Resolution.from_size(
            camera_resolution[0], camera_resolution[1]
        ) or Resolution.closest(camera_resolution[0], camera_resolution[1])
    fps = config.get("fps", 30)
    enable_depth = config.get("enable_depth", False)
    depth_fps = config.get("depth_fps", 30)
    depth_resolution = config.get("depth_resolution")
    if depth_resolution is None:
        depth_resolution = camera_resolution
    elif isinstance(depth_resolution, (list, tuple)):
        depth_resolution = Resolution.from_size(
            depth_resolution[0], depth_resolution[1]
        ) or Resolution.closest(depth_resolution[0], depth_resolution[1])
    depth_publish_interval = config.get("depth_publish_interval", 30)
    camera_name = config.get("camera_name")
    fourcc = config.get("fourcc")
    keyframe_interval = config.get("keyframe_interval")

    async def _run():
        async_stop_event = asyncio.Event()

        if not client.mqtt.connected:
            client.mqtt.connect()
        max_wait = 10.0
        wait_start = time.time()
        while not client.mqtt.connected:
            if time.time() - wait_start > max_wait:
                raise RuntimeError(
                    "Failed to connect to MQTT broker - cannot send WebRTC offer"
                )
            await asyncio.sleep(0.1)

        camera_type_lower = camera_type.lower()
        if camera_type_lower == "cv2":
            streamer = CV2CameraStreamer(
                client=client.mqtt,
                camera_id=camera_id,
                fps=fps,
                resolution=camera_resolution,
                twin_uuid=twin_uuid,
                time_reference=time_reference,
                auto_reconnect=True,
                camera_name=camera_name,
                fourcc=fourcc,
                keyframe_interval=keyframe_interval,
            )
        elif camera_type_lower == "realsense":
            streamer = RealSenseStreamer(
                client=client.mqtt,
                color_fps=fps,
                depth_fps=depth_fps,
                color_resolution=camera_resolution,
                depth_resolution=depth_resolution,
                enable_depth=enable_depth,
                depth_publish_interval=depth_publish_interval,
                twin_uuid=twin_uuid,
                time_reference=time_reference,
                auto_reconnect=True,
                camera_name=camera_name,
            )
        else:
            raise ValueError(
                f"Unsupported camera type: {camera_type}. Use 'cv2' or 'realsense'."
            )

        async def monitor_stop():
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
            async_stop_event.set()

        monitor_task = asyncio.create_task(monitor_stop())

        # Wrap callback to inject camera_name for per-camera status tracking
        cam_name = camera_name or "default"

        def wrapped_callback(s: str, m: str) -> None:
            if command_callback:
                try:
                    command_callback(s, m, cam_name)
                except TypeError:
                    command_callback(s, m)

        try:
            await streamer.run_with_auto_reconnect(
                stop_event=async_stop_event,
                command_callback=wrapped_callback,
            )
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
    except Exception as e:
        logger.error(f"Camera stream error (twin={twin_uuid}): {e}")
        raise
    finally:
        if loop is not None:
            try:
                loop.close()
            except Exception:
                pass


CameraTwinOrDepth = Union["CameraTwin", "DepthCameraTwin"]
TwinOrWithOverrides = Union[
    CameraTwinOrDepth,
    Tuple[CameraTwinOrDepth, Dict[str, Any]],
]


class CameraStreamManager:
    """Manages multiple camera streams in background threads.

    Accepts twins directly; config is inferred from twin type and capabilities.
    Use (twin, overrides) for per-twin overrides (camera_id, fps, etc.).

    Uses run_with_auto_reconnect for each stream (command handling, reconnection).
    Designed for teleoperation and edge scripts that run camera streaming alongside
    other synchronous code.

    Example:
        >>> manager = CameraStreamManager(
        ...     client=cw,
        ...     twins=[
        ...         camera_twin,
        ...         (rs_twin, {"camera_id": 1, "fps": 15}),
        ...     ],
        ...     stop_event=stop_event,
        ...     time_reference=time_reference,
        ...     command_callback=callback,
        ... )
        >>> manager.start()
        >>> # ... run teleop loop ...
        >>> stop_event.set()
        >>> manager.join()
    """

    def __init__(
        self,
        client: "Cyberwave",
        twins: List[TwinOrWithOverrides],
        stop_event: threading.Event,
        time_reference: "TimeReference",
        command_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize the camera stream manager.

        Args:
            client: Cyberwave client instance
            twins: List of CameraTwin or DepthCameraTwin, or (twin, overrides) tuples.
                Config is inferred from twin type (DepthCameraTwin -> realsense, else cv2)
                and capabilities (sensors[].id -> camera_name). Overrides can include:
                camera_id, fps, camera_resolution, enable_depth, depth_fps, etc.
            stop_event: Threading event to signal all streams to stop
            time_reference: Time reference for sync
            command_callback: Optional callback(status, message) for command responses
        """
        self.client = client
        self._configs: List[Dict[str, Any]] = []
        for item in twins:
            if isinstance(item, tuple):
                twin, overrides = item
                self._configs.append(_infer_config_from_twin(twin, overrides))
            else:
                self._configs.append(_infer_config_from_twin(item))
        self.stop_event = stop_event
        self.time_reference = time_reference
        self.command_callback = command_callback
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        """Start all camera streams in background threads."""
        for config in self._configs:
            t = threading.Thread(
                target=_run_streamer_in_thread,
                args=(
                    self.client,
                    config,
                    self.stop_event,
                    self.time_reference,
                    self.command_callback,
                ),
                daemon=True,
            )
            self._threads.append(t)
            t.start()
        logger.info(f"Started {len(self._threads)} camera stream(s)")

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for all camera stream threads to finish."""
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
