"""
High-level Twin abstraction for intuitive digital twin control
"""

import asyncio
from copy import deepcopy
import json
import math
import os
import threading
import time
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Callable, Type

if TYPE_CHECKING:
    from .client import Cyberwave
    from .camera import CameraStreamer
    from .motion import TwinMotionHandle, TwinNavigationHandle
    from .keyboard import KeyboardTeleop
    from .alerts import TwinAlertManager
    from cyberwave.rest.models.twin_joint_calibration_schema import (
        TwinJointCalibrationSchema,
    )

from .exceptions import CyberwaveError
from .constants import SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE


# Load capabilities cache for runtime class selection
_CAPABILITIES_CACHE: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)


def _run_coroutine_blocking(coro) -> None:  # type: ignore[no-untyped-def]
    """Run an async coroutine in a blocking fashion, compatible with running event loops.

    When called from an environment that already has a running event loop (e.g. Jupyter
    notebooks, Google Colab, IPython), ``asyncio.run()`` raises a ``RuntimeError``.
    In those cases we spin up a dedicated background thread with its own event loop so
    the coroutine can run to completion while the caller's thread blocks normally.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside a running event loop (e.g. Jupyter/Colab).  Run the coroutine
        # in a separate OS thread that owns a fresh event loop.
        exc_holder: list = []

        def _run_in_thread() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(coro)
            except Exception as exc:
                exc_holder.append(exc)
            finally:
                new_loop.close()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()
        t.join()
        if exc_holder:
            raise exc_holder[0]
    else:
        asyncio.run(coro)


def _normalize_locomotion_source_type(source_type: Optional[str]) -> Optional[str]:
    """Preserve legacy ``sim`` callers while publishing ``sim_tele`` commands."""
    if source_type == SOURCE_TYPE_SIM:
        return SOURCE_TYPE_SIM_TELE
    return source_type


def _default_control_source_type(client: Any) -> str:
    runtime_mode = getattr(getattr(client, "config", None), "runtime_mode", "live")
    return SOURCE_TYPE_SIM_TELE if runtime_mode == "simulation" else SOURCE_TYPE_TELE


def _load_capabilities_cache() -> Dict[str, Any]:
    """Load the capabilities cache from JSON file."""
    global _CAPABILITIES_CACHE
    if _CAPABILITIES_CACHE is None:
        cache_path = Path(__file__).parent / "assets_capabilities.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                _CAPABILITIES_CACHE = json.load(f)
        else:
            _CAPABILITIES_CACHE = {}
    return _CAPABILITIES_CACHE


def _get_asset_capabilities(registry_id: str) -> Dict[str, Any]:
    """Get capabilities for an asset by registry_id."""
    cache = _load_capabilities_cache()
    asset_data = cache.get(registry_id, {})
    return asset_data.get("capabilities", {})


def _decode_frame(jpeg_bytes: bytes, format: str) -> Any:
    """Decode JPEG bytes into the requested output format.

    Args:
        jpeg_bytes: Raw JPEG image bytes.
        format: ``"path"`` | ``"bytes"`` | ``"numpy"`` | ``"pil"``.

    Returns:
        Decoded frame in the chosen format.
    """
    if format == "bytes":
        return jpeg_bytes

    if format == "path":
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".jpg", prefix="cyberwave_")
        with os.fdopen(fd, "wb") as f:
            f.write(jpeg_bytes)
        return path

    if format == "numpy":
        try:
            import numpy as np
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            raise CyberwaveError(
                "numpy and opencv-python are required for format='numpy'. "
                "Install with: pip install numpy opencv-python"
            )
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise CyberwaveError("Failed to decode JPEG frame")
        return frame

    if format == "pil":
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            raise CyberwaveError(
                "Pillow is required for format='pil'. Install with: pip install Pillow"
            )
        import io

        return Image.open(io.BytesIO(jpeg_bytes))

    raise CyberwaveError(
        f"Unknown format '{format}'. Supported: 'path', 'bytes', 'numpy', 'pil'"
    )


class JointController:
    """Controller for robot joints"""

    def __init__(self, twin: "Twin"):
        self.twin = twin
        self._joint_states: Optional[Dict[str, float]] = None

    def refresh(self):
        """Refresh joint states from the server"""
        try:
            states = self.twin.client.twins.get_joint_states(self.twin.uuid)
            parsed: Dict[str, float] = {}

            legacy = getattr(states, "joint_states", None)
            if legacy:
                for js in legacy:
                    parsed[str(getattr(js, "joint_name", ""))] = float(
                        getattr(js, "position", 0.0)
                    )
            else:
                names = getattr(states, "name", None)
                positions = getattr(states, "position", None)
                if isinstance(names, list) and isinstance(positions, list):
                    for joint_name, pos in zip(names, positions):
                        parsed[str(joint_name)] = float(pos)

            self._joint_states = parsed
        except Exception as e:
            raise CyberwaveError(f"Failed to refresh joint states: {e}")

    def get(self, joint_name: str) -> float:
        """Get current position of a joint"""
        if self._joint_states is None:
            self.refresh()

        # After refresh, _joint_states should be a dict
        if self._joint_states is None or joint_name not in self._joint_states:
            raise CyberwaveError(f"Joint '{joint_name}' not found")

        return self._joint_states[joint_name]

    def set(
        self,
        joint_name: str,
        position: float,
        degrees: bool = True,
        timestamp: Optional[float] = None,
        source_type: Optional[str] = None,
    ):
        """
        Set position of a joint

        Args:
            joint_name: Name of the joint
            position: Target position
            degrees: If True, position is in degrees; otherwise radians
            timestamp: Unix timestamp for the update
            source_type: Source type (e.g. SOURCE_TYPE_EDGE_LEADER, SOURCE_TYPE_EDGE_FOLLOWER)
        """
        if degrees:
            position = math.radians(position)

        if source_type is None:
            source_type = _default_control_source_type(self.twin.client)

        try:
            # Connect to MQTT if not already connected
            self.twin._connect_to_mqtt_if_not_connected()

            # Update joint state via MQTT
            self.twin.client.mqtt.update_joint_state(
                self.twin.uuid,
                joint_name,
                position=position,
                timestamp=timestamp,
                source_type=source_type,
            )

            # Update cached state
            if self._joint_states is None:
                self._joint_states = {}
            self._joint_states[joint_name] = position

        except Exception as e:
            raise CyberwaveError(f"Failed to set joint '{joint_name}': {e}")

    def __getattr__(self, name: str) -> float:
        """Allow accessing joints as attributes (e.g., joints.arm_joint)"""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return self.get(name)

    def __setattr__(self, name: str, value: float):
        """Allow setting joints as attributes (e.g., joints.arm_joint = 45)"""
        if name in ["twin", "_joint_states"]:
            super().__setattr__(name, value)
        else:
            self.set(name, value)

    def list(self) -> List[str]:
        """Get list of all joint names"""
        if self._joint_states is None:
            self.refresh()
        if self._joint_states is None:
            return []
        return list(self._joint_states.keys())

    def get_all(self) -> Dict[str, float]:
        """Get all joint states as a dictionary"""
        if self._joint_states is None:
            self.refresh()
        if self._joint_states is None:
            return {}
        return self._joint_states.copy()

    def print_joint_states(self) -> None:
        """Print all joint states in a human-readable table (radians and degrees).

        Fetches the latest joint states from the server and prints them in a
        formatted table showing each joint name with its position in both radians
        and degrees.

        Example output::

            Joint States for twin <twin-uuid>:
            ┌──────────────────────────────────────┬───────────────────┬──────────────────┐
            │ Joint                                │   Radians         │   Degrees        │
            ├──────────────────────────────────────┼───────────────────┼──────────────────┤
            │ shoulder_joint                       │    0.7854 rad     │    45.00 °       │
            │ elbow_joint                          │    0.0000 rad     │     0.00 °       │
            └──────────────────────────────────────┴───────────────────┴──────────────────┘
        """
        self.refresh()
        states = self.get_all()

        if not states:
            print(f"No joint states found for twin {self.twin.uuid}")
            return

        col_name_w = max(len("Joint"), max(len(n) for n in states))
        header = (
            f"{'Joint':<{col_name_w}}  {'Radians':>12}  {'Degrees':>12}"
        )
        separator = "-" * len(header)

        print(f"\nJoint states for twin {self.twin.uuid}:")
        print(separator)
        print(header)
        print(separator)
        for name, position_rad in sorted(states.items()):
            position_deg = math.degrees(position_rad)
            print(
                f"{name:<{col_name_w}}  {position_rad:>10.4f} rad  {position_deg:>10.2f} °"
            )
        print(separator)


class TwinControllerHandle:
    """Handle for controller functionality like keyboard teleop."""

    def __init__(self, twin: "Twin"):
        self._twin = twin

    def keyboard(
        self,
        bindings: Any,
        *,
        step: float = 0.05,
        rate_hz: int = 20,
        fetch_initial: bool = True,
        verbose: bool = True,
    ) -> "KeyboardTeleop":
        """
        Create a keyboard teleop controller.

        Args:
            bindings: KeyboardBindings instance or list of binding dicts
            step: Position change per keypress (degrees)
            rate_hz: Polling rate in Hz
            fetch_initial: Whether to fetch initial joint positions
            verbose: Whether to print status messages

        Returns:
            KeyboardTeleop instance ready to run
        """
        from .keyboard import KeyboardBindings, KeyboardTeleop

        payload = (
            bindings.build() if isinstance(bindings, KeyboardBindings) else bindings
        )
        return KeyboardTeleop(
            self._twin,
            payload,
            step=step,
            rate_hz=rate_hz,
            fetch_initial=fetch_initial,
            verbose=verbose,
        )


class TwinCameraHandle:
    """Lightweight namespace for camera/vision operations on a twin.

    Provides a discoverable API surface for all vision-related methods,
    similar to ``twin.joints`` or ``twin.navigation``.

    Example:
        >>> frame = twin.camera.read()           # numpy array (BGR)
        >>> path  = twin.camera.snapshot()        # save JPEG to temp file
        >>> twin.camera.stream(fps=15)            # blocking video stream
    """

    def __init__(self, twin: "Twin"):
        self._twin = twin

    def read(
        self,
        format: str = "numpy",
        *,
        sensor_id: Optional[str] = None,
        mock: bool = False,
    ) -> Any:
        """Read the latest frame, defaulting to a numpy BGR array.

        Behaves like ``cv2.VideoCapture.read()`` — returns the most recent
        frame from the twin's camera sensor.

        Args:
            format: Output format — ``"numpy"`` (default), ``"pil"``,
                ``"bytes"``, or ``"path"``.
            sensor_id: Sensor id for multi-camera twins. Omit for single-camera.
            mock: Request a deterministic mock frame.

        Returns:
            Frame in the requested format.
        """
        return self._twin.capture_frame(format=format, sensor_id=sensor_id, mock=mock)

    def snapshot(
        self,
        path: Optional[str] = None,
        *,
        sensor_id: Optional[str] = None,
        mock: bool = False,
    ) -> str:
        """Save a JPEG snapshot to disk and return the file path.

        Args:
            path: Destination file path. A temporary file is created when *None*.
            sensor_id: Sensor id for multi-camera twins.
            mock: Request a deterministic mock frame.

        Returns:
            Absolute path to the saved JPEG file.
        """
        if path is None:
            return self._twin.capture_frame(
                format="path", sensor_id=sensor_id, mock=mock
            )
        jpeg_bytes = self._twin.get_latest_frame(sensor_id=sensor_id, mock=mock)
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return os.path.abspath(path)

    def stream(self, fps: int = 30, camera_id: int | str = 0) -> None:
        """Start a blocking video stream (Ctrl+C to stop).

        Delegates to ``CameraTwin.start_streaming``. Raises if the twin
        does not support local camera streaming.

        Args:
            fps: Frames per second.
            camera_id: Camera device ID or stream URL.
        """
        if not hasattr(self._twin, "start_streaming"):
            raise CyberwaveError(
                "Video streaming requires a twin with camera sensors. "
                "This twin does not have streaming capabilities."
            )
        self._twin.start_streaming(fps=fps, camera_id=camera_id)

    def edge_photo(
        self,
        format: str = "bytes",
        *,
        timeout: float = 5.0,
    ) -> Any:
        """Request a photo from the edge device via MQTT.

        Sends a ``take_photo`` command and waits for the edge to respond
        on the ``camera/photo`` topic. Unlike :meth:`read` (which fetches
        the latest cached frame via REST), this triggers a fresh capture on
        the physical device.

        Args:
            format: Output format — ``"bytes"`` (default) or ``"numpy"``.
            timeout: Seconds to wait for the edge response.

        Returns:
            Frame in the requested format.

        Raises:
            CyberwaveError: On timeout, edge error, or missing image data.
        """
        import base64

        twin = self._twin
        twin._connect_to_mqtt_if_not_connected()
        mqtt = twin.client.mqtt

        topic_prefix = twin.client.config.topic_prefix or ""
        photo_topic = f"{topic_prefix}cyberwave/twin/{twin.uuid}/camera/photo"
        command_topic = f"{topic_prefix}cyberwave/twin/{twin.uuid}/command"

        result_holder: Dict[str, Any] = {}
        event = threading.Event()

        def _on_photo(payload_str: str) -> None:
            try:
                result_holder["data"] = json.loads(payload_str)
            except Exception as exc:
                result_holder["error"] = str(exc)
            event.set()

        mqtt.subscribe(photo_topic, _on_photo)
        try:
            mqtt.publish(
                command_topic,
                {
                    "command": "take_photo",
                    "source_type": "tele",
                    "timestamp": time.time(),
                },
            )

            if not event.wait(timeout):
                raise CyberwaveError(
                    f"Timed out waiting for take_photo response after {timeout}s"
                )

            if "error" in result_holder:
                raise CyberwaveError(
                    f"Failed to parse edge photo response: {result_holder['error']}"
                )

            data = result_holder["data"]

            if data.get("status") == "error":
                raise CyberwaveError(data.get("message", "Edge returned an error"))

            if "image" not in data:
                raise CyberwaveError(
                    "Edge photo response missing 'image' field"
                )

            jpeg_bytes = base64.b64decode(data["image"])
            return _decode_frame(jpeg_bytes, format)
        finally:
            mqtt.unsubscribe(photo_topic)

    def edge_photos(
        self,
        count: int,
        interval_ms: int = 100,
        format: str = "bytes",
        *,
        timeout: float = 5.0,
    ) -> List[Any]:
        """Capture multiple photos from the edge device.

        Calls :meth:`edge_photo` ``count`` times with ``interval_ms``
        delay between each capture.

        Args:
            count: Number of photos to capture.
            interval_ms: Delay between captures in milliseconds.
            format: Output format (same as :meth:`edge_photo`).
            timeout: Per-photo timeout in seconds.

        Returns:
            List of frames in the requested format.
        """
        frames: List[Any] = []
        for i in range(count):
            frames.append(self.edge_photo(format=format, timeout=timeout))
            if i < count - 1:
                time.sleep(interval_ms / 1000.0)
        return frames


class Twin:
    """
    High-level abstraction for a digital twin.

    Provides intuitive methods for controlling position, rotation, scale,
    and joint states of a digital twin.

    Example:
        >>> twin = client.twin("the-robot-studio/so101")
        >>> twin.edit_position(x=1, y=0, z=0.5)
        >>> twin.rotate(yaw=90)
        >>> twin.joints.arm_joint = 45
    """

    def __init__(self, client: "Cyberwave", twin_data: Any):
        """
        Initialize a Twin instance

        Args:
            client: Cyberwave client instance
            twin_data: Twin schema data from API
        """
        self.client = client
        self._data = twin_data
        self.joints = JointController(self)

        # Cache for current state
        self._position: Optional[Dict[str, float]] = None
        self._rotation: Optional[Dict[str, float]] = None

        # Lazy-initialized motion, navigation, alerts, and camera handles
        self._motion: Optional["TwinMotionHandle"] = None
        self._navigation: Optional["TwinNavigationHandle"] = None
        self._alerts: Optional["TwinAlertManager"] = None
        self._camera_handle: Optional["TwinCameraHandle"] = None
        self._scale: Optional[Dict[str, float]] = None

    @property
    def uuid(self) -> str:
        """Get twin UUID"""
        return (
            self._data.uuid
            if hasattr(self._data, "uuid")
            else str(self._data.get("uuid", ""))
        )

    @property
    def name(self) -> str:
        """Get twin name"""
        return (
            self._data.name
            if hasattr(self._data, "name")
            else str(self._data.get("name", ""))
        )

    @property
    def asset_id(self) -> str:
        """Get asset ID"""
        return (
            self._data.asset_uuid
            if hasattr(self._data, "asset_uuid")
            else str(self._data.get("asset_uuid", ""))
        )

    @property
    def environment_id(self) -> str:
        """Get environment ID"""
        return (
            self._data.environment_uuid
            if hasattr(self._data, "environment_uuid")
            else str(self._data.get("environment_uuid", ""))
        )

    @property
    def parent(self) -> Optional["Twin"]:
        """Get this twin's parent twin, if docked."""
        parent_uuid = None
        if hasattr(self._data, "attach_to_twin_uuid"):
            parent_uuid = self._data.attach_to_twin_uuid
        elif isinstance(self._data, dict):
            parent_uuid = self._data.get("attach_to_twin_uuid")

        if not parent_uuid:
            return None

        try:
            return self.client.twins.get(str(parent_uuid))
        except Exception as e:
            raise CyberwaveError(f"Failed to fetch parent twin '{parent_uuid}': {e}")

    @property
    def children(self) -> List["Twin"]:
        """Get child twins docked to this twin."""
        child_twin_uuids: Any = []
        if hasattr(self._data, "child_twin_uuids"):
            child_twin_uuids = self._data.child_twin_uuids or []
        elif isinstance(self._data, dict):
            child_twin_uuids = self._data.get("child_twin_uuids") or []

        if not isinstance(child_twin_uuids, list) or not child_twin_uuids:
            return []

        children: List["Twin"] = []
        for child_uuid in child_twin_uuids:
            if not child_uuid:
                continue
            try:
                child_twin = self.client.twins.get(str(child_uuid))
                children.append(child_twin)
            except Exception as e:
                raise CyberwaveError(
                    f"Failed to fetch child twin '{child_uuid}': {e}"
                ) from e
        return children

    @property
    def motion(self) -> "TwinMotionHandle":
        """
        Access motion control for poses and animations.

        Example:
            >>> twin.motion.asset.pose("Picking from below", transition_ms=800)
            >>> twin.motion.twin.animation("wave", transition_ms=500)
            >>> keyframes = twin.motion.asset.list_keyframes()

        Returns:
            TwinMotionHandle for motion control
        """
        if self._motion is None:
            from .motion import TwinMotionHandle

            self._motion = TwinMotionHandle(self)
        return self._motion

    @property
    def navigation(self) -> "TwinNavigationHandle":
        """
        Access navigation control for waypoint-based movement.

        Example:
            >>> twin.navigation.goto([1, 2, 0])
            >>> twin.navigation.follow_path([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
            >>> twin.navigation.stop()

        Returns:
            TwinNavigationHandle for navigation control
        """
        if self._navigation is None:
            from .motion import TwinNavigationHandle

            self._navigation = TwinNavigationHandle(self)
        return self._navigation

    @property
    def alerts(self) -> "TwinAlertManager":
        """
        Access alert management for this twin.

        Example:
            >>> alert = twin.alerts.create(name="Calibration needed")
            >>> for a in twin.alerts.list():
            ...     print(a.name, a.status)
            >>> alert.resolve()

        Returns:
            TwinAlertManager for creating / listing / managing alerts
        """
        if self._alerts is None:
            from .alerts import TwinAlertManager

            self._alerts = TwinAlertManager(self)
        return self._alerts

    @property
    def controller(self) -> "TwinControllerHandle":
        """
        Access controller functionality for keyboard teleop.

        Example:
            >>> from cyberwave import KeyboardBindings
            >>> bindings = KeyboardBindings().bind("W", "joint1", "increase")
            >>> teleop = twin.controller.keyboard(bindings, step=2.0)
            >>> teleop.run()

        Returns:
            TwinControllerHandle for controller access
        """
        return TwinControllerHandle(self)

    @property
    def camera(self) -> "TwinCameraHandle":
        """
        Access camera/vision operations for this twin.

        Provides a lightweight namespace with methods like ``read()``,
        ``snapshot()``, and ``stream()`` — keeping the Twin class clean
        while making all vision ops easily discoverable.

        Example:
            >>> frame = twin.camera.read()            # numpy BGR array
            >>> path  = twin.camera.snapshot()         # save JPEG to temp file
            >>> twin.camera.stream(fps=15)             # blocking video stream

        Returns:
            TwinCameraHandle for camera access
        """
        if self._camera_handle is None:
            self._camera_handle = TwinCameraHandle(self)
        return self._camera_handle

    def refresh(self):
        """Refresh twin data from the server"""
        try:
            self._data = self.client.twins.get_raw(self.uuid)
            self._position = None
            self._rotation = None
            self._scale = None
        except Exception as e:
            raise CyberwaveError(f"Failed to refresh twin: {e}")

    def edit_position(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ):
        """
        Edit the twin's position in the environment.

        NOTE: Does not move the twin in the real world.

        Args:
            x: X coordinate (optional, keeps current if None)
            y: Y coordinate (optional, keeps current if None)
            z: Z coordinate (optional, keeps current if None)
        """
        # Get current position if needed
        current = self._get_current_position()

        update_data = {
            "position_x": x if x is not None else current.get("x", 0),
            "position_y": y if y is not None else current.get("y", 0),
            "position_z": z if z is not None else current.get("z", 0),
        }

        self._update_state(update_data)

        # Update cache
        self._position = {
            "x": update_data["position_x"],
            "y": update_data["position_y"],
            "z": update_data["position_z"],
        }

    def edit_rotation(
        self,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        quaternion: Optional[List[float]] = None,
    ):
        """
        Edit the twin's rotation in the environment.
        NOTE: Does not rotate the twin in the real world.

        Args:
            yaw: Yaw angle in degrees (rotation around Z axis)
            pitch: Pitch angle in degrees (rotation around Y axis)
            roll: Roll angle in degrees (rotation around X axis)
            quaternion: Quaternion [x, y, z, w] (alternative to euler angles)
        """
        if quaternion is not None:
            if len(quaternion) != 4:
                raise CyberwaveError("Quaternion must be [x, y, z, w]")

            update_data = {
                "rotation_x": quaternion[0],
                "rotation_y": quaternion[1],
                "rotation_z": quaternion[2],
                "rotation_w": quaternion[3],
            }
        else:
            # Convert euler angles to quaternion
            quat = self._euler_to_quaternion(roll or 0, pitch or 0, yaw or 0)
            update_data = {
                "rotation_x": quat[0],
                "rotation_y": quat[1],
                "rotation_z": quat[2],
                "rotation_w": quat[3],
            }

        self._update_state(update_data)

        # Update cache
        self._rotation = {
            "x": update_data["rotation_x"],
            "y": update_data["rotation_y"],
            "z": update_data["rotation_z"],
            "w": update_data["rotation_w"],
        }

    def edit_scale(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ):
        """
        Edit the twin's scale in the environment.
        NOTE: Does not scale the twin in the real world (nothing can be scaled in the real world).

        Args:
            x: X scale factor
            y: Y scale factor
            z: Z scale factor
        """
        current = self._get_current_scale()

        update_data = {
            "scale_x": x if x is not None else current.get("x", 1),
            "scale_y": y if y is not None else current.get("y", 1),
            "scale_z": z if z is not None else current.get("z", 1),
        }

        self._update_state(update_data)

        # Update cache
        self._scale = {
            "x": update_data["scale_x"],
            "y": update_data["scale_y"],
            "z": update_data["scale_z"],
        }

    def _data_get(self, field: str, default: Any = None) -> Any:
        """Read a field from TwinSchema object or dict payload."""
        if hasattr(self._data, field):
            return getattr(self._data, field)
        if isinstance(self._data, dict):
            return self._data.get(field, default)
        return default

    def _build_deepcopy_payload_for_recreation(self) -> Dict[str, Any]:
        """Build a deep-copied payload for recreating this twin in another environment."""
        # Keep this aligned with backend clone semantics where possible using public
        # twin create/update fields exposed by the SDK.
        fields_to_copy = (
            "name",
            "description",
            "position_x",
            "position_y",
            "position_z",
            "rotation_w",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "scale_x",
            "scale_y",
            "scale_z",
            "kinematics_override",
            "joint_calibration",
            "metadata",
            "controller_policy_uuid",
            "attach_offset_x",
            "attach_offset_y",
            "attach_offset_z",
            "attach_offset_rotation_w",
            "attach_offset_rotation_x",
            "attach_offset_rotation_y",
            "attach_offset_rotation_z",
            "fixed_base",
        )
        payload: Dict[str, Any] = {}
        for field in fields_to_copy:
            value = self._data_get(field)
            if value is None:
                continue
            payload[field] = deepcopy(value)
        return payload

    def _delete_environment(self, environment_uuid: str) -> None:
        """Delete an environment, supporting both project and standalone paths."""
        environment = self.client.environments.get(environment_uuid)  # type: ignore
        project_uuid = (
            environment.project_uuid
            if hasattr(environment, "project_uuid")
            else (
                environment.get("project_uuid")
                if isinstance(environment, dict)
                else None
            )
        )
        if project_uuid:
            self.client.environments.delete(environment_uuid, str(project_uuid))  # type: ignore
            return

        # Fallback for standalone environments (delete endpoint exists in backend
        # but may not always be exposed by generated SDK stubs).
        _param = self.client._api_client.param_serialize(
            method="DELETE",
            resource_path="/api/v1/environments/{uuid}",
            path_params={"uuid": environment_uuid},
            auth_settings=["CustomTokenAuthentication"],
        )
        response_data = self.client._api_client.call_api(*_param)
        response_data.read()

    def add_to_environment(self, environment_uuid: str) -> "Twin":
        """Recreate this twin in another environment and delete the original twin."""
        if not environment_uuid:
            raise CyberwaveError("environment_uuid is required")

        source_environment_uuid = self.environment_id
        source_twin_uuid = self.uuid
        if str(source_environment_uuid) == str(environment_uuid):
            return self

        try:
            payload = self._build_deepcopy_payload_for_recreation()

            recreated_twin_data = self.client.twins.create(  # type: ignore
                asset_id=self.asset_id,
                environment_id=environment_uuid,
                **payload,
            )

            self.client.twins.delete(source_twin_uuid)  # type: ignore

            remaining_twins = self.client.twins.list(  # type: ignore
                environment_id=source_environment_uuid
            )
            if not remaining_twins:
                self._delete_environment(source_environment_uuid)

            self._data = recreated_twin_data
            self._position = None
            self._rotation = None
            self._scale = None
            return self
        except Exception as e:
            raise CyberwaveError(
                f"Failed to add twin to environment '{environment_uuid}': {e}"
            )

    def delete(self) -> None:
        """Delete this twin"""
        try:
            self.client.twins.delete(self.uuid)  # type: ignore
        except Exception as e:
            raise CyberwaveError(f"Failed to delete twin: {e}")

    def get_latest_frame(
        self,
        sensor_id: Optional[str] = None,
        mock: bool = False,
        source_type: Optional[str] = None,
    ) -> bytes:
        """Get the latest JPEG frame available for this twin.

        Args:
            sensor_id: Optional camera sensor id for multi-camera twins.
            mock: If true, request deterministic mock JPEG bytes.
            source_type: Optional ``"sim"``/``"tele"`` override.
                When omitted, this method follows ``cw.affect(...)``.

        Returns:
            JPEG bytes from the latest frame.
        """
        try:
            resolved_source_type = source_type
            if resolved_source_type is None:
                client_config = getattr(self.client, "config", None)
                configured_source_type = getattr(client_config, "source_type", None)
                if isinstance(configured_source_type, str):
                    normalized_source_type = configured_source_type.strip().lower()
                    if normalized_source_type in {"sim", "simulation"}:
                        resolved_source_type = "sim"
                    elif normalized_source_type in {
                        "tele",
                        "real-world",
                        "real",
                        "teleoperation",
                        "edge",
                    }:
                        resolved_source_type = "tele"

            manager_kwargs: Dict[str, Any] = {
                "sensor_id": sensor_id,
                "mock": mock,
            }
            if resolved_source_type in {"sim", "tele"}:
                manager_kwargs["source_type"] = resolved_source_type

            return self.client.twins.get_latest_frame(  # type: ignore
                self.uuid,
                **manager_kwargs,
            )
        except Exception as e:
            raise CyberwaveError(
                f"Failed to get latest frame for twin {self.uuid}: {e}"
            )

    def capture_frame(
        self,
        format: str = "path",
        *,
        sensor_id: Optional[str] = None,
        mock: bool = False,
        source_type: Optional[str] = None,
    ) -> Any:
        """Capture a single frame from the twin's camera sensor.

        Fetches the latest JPEG frame via the REST API and converts it to
        the requested output format.

        Args:
            format: Output format:

                - ``"path"`` (default) — save to a temp file, return its path.
                - ``"bytes"`` — raw JPEG bytes.
                - ``"numpy"`` — BGR ``numpy.ndarray`` (requires *numpy* + *opencv-python*).
                - ``"pil"`` — ``PIL.Image`` (requires *Pillow*).
            sensor_id: Sensor id for multi-camera twins.  Omit when the twin
                has a single RGB sensor.
            mock: Request a deterministic mock frame (useful for testing).
            source_type: Optional source selector (``"sim"``/``"tele"``).
                When omitted, this follows the active ``cw.affect(...)`` mode.

        Returns:
            Frame in the requested format.

        Raises:
            CyberwaveError: If the sensor is not streaming, the format is
                unknown, or an optional dependency is missing.

        Example:
            >>> frame_path = twin.capture_frame()                  # temp JPEG path
            >>> frame_np   = twin.capture_frame("numpy")           # numpy BGR array
            >>> frame_pil  = twin.capture_frame("pil")             # PIL Image
            >>> frame_raw  = twin.capture_frame("bytes")           # raw JPEG bytes
        """
        jpeg_bytes = self.get_latest_frame(
            sensor_id=sensor_id,
            mock=mock,
            source_type=source_type,
        )
        return _decode_frame(jpeg_bytes, format)

    def capture_frames(
        self,
        count: int,
        interval_ms: int = 100,
        format: str = "path",
        *,
        sensor_id: Optional[str] = None,
        mock: bool = False,
        source_type: Optional[str] = None,
    ) -> Any:
        """Capture multiple frames with a delay between each grab.

        Useful for quick data-collection without setting up a full stream.

        Args:
            count: Number of frames to capture.
            interval_ms: Delay in milliseconds between consecutive captures.
            format: Output format (same options as :meth:`capture_frame`).
                When ``"path"`` (default), returns the path to a temporary
                folder containing numbered JPEG files.  For all other
                formats the return value is a ``list``.
            sensor_id: Sensor id for multi-camera twins.
            mock: Request deterministic mock frames.
            source_type: Optional source selector (``"sim"``/``"tele"``).

        Returns:
            A folder path (``format="path"``) or a list of frames.

        Example:
            >>> folder = twin.capture_frames(5, interval_ms=200)
            >>> frames = twin.capture_frames(5, format="numpy")
        """
        import time as _time
        import tempfile as _tempfile

        if count < 1:
            raise CyberwaveError("count must be >= 1")

        if format == "path":
            folder = _tempfile.mkdtemp(prefix="cyberwave_frames_")
            for i in range(count):
                jpeg_bytes = self.get_latest_frame(
                    sensor_id=sensor_id,
                    mock=mock,
                    source_type=source_type,
                )
                frame_path = os.path.join(folder, f"frame_{i:04d}.jpg")
                with open(frame_path, "wb") as f:
                    f.write(jpeg_bytes)
                if i < count - 1:
                    _time.sleep(interval_ms / 1000.0)
            return folder

        frames = []
        for i in range(count):
            frame = self.capture_frame(
                format=format,
                sensor_id=sensor_id,
                mock=mock,
                source_type=source_type,
            )
            frames.append(frame)
            if i < count - 1:
                _time.sleep(interval_ms / 1000.0)
        return frames

    def _update_state(self, data: Dict[str, Any]):
        """Update twin state via API"""
        try:
            self.client.twins.update_state(self.uuid, data)  # type: ignore
        except Exception as e:
            raise CyberwaveError(f"Failed to update twin state: {e}")

    def _get_current_position(self) -> Dict[str, float]:
        """Get current position from cache or server"""
        if self._position is None:
            # First try to use existing data without making an API call
            if hasattr(self._data, "position_x"):
                self._position = {
                    "x": self._data.position_x,
                    "y": self._data.position_y,
                    "z": self._data.position_z,
                }
            elif isinstance(self._data, dict) and "position_x" in self._data:
                self._position = {
                    "x": self._data.get("position_x", 0),
                    "y": self._data.get("position_y", 0),
                    "z": self._data.get("position_z", 0),
                }
            else:
                # Only refresh from server if we don't have the data
                self.refresh()
                if hasattr(self._data, "position_x"):
                    self._position = {
                        "x": self._data.position_x,
                        "y": self._data.position_y,
                        "z": self._data.position_z,
                    }
                else:
                    self._position = {"x": 0, "y": 0, "z": 0}
        return self._position

    def _get_current_scale(self) -> Dict[str, float]:
        """Get current scale from cache or server"""
        if self._scale is None:
            # First try to use existing data without making an API call
            if hasattr(self._data, "scale_x"):
                self._scale = {
                    "x": self._data.scale_x,
                    "y": self._data.scale_y,
                    "z": self._data.scale_z,
                }
            elif isinstance(self._data, dict) and "scale_x" in self._data:
                self._scale = {
                    "x": self._data.get("scale_x", 1),
                    "y": self._data.get("scale_y", 1),
                    "z": self._data.get("scale_z", 1),
                }
            else:
                # Only refresh from server if we don't have the data
                self.refresh()
                if hasattr(self._data, "scale_x"):
                    self._scale = {
                        "x": self._data.scale_x,
                        "y": self._data.scale_y,
                        "z": self._data.scale_z,
                    }
                else:
                    self._scale = {"x": 1, "y": 1, "z": 1}
        return self._scale

    def _get_current_rotation(self) -> Dict[str, float]:
        """Get current rotation from cache or server"""
        if self._rotation is None:
            # First try to use existing data without making an API call
            if hasattr(self._data, "rotation_w"):
                self._rotation = {
                    "w": self._data.rotation_w,
                    "x": self._data.rotation_x,
                    "y": self._data.rotation_y,
                    "z": self._data.rotation_z,
                }
            elif isinstance(self._data, dict) and "rotation_w" in self._data:
                self._rotation = {
                    "w": self._data.get("rotation_w", 1.0),
                    "x": self._data.get("rotation_x", 0.0),
                    "y": self._data.get("rotation_y", 0.0),
                    "z": self._data.get("rotation_z", 0.0),
                }
            else:
                # Only refresh from server if we don't have the data
                self.refresh()
                if hasattr(self._data, "rotation_w"):
                    self._rotation = {
                        "w": self._data.rotation_w,
                        "x": self._data.rotation_x,
                        "y": self._data.rotation_y,
                        "z": self._data.rotation_z,
                    }
                else:
                    self._rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        return self._rotation

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
        """
        Convert euler angles (degrees) to quaternion

        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees

        Returns:
            [x, y, z, w] quaternion
        """
        # Convert to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # Calculate quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [x, y, z, w]

    def __repr__(self) -> str:
        return f"Twin(uuid='{self.uuid}', name='{self.name}')"

    def _connect_to_mqtt_if_not_connected(self):
        """Connect to MQTT if not connected"""
        if not self.client.mqtt.connected:
            self.client.mqtt.connect()

    def subscribe(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin(self.uuid, on_update)

    def subscribe_position(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to movement updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin_position(self.uuid, on_update)

    def subscribe_rotation(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to rotation updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin_rotation(self.uuid, on_update)

    def subscribe_joints(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to joint updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_joint_states(self.uuid, on_update)

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get twin capabilities from the underlying data."""
        if hasattr(self._data, "capabilities"):
            return self._data.capabilities or {}
        elif isinstance(self._data, dict):
            return self._data.get("capabilities", {})
        return {}

    def has_capability(self, capability: str) -> bool:
        """Check if the twin has a specific capability."""
        return bool(self.capabilities.get(capability, False))

    def has_sensor(self, sensor_type: Optional[str] = None) -> bool:
        """Check if the twin has sensors, optionally of a specific type."""
        sensors = self.capabilities.get("sensors", [])
        if not sensors:
            return False
        if sensor_type is None:
            return True
        return any(s.get("type") == sensor_type for s in sensors)

    # =========================================================================
    # Universal Schema APIs
    # =========================================================================

    def get_controllable_joint_names(self) -> List[str]:
        """
        Get controllable joint names from the twin's universal schema.

        Returns joint names for revolute, prismatic, and continuous joints,
        sorted by name for consistent ordering (e.g. _1, _2, _3 for SO101).

        Matches the logic used by the backend (get_controllable_joints) and
        recording tasks. Use these names for joint updates and initial observations.

        Returns:
            List of joint names (e.g. ["_1", "_2", "_3", "_4", "_5", "_6"] for SO101)

        Example:
            >>> joint_names = twin.get_controllable_joint_names()
            >>> twin.joints.set(joint_names[0], 0.5, degrees=False)
        """
        CONTROLLABLE_JOINT_TYPES = frozenset({"revolute", "prismatic", "continuous"})
        schema = self.get_schema()
        if not schema:
            return []
        joints = schema.get("joints", [])
        controllable = [
            j["name"]
            for j in joints
            if isinstance(j, dict)
            and j.get("name")
            and j.get("type") in CONTROLLABLE_JOINT_TYPES
        ]
        return sorted(controllable)

    def get_schema(self, path: str = "") -> Any:
        """Get value at a specific JSON Pointer path in the twin's universal schema.

        Args:
            path: JSON Pointer path (e.g., "/sensors/0", "/extensions/cyberwave/capabilities")
                 Empty string returns the entire schema

        Returns:
            The value at the specified path (can be dict, list, string, etc.)

        Example:
            # Get entire schema
            schema = twin.get_schema()

            # Get specific path
            sensor = twin.get_schema("/sensors/0")

            # Get capabilities
            capabilities = twin.get_schema("/extensions/cyberwave/capabilities")
        """
        result = self.client.twins.get_universal_schema_at_path(self.uuid, path)
        return result.get("value")

    def update_schema(
        self, path: str, value: Any, op: str = "replace"
    ) -> Dict[str, Any]:
        """Update the twin's universal schema using JSON Pointer operations.

        Args:
            path: JSON Pointer path to update (e.g., "/sensors/0/parameters/id")
            value: Value to set at the path
            op: Operation type - "add" or "replace" (default: "replace")

        Returns:
            Dict with the updated schema and operation details

        Example:
            # Update a sensor ID
            twin.update_schema(
                path="/sensors/0/parameters/id",
                value="my_camera"
            )

            # Add a new capability
            twin.update_schema(
                path="/extensions/cyberwave/capabilities/can_fly",
                value=True,
                op="add"
            )
        """
        return self.client.twins.patch_universal_schema(self.uuid, path, value, op)

    def get_calibration(
        self, robot_type: Optional[str] = None
    ) -> "TwinJointCalibrationSchema":
        """
        Get calibration data for this twin.

        Args:
            robot_type: Optional robot type filter ("leader" or "follower").
                       If None, returns all calibration data.

        Returns:
            TwinJointCalibrationSchema containing calibration data

        Example:
            >>> calibration = twin.get_calibration(robot_type="leader")
            >>> print(calibration.joint_calibration["shoulder_pan"].range_min)
        """
        return self.client.twins.get_calibration(self.uuid, robot_type=robot_type)

    def update_calibration(
        self,
        joint_calibration: Dict[str, Dict[str, Any]],
        robot_type: str,
    ) -> "TwinJointCalibrationSchema":
        """
        Update calibration data for this twin.

        Args:
            joint_calibration: Dictionary mapping joint names to calibration data.
                             Each calibration dict should contain:
                             - range_min: float
                             - range_max: float
                             - homing_offset: float
                             - drive_mode: int or str
                             - id: int or str (motor ID)
            robot_type: Robot type ("leader" or "follower")

        Returns:
            Updated TwinJointCalibrationSchema

        Example:
            >>> calibration = {
            ...     "shoulder_pan": {
            ...         "range_min": 0.0,
            ...         "range_max": 4095.0,
            ...         "homing_offset": 2047.5,
            ...         "drive_mode": 0,
            ...         "id": 1
            ...     },
            ... }
            >>> result = twin.update_calibration(calibration, "leader")
            >>> print(result.joint_calibration["shoulder_pan"].range_min)
        """
        return self.client.twins.update_calibration(
            self.uuid, joint_calibration, robot_type
        )

    def delete_calibration(self, robot_type: Optional[str] = None) -> None:
        """
        Delete calibration data for this twin.

        Args:
            robot_type: Optional. "leader" or "follower" to clear only that type.
                       If None, clears both leader and follower calibration.
        """
        self.client.twins.delete_calibration(self.uuid, robot_type=robot_type)


class CameraTwin(Twin):
    """
    Twin with camera/sensor capabilities.

    Provides methods for video streaming and frame capture for twins
    that have RGB or depth sensors.

    Example:
        >>> twin = client.twin("unitree/go2")  # Returns CameraTwin if has sensors
        >>> await twin.stream_video_background(fps=15)
        >>> frame = twin.capture_frame()
    """

    _camera_streamer: Optional["CameraStreamer"] = None

    @property
    def default_camera_name(self) -> str:
        """Default sensor/camera id for WebRTC signaling (``sensor`` in offers).

        Uses the first entry in :attr:`Twin.capabilities` ``sensors`` and its ``id``,
        same rule as :func:`cyberwave.sensor.manager._infer_config_from_twin`.
        Falls back to ``"default"`` when missing or empty.
        """
        sensors = self.capabilities.get("sensors", [])
        if sensors and isinstance(sensors[0], dict):
            sid = sensors[0].get("id")
            return str(sid) if sid is not None else "default"
        return "default"

    def streamer(self) -> "CameraStreamer":
        """Get the camera streamer."""
        if self._camera_streamer is None:
            raise CyberwaveError("Camera streamer not initialized")
        return self._camera_streamer

    async def stream_video_background(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        fourcc: Optional[str] = None,
        camera_name: Optional[str] = None,
        **kwargs,
    ) -> "CameraStreamer":
        """
        Start video streaming in the background. Non-blocking.

        Returns immediately with the streamer so you can run other code.
        Use stream_video_background() for simple blocking scripts.

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID or stream URL (default: 0)
            fourcc: Optional FOURCC for local V4L2/USB cameras (e.g. ``'MJPG'``). If omitted,
                :class:`~cyberwave.sensor.camera_cv2.CV2VideoTrack` tries ``MJPG`` by default.
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``resolution``, ``keyframe_interval``, ``frame_callback``, ``time_reference``).

        Returns:
            CameraStreamer instance for managing the stream
        """
        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_id=camera_id,
            fps=fps,
            fourcc=fourcc,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )
        await self._camera_streamer.start()
        return self._camera_streamer

    async def stop_streaming(self) -> None:
        """Stop camera streaming."""
        if self._camera_streamer is not None:
            await self._camera_streamer.stop()
            self._camera_streamer = None

    def start_streaming(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        camera_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Stream video until Ctrl+C. Blocking.

        Starts video streaming and blocks until KeyboardInterrupt (Ctrl+C).
        Ideal for 2-line scripts: twin = cw.twin(...); twin.start_streaming()

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID or stream URL (default: 0)
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``fourcc``, ``resolution``, ``keyframe_interval``, ``frame_callback``).
        """
        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_id=camera_id,
            fps=fps,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )

        async def _run():
            await self._camera_streamer.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self._camera_streamer.stop()
                self._camera_streamer = None

        try:
            _run_coroutine_blocking(_run())
        except KeyboardInterrupt:
            pass
        finally:
            if self._camera_streamer is not None:
                try:
                    _run_coroutine_blocking(self._camera_streamer.stop())
                except Exception:
                    pass
                self._camera_streamer = None

    def __repr__(self) -> str:
        sensors = self.capabilities.get("sensors", [])
        sensor_types = [s.get("type", "unknown") for s in sensors]
        return f"CameraTwin(uuid='{self.uuid}', name='{self.name}', sensors={sensor_types})"


class DepthCameraTwin(CameraTwin):
    """
    Twin with depth camera capabilities.

    Extends CameraTwin with depth-specific methods for point cloud
    generation and depth frame capture.
    """

    _camera_streamer: Optional["CameraStreamer"] = None

    def streamer(self) -> "CameraStreamer":
        """Get the camera streamer."""
        if self._camera_streamer is None:
            raise CyberwaveError("Camera streamer not initialized")
        return self._camera_streamer

    async def stop_streaming(self) -> None:
        """Stop camera streaming."""
        if self._camera_streamer is not None:
            # The streamer handles cleanup in its stop method
            await self._camera_streamer.stop()
            self._camera_streamer = None

    async def stream_video_background(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        fourcc: Optional[str] = None,
        camera_name: Optional[str] = None,
        *,
        enable_depth: bool = True,
        **kwargs,
    ) -> "CameraStreamer":
        """
        Start video streaming in the background. Non-blocking.

        Returns immediately with the streamer so you can run other code.
        Use start_streaming() for simple blocking scripts.

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID (default: 0)
            fourcc: Optional FOURCC code (inherited from CameraTwin, unused for RealSense)
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            enable_depth: Enable depth streaming (default: True for DepthCameraTwin)
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``resolution``, ``keyframe_interval``, ``time_reference``).

        Returns:
            CameraStreamer instance for managing the stream
        """
        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_type="realsense",
            camera_id=camera_id,
            fps=fps,
            enable_depth=enable_depth,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )
        await self._camera_streamer.start()
        return self._camera_streamer

    def start_streaming(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        enable_depth: bool = True,
        camera_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Stream video until Ctrl+C. Blocking.

        Starts video streaming and blocks until KeyboardInterrupt (Ctrl+C).
        Ideal for 2-line scripts: twin = cw.twin(...); twin.start_streaming()

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID (default: 0)
            enable_depth: Enable depth streaming (default: True for DepthCameraTwin)
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``resolution``, ``keyframe_interval``, ``time_reference``).
        """
        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_type="realsense",
            camera_id=camera_id,
            fps=fps,
            enable_depth=enable_depth,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )

        async def _run():
            await self._camera_streamer.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self._camera_streamer.stop()
                self._camera_streamer = None

        try:
            _run_coroutine_blocking(_run())
        except KeyboardInterrupt:
            pass
        finally:
            if self._camera_streamer is not None:
                try:
                    _run_coroutine_blocking(self._camera_streamer.stop())
                except Exception:
                    pass
                self._camera_streamer = None

    def capture_depth_frame(self) -> bytes:
        """
        Capture a single depth frame.

        Returns:
            Raw depth frame bytes
        """
        raise NotImplementedError(
            "capture_depth_frame() requires an active depth stream. "
            "Use stream_video_background() first."
        )

    def get_point_cloud(self) -> List[tuple]:
        """
        Get point cloud from depth sensor.

        Returns:
            List of (x, y, z) tuples representing 3D points
        """
        raise NotImplementedError(
            "get_point_cloud() requires depth sensor data processing. "
            "This feature is not yet implemented."
        )

    def __repr__(self) -> str:
        return f"DepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteTwin(Twin):
    """
    Twin that can locomote across space.

    Provides methods for locomotion including movement and rotation.

    Note: Flying twins can locomoate AND fly, so a flying twin is a subset of the LocomoteTwin
    """

    def move(self, position: List[float]):
        """
        DEPRECATED: See warning

        Support for move will be dropped in future versions of the SDK
        """
        logger.warning(
            """move() is deprecated as a way to send commands. You have these two options:
                - Use edit_position if you want to edit the digital twin position in your environemnt, in order to reproduce a real environment in Cyberwave
                - Use move_forward or move_backward if you want your robot to navigate the world
            """
        )
        return

    def move_forward(self, distance: float, source_type: Optional[str] = None):
        """
        Sends a command to a locomotion robot to move in the direction it is facing.

        Args:
            distance: Distance to move in meters
            source_type: ``"sim_tele"``/``"sim"`` for simulation, ``"tele"`` for the real robot.
                Falls back to the client-level setting from ``cw.affect()``.

        Note: This is different than edit_position. edit_position edits the twin in the Editor so that you can
        set up your environment. move_forward sends a command to the robot to move in the direction it is facing.
        """
        if source_type is None:
            source_type = _default_control_source_type(self.client)
        source_type = _normalize_locomotion_source_type(source_type)
        if source_type not in [SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE]:
            raise ValueError(
                f"Invalid source type '{source_type}' for move_forward. "
                "Use cw.affect('simulation') or cw.affect('real-world'), "
                "or pass source_type='sim' / 'sim_tele' / 'tele' directly."
            )
        # Send movement command via MQTT
        self._connect_to_mqtt_if_not_connected()
        topic_prefix = self.client.config.topic_prefix or ""
        self.client.mqtt.publish(
            f"{topic_prefix}cyberwave/twin/{self.uuid}/command",
            {
                "source_type": source_type,
                "command": "move_forward",
                "data": {"linear_x": distance, "angular_z": 0.0},
                "timestamp": time.time(),
            },
        )

    def move_backward(self, distance: float, source_type: Optional[str] = None):
        """
        Sends a command to a locomotion robot to move backward.

        Args:
            distance: Distance to move in meters
            source_type: ``"sim_tele"``/``"sim"`` for simulation, ``"tele"`` for the real robot.
                Falls back to the client-level setting from ``cw.affect()``.

        Note: This is different than edit_position. edit_position edits the twin in the Editor so that you can
        set up your environment. move_backward sends a command to the robot to move in reverse.
        """
        if source_type is None:
            source_type = _default_control_source_type(self.client)
        source_type = _normalize_locomotion_source_type(source_type)
        if source_type not in [SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE]:
            raise ValueError(
                f"Invalid source type '{source_type}' for move_backward. "
                "Use cw.affect('simulation') or cw.affect('real-world'), "
                "or pass source_type='sim' / 'sim_tele' / 'tele' directly."
            )
        # Send movement command via MQTT
        self._connect_to_mqtt_if_not_connected()
        topic_prefix = self.client.config.topic_prefix or ""
        self.client.mqtt.publish(
            f"{topic_prefix}cyberwave/twin/{self.uuid}/command",
            {
                "source_type": source_type,
                "command": "move_backward",
                "data": {"linear_x": distance, "angular_z": 0.0},
                "timestamp": time.time(),
            },
        )

    def turn_left(self, angle: float = 1.5, source_type: Optional[str] = None):
        """
        Sends a command to a locomotion robot to turn left.

        Args:
            angle: Angle to turn in radians (default: 1.5)
            source_type: ``"sim_tele"``/``"sim"`` for simulation, ``"tele"`` for the real robot.
                Falls back to the client-level setting from ``cw.affect()``.
        """
        if source_type is None:
            source_type = _default_control_source_type(self.client)
        source_type = _normalize_locomotion_source_type(source_type)
        if source_type not in [SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE]:
            raise ValueError(
                f"Invalid source type '{source_type}' for turn_left. "
                "Use cw.affect('simulation') or cw.affect('real-world'), "
                "or pass source_type='sim' / 'sim_tele' / 'tele' directly."
            )
        # Send movement command via MQTT
        self._connect_to_mqtt_if_not_connected()
        topic_prefix = self.client.config.topic_prefix or ""
        self.client.mqtt.publish(
            f"{topic_prefix}cyberwave/twin/{self.uuid}/command",
            {
                "source_type": source_type,
                "command": "turn_left",
                "data": {"linear_x": 0, "angular_z": angle},
                "timestamp": time.time(),
            },
        )

    def turn_right(self, angle: float = 1.5, source_type: Optional[str] = None):
        """
        Sends a command to a locomotion robot to turn right.

        Args:
            angle: Angle to turn in radians (default: 1.5)
            source_type: ``"sim_tele"``/``"sim"`` for simulation, ``"tele"`` for the real robot.
                Falls back to the client-level setting from ``cw.affect()``.
        """
        if source_type is None:
            source_type = _default_control_source_type(self.client)
        source_type = _normalize_locomotion_source_type(source_type)
        if source_type not in [SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE]:
            raise ValueError(
                f"Invalid source type '{source_type}' for turn_right. "
                "Use cw.affect('simulation') or cw.affect('real-world'), "
                "or pass source_type='sim' / 'sim_tele' / 'tele' directly."
            )
        # Send movement command via MQTT
        self._connect_to_mqtt_if_not_connected()
        topic_prefix = self.client.config.topic_prefix or ""
        self.client.mqtt.publish(
            f"{topic_prefix}cyberwave/twin/{self.uuid}/command",
            {
                "source_type": source_type,
                "command": "turn_right",
                "data": {"linear_x": 0, "angular_z": angle},
                "timestamp": time.time(),
            },
        )

    def rotate(
        self,
        *,
        w: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> None:
        """
        DEPRECATED: Use edit_rotation instead
        """
        logger.warning("rotate() is deprecated. Use edit_rotation() instead.")
        self.edit_rotation(yaw=yaw, pitch=pitch, roll=roll)


class FlyingTwin(Twin):
    """
    Twin with flight capabilities (drones, UAVs).

    Provides methods for aerial control including takeoff, landing,
    and hovering.
    """

    def takeoff(self, altitude: float = 1.0) -> None:
        """
        Take off to the specified altitude.

        Args:
            altitude: Target altitude in meters (default: 1.0)
        """
        self._connect_to_mqtt_if_not_connected()
        # Send takeoff command via MQTT
        self.client.mqtt.publish(
            f"twins/{self.uuid}/commands/takeoff", {"altitude": altitude}
        )
        # if the default source is telesim, also update the hovering status. In other scenarios (e.g. actual drone driver running) the hovering status is updated by the driver.
        if _default_control_source_type(self.client) == SOURCE_TYPE_SIM_TELE:
            self.set_hovering_status(hovering=True, hovering_altitude=altitude)

    def land(self) -> None:
        """Land the drone."""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(f"twins/{self.uuid}/commands/land", {})
        # if the default source is telesim, also update the hovering status. In other scenarios (e.g. actual drone driver running) the hovering status is updated by the driver.
        if _default_control_source_type(self.client) == SOURCE_TYPE_SIM_TELE:
            self.set_hovering_status(hovering=False)

    def hover(self) -> None:
        """Hover in place."""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(f"twins/{self.uuid}/commands/hover", {})
        # if the default source is telesim, also update the hovering status. In other scenarios (e.g. actual drone driver running) the hovering status is updated by the driver.
        if _default_control_source_type(self.client) == SOURCE_TYPE_SIM_TELE:
            self.set_hovering_status(hovering=True)

    # ------------------------------------------------------------------
    # Hovering status helpers
    # ------------------------------------------------------------------

    def is_hovering(self) -> bool:
        """
        Return True if this twin is currently in hovering mode.

        The hovering state is stored in ``twin.metadata.status.controller_requested_hovering``.
        This method reads the locally-cached twin data; call :meth:`refresh`
        first if you need the latest server-side value.

        Returns:
            bool: True when metadata.status.controller_requested_hovering is True, False otherwise.
        """
        meta: Dict[str, Any] = {}
        if hasattr(self._data, "metadata") and self._data.metadata:
            meta = dict(self._data.metadata)
        elif isinstance(self._data, dict):
            meta = self._data.get("metadata") or {}
        return bool(
            meta.get("status", {}).get("controller_requested_hovering", False)
        )

    def get_hovering_status(self) -> Dict[str, Any]:
        """
        Return the hovering status dict from this twin's metadata.

        The returned dict follows the schema::

            {
                "controller_requested_hovering": bool,
                "controller_requested_hovering_altitude": float | None,  # altitude in metres
            }

        This method reads the locally-cached twin data; call :meth:`refresh`
        first if you need the latest server-side value.

        Returns:
            dict: Hovering status with keys ``controller_requested_hovering`` and
            optionally ``controller_requested_hovering_altitude``.
        """
        meta: Dict[str, Any] = {}
        if hasattr(self._data, "metadata") and self._data.metadata:
            meta = dict(self._data.metadata)
        elif isinstance(self._data, dict):
            meta = self._data.get("metadata") or {}
        status = meta.get("status") or {}
        return {
            "controller_requested_hovering": bool(
                status.get("controller_requested_hovering", False)
            ),
            "controller_requested_hovering_altitude": status.get(
                "controller_requested_hovering_altitude"
            ),
        }

    def set_hovering_status(
        self,
        *,
        hovering: bool,
        hovering_altitude: Optional[float] = None,
    ) -> None:
        """
        Persist the hovering status to the twin's metadata on the server.

        This performs a deep-merge into ``twin.metadata.status`` so that
        other metadata fields are not overwritten.

        Args:
            hovering: Whether the drone is currently hovering.
            hovering_altitude: Current altitude in meters. Required (or
                strongly recommended) when ``hovering`` is True.  Pass
                ``None`` to leave any existing value unchanged.

        Example::

            twin.set_hovering_status(hovering=True, hovering_altitude=2.5)
            twin.set_hovering_status(hovering=False)

        The values are persisted under
        ``twin.metadata.status.controller_requested_hovering`` and
        ``twin.metadata.status.controller_requested_hovering_altitude``.
        """
        # Read current metadata so we can merge rather than overwrite
        meta: Dict[str, Any] = {}
        if hasattr(self._data, "metadata") and self._data.metadata:
            meta = dict(self._data.metadata)
        elif isinstance(self._data, dict):
            meta = dict(self._data.get("metadata") or {})

        status: Dict[str, Any] = dict(meta.get("status") or {})
        status["controller_requested_hovering"] = hovering
        if hovering_altitude is not None:
            status["controller_requested_hovering_altitude"] = hovering_altitude
        elif not hovering:
            # Clear altitude when landing so stale values don't persist
            status.pop("controller_requested_hovering_altitude", None)

        meta["status"] = status

        try:
            self.client.twins.update(self.uuid, metadata=meta)  # type: ignore[union-attr]
        except Exception as exc:
            raise CyberwaveError(
                f"Failed to update hovering status for twin {self.uuid}: {exc}"
            ) from exc

        # Keep local cache in sync
        if hasattr(self._data, "metadata"):
            self._data.metadata = meta  # type: ignore[assignment]
        elif isinstance(self._data, dict):
            self._data["metadata"] = meta

    def __repr__(self) -> str:
        return f"FlyingTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperTwin(Twin):
    """
    Twin with gripper/manipulation capabilities.

    Provides methods for controlling grippers and end effectors.
    """

    def grip(self, force: float = 1.0) -> None:
        """
        Close the gripper with specified force.

        Args:
            force: Grip force (0.0 to 1.0, default: 1.0)
        """
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(
            f"twins/{self.uuid}/commands/grip", {"force": max(0.0, min(1.0, force))}
        )

    def release(self) -> None:
        """Open the gripper."""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(f"twins/{self.uuid}/commands/release", {})

    def __repr__(self) -> str:
        return f"GripperTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingCameraTwin(FlyingTwin, CameraTwin):
    """Twin with both flight and camera capabilities (camera drones)."""

    def __repr__(self) -> str:
        return f"FlyingCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperCameraTwin(GripperTwin, CameraTwin):
    """Twin with both gripper and camera capabilities (manipulators with vision)."""

    def __repr__(self) -> str:
        return f"GripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperDepthCameraTwin(GripperTwin, DepthCameraTwin):
    """Twin with both gripper and depth camera capabilities (manipulators with vision)."""

    def __repr__(self) -> str:
        return f"GripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperTwin(LocomoteTwin, GripperTwin):
    """Twin with both locomotive and gripper capabilities (robots with grippers)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingGripperDepthCameraTwin(FlyingTwin, GripperDepthCameraTwin):
    """Twin with both flight and gripper and depth camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingGripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperDepthCameraTwin(LocomoteTwin, GripperDepthCameraTwin):
    """Twin with both locomotive and gripper and depth camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteDepthCameraTwin(LocomoteTwin, DepthCameraTwin):
    """Twin with both locomotive and depth camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperCameraTwin(LocomoteTwin, GripperCameraTwin):
    """Twin with both locomotive and gripper and camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteCameraTwin(LocomoteTwin, CameraTwin):
    """Twin with both locomotive and camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingGripperCameraTwin(FlyingTwin, GripperCameraTwin):
    """Twin with both flight and gripper and camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingGripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingDepthCameraTwin(FlyingTwin, DepthCameraTwin):
    """Twin with both flight and depth camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


def _select_twin_class(capabilities: Dict[str, Any]) -> Type[Twin]:
    """
    Select the appropriate Twin subclass based on capabilities.

    Args:
        capabilities: Asset capabilities dictionary

    Returns:
        The most appropriate Twin subclass
    """
    has_sensors = bool(capabilities.get("sensors", []))
    has_depth = any(s.get("type") == "depth" for s in capabilities.get("sensors", []))
    can_fly = capabilities.get("can_fly", False)
    can_locomote = capabilities.get("can_locomote", False)
    can_grip = capabilities.get("can_grip", False)

    # Select class based on combination of capabilities
    if can_fly:
        if can_grip and has_depth:
            return FlyingGripperDepthCameraTwin
        elif can_grip and has_sensors:
            return FlyingGripperCameraTwin
        elif has_sensors:
            return FlyingCameraTwin
        elif has_depth:
            return FlyingDepthCameraTwin
        elif can_grip:
            return FlyingGripperCameraTwin
        else:
            return FlyingTwin
    elif can_locomote:
        if can_grip and has_depth:
            return LocomoteGripperDepthCameraTwin
        elif can_grip and has_sensors:
            return LocomoteGripperCameraTwin
        elif can_grip:
            return LocomoteGripperTwin
        elif has_depth:
            return LocomoteDepthCameraTwin
        elif has_sensors:
            return LocomoteCameraTwin
        else:
            return LocomoteTwin
    elif can_grip and has_sensors:
        return GripperCameraTwin
    elif can_grip and has_depth:
        return GripperDepthCameraTwin
    elif can_fly:
        return FlyingTwin
    elif can_locomote:
        return LocomoteTwin
    elif can_grip:
        return GripperTwin
    elif has_depth:
        return DepthCameraTwin
    elif has_sensors:
        return CameraTwin
    else:
        return Twin


def create_twin(
    client: "Cyberwave",
    twin_data: Any,
    registry_id: Optional[str] = None,
) -> Twin:
    """
    Factory function to create the appropriate Twin subclass.

    This function examines the twin's capabilities and returns an instance
    of the most appropriate Twin subclass, providing IDE autocomplete
    for capability-specific methods.

    Args:
        client: Cyberwave client instance
        twin_data: Twin schema data from API
        registry_id: Optional asset registry ID for capability lookup

    Returns:
        Appropriate Twin subclass instance (CameraTwin, FlyingTwin, etc.)

    Example:
        >>> twin = create_twin(client, twin_data, "unitree/go2")
        >>> # twin is CameraTwin with start_streaming() available
    """
    # Get capabilities - prefer cached JSON which has complete capability data
    capabilities = {}

    if registry_id:
        # Use cached capabilities from JSON (most complete source)
        capabilities = _get_asset_capabilities(registry_id)

    # Fall back to twin_data capabilities if no cached data
    if not capabilities:
        if hasattr(twin_data, "capabilities") and twin_data.capabilities:
            caps = twin_data.capabilities
            # Convert to dict if it's an object
            capabilities = (
                caps if isinstance(caps, dict) else getattr(caps, "__dict__", {})
            )
        elif isinstance(twin_data, dict) and twin_data.get("capabilities"):
            capabilities = twin_data["capabilities"]

    # Select and instantiate the appropriate class
    twin_class = _select_twin_class(capabilities)
    return twin_class(client, twin_data)
