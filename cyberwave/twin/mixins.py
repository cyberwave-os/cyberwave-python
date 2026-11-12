"""Capability mixins — attach handles only on the twin classes that need them."""

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from ..exceptions import CyberwaveError

from .capabilities.joints import FIRST_READ_TIMEOUT_S
from .commands import TwinCommandsHandle

if TYPE_CHECKING:
    from ..motion import TwinMotionHandle, TwinNavigationHandle
    from .base import Twin


class PolicyCapableMixin:
    """Controller policy attach and outbound-command gating."""

    _policy_handle: Optional[Any] = None

    def _prepare_outbound_command(self) -> None:
        """Gate outbound MQTT commands behind controller-policy attachment."""
        self.policy.ensure_attached()

    @property
    def policy(self) -> Any:
        """Controller policy attach, list, and keyboard teleop."""
        if self._policy_handle is None:
            from ..managers.policies import TwinPolicyHandle

            self._policy_handle = TwinPolicyHandle(self)  # type: ignore[arg-type]
        return self._policy_handle


class LocomotionCapableMixin:
    """Ground locomotion handle (``can_locomote`` twins)."""

    _locomotion: Optional[Any] = None

    @property
    def locomotion(self) -> Any:
        if self._locomotion is None:
            from .capabilities.locomotion import LocomotionHandle

            self._locomotion = LocomotionHandle(self)  # type: ignore[arg-type]
        return self._locomotion


class FlightCapableMixin:
    """Flight command handle (``can_fly`` twins)."""

    _flight: Optional[Any] = None

    @property
    def flight(self) -> Any:
        if self._flight is None:
            from .capabilities.flight import FlightHandle

            self._flight = FlightHandle(self)  # type: ignore[arg-type]
        return self._flight


class GripperCapableMixin:
    """Gripper / end-effector handle (``can_grip`` twins)."""

    _gripper: Optional[Any] = None

    @property
    def gripper(self) -> Any:
        if self._gripper is None:
            from .capabilities.gripper import GripperHandle

            self._gripper = GripperHandle(self)  # type: ignore[arg-type]
        return self._gripper


class JointsCapableMixin:
    """Joint-space pose for manipulators (``has_joints`` twins)."""

    _joints_handle: Optional[Any] = None

    @property
    def joints(self) -> Any:
        from .capabilities.joints import JointsHandle

        if self._joints_handle is None:
            self._joints_handle = JointsHandle(self)  # type: ignore[arg-type]
        return self._joints_handle

    def get_joints(
        self,
        *,
        what_joints: Optional[Sequence[str]] = None,
        what_data: Sequence[str] = ("position",),
        timeout: float = FIRST_READ_TIMEOUT_S,
        after_update_callback: Optional[Any] = None,
    ) -> Any:
        """Shortcut for :meth:`joints.get` (stable twin-level API).

        Returns a live :class:`~cyberwave.twin.capabilities.joints.JointStateView`
        that auto-updates (default ``{joint_name: position}`` in radians). Pass
        ``what_data`` to read additional fields — ``position``, ``velocity``,
        ``acceleration``, and ``effort`` (effort/torque) — either one kind or
        several (nested dict keyed by kind). Pass *after_update_callback* to run a
        function on every update (cancel via the view's ``stop()``).
        """
        return self.joints.get(
            what_joints=what_joints,
            what_data=what_data,
            timeout=timeout,
            after_update_callback=after_update_callback,
        )

    def get_pose(
        self,
        *,
        what_joints: Optional[Sequence[str]] = None,
        what_data: Sequence[str] = ("position",),
        timeout: float = FIRST_READ_TIMEOUT_S,
        after_update_callback: Optional[Any] = None,
    ) -> Any:
        """Joint-space pose alias for :meth:`get_joints` on manipulator twins."""
        return self.get_joints(
            what_joints=what_joints,
            what_data=what_data,
            timeout=timeout,
            after_update_callback=after_update_callback,
        )

    def set_joints(
        self,
        values: Mapping[str, float] | float | str,
        position: Optional[float] = None,
        *,
        joint: Optional[str] = None,
        what_joints: Optional[Sequence[str]] = None,
        what_data: str = "position",
        degrees: bool = False,
        mode: str = "absolute",
        source_type: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Shortcut for :meth:`joints.set` (stable twin-level API).

        Supports the same ``what_data`` kinds as :meth:`get_joints` (default
        ``position``). Use ``degrees=True`` when setting positions in degrees.
        """
        return self.joints.set(
            values,
            position=position,
            joint=joint,
            what_joints=what_joints,
            what_data=what_data,
            degrees=degrees,
            mode=mode,
            source_type=source_type,
            timestamp=timestamp,
        )

    def set_pose(
        self,
        pose: Any,
        *,
        mode: str = "absolute",
        motion: str = "ptp",
        reference_frame: Optional[str] = None,
        backend: Optional[str] = None,
        expected_joints: Optional[Mapping[str, float]] = None,
        scene_revision: Optional[str] = None,
        plan_only: bool = False,
        source_type: Optional[str] = None,
        **joint_options: Any,
    ) -> None:
        """Move the tool to a Cartesian pose: the arm's ``set_pose`` command.

        *pose* is a full Cartesian pose -- a position in metres AND a rotation
        in radians or as a quaternion, never degrees -- in any of these forms:

        * a :class:`~cyberwave.data.state_representation.CartesianPose`, such
          as ``twin.pose.get()``; its reference frame travels with it;
        * ``[x, y, z, roll, pitch, yaw]`` -- extrinsic XYZ;
        * ``[x, y, z, qx, qy, qz, qw]``;
        * ``(position, rotation)`` -- two arrays, ``[x, y, z]`` and either
          ``[roll, pitch, yaw]`` or ``[qx, qy, qz, qw]``;
        * a waypoint ``{"position": ..., "orientation" | "rotation": {x, y, z,
          w}, "reference_frame": ...}``, e.g. one from
          :class:`~cyberwave.navigation.NavigationPlan`.

        Every other argument is optional. ``mode="relative"`` moves the tool BY
        the pose instead of TO it, along its own axes unless
        ``reference_frame`` says otherwise, starting from the tool's measured
        pose when the command arrives; ``motion="lin"`` moves it along a
        straight line; ``reference_frame`` names any frame of the robot
        description (or ``"base"`` / ``"tool"``); ``backend`` picks a
        registered planner other than the driver's default; ``plan_only``
        plans without moving. The driver reports the result on
        ``state/motion``.

        A mapping of joint names to positions is the old joint-space form of
        this method and still sets the joints (:meth:`set_joints`), with a
        ``DeprecationWarning``; ``degrees``, ``timestamp`` and ``what_data``
        are accepted for that form only.
        """
        if _is_joint_mapping(pose):
            warnings.warn(
                "set_pose() with joint positions is deprecated; use set_joints() "
                "for joint space. set_pose() now moves the tool to a Cartesian pose.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.set_joints(
                pose, mode=mode, source_type=source_type, **joint_options
            )
        if joint_options:
            raise TypeError(
                f"set_pose() got unexpected arguments {sorted(joint_options)}: a "
                "Cartesian pose is metres plus radians or a quaternion, with no "
                "degrees, timestamp or what_data"
            )
        data: Dict[str, Any] = {"pose": _wire_pose(pose)}
        options = {
            "mode": (mode, "absolute"),
            "motion": (motion, "ptp"),
            "reference_frame": (reference_frame, None),
            "backend": (backend, None),
            "expected_joints": (
                None if expected_joints is None else dict(expected_joints),
                None,
            ),
            "scene_revision": (scene_revision, None),
            "plan_only": (plan_only, False),
        }
        # Only what differs from the driver's defaults goes on the wire, so a
        # driver's default stays the driver's to choose.
        data.update(
            {
                key: value
                for key, (value, default) in options.items()
                if value != default
            }
        )
        self.publish_command("set_pose", data, source_type=source_type)  # type: ignore[attr-defined]

    def move(self, pose: Any, **options: Any) -> None:
        """Shortcut for :meth:`set_pose`: move the tool to (or by) *pose*."""
        return self.set_pose(pose, **options)

    #: ``twin.pose.set()`` moves the tool on an arm (see :class:`PoseHandle`).
    _pose_set_moves_the_tool = True


#: Keys that make a mapping a Cartesian waypoint rather than joint positions.
#: No bare ``x`` / ``y`` / ``z``: those are joint names on a gantry.
_POSE_KEYS = frozenset(
    {"position", "orientation", "rotation", "reference_frame", "frame_id"}
)


def _is_joint_mapping(pose: Any) -> bool:
    return isinstance(pose, Mapping) and not (set(pose) & _POSE_KEYS)


def _wire_pose(pose: Any) -> Any:
    """*pose* as JSON: a flat list of six or seven floats, or a waypoint dict.

    Every form carries a rotation. A waypoint's ``rotation`` list follows the
    navigation convention (``[w, x, y, z]``) and leaves named; an
    ``orientation`` must already be a named ``{x, y, z, w}``, because a bare
    list of four is ambiguous.
    """
    from ..data.state_representation import CartesianPose

    if isinstance(pose, CartesianPose):
        out: Dict[str, Any] = {
            "position": pose.translation_dict(),
            "orientation": pose.orientation_dict(),
        }
        if pose.frame_id():
            out["reference_frame"] = pose.frame_id()
        return out
    if isinstance(pose, Mapping):
        if "position" not in pose or not ({"orientation", "rotation"} & set(pose)):
            raise ValueError(
                "a pose needs a position and a rotation (orientation or rotation)"
            )
        from ..navigation import _normalize_position, _normalize_rotation

        out = {}
        for key, value in pose.items():
            if key == "position":
                out[key] = _normalize_position(value)
            elif key == "rotation":
                out[key] = _normalize_rotation(value, None)
            elif key == "orientation":
                if not isinstance(value, Mapping):
                    raise ValueError("orientation must be a named {x, y, z, w}")
                out[key] = {name: float(v) for name, v in value.items()}
            elif key in ("reference_frame", "frame_id"):
                out[key] = value
        return out
    parts = list(pose)
    if len(parts) == 2 and not any(isinstance(part, numbers.Real) for part in parts):
        position, rotation = ([float(v) for v in part] for part in parts)
        if len(position) != 3 or len(rotation) not in (3, 4):
            raise ValueError(
                "(position, rotation) must be ([x, y, z], [roll, pitch, yaw]) or "
                "([x, y, z], [qx, qy, qz, qw])"
            )
        parts = [*position, *rotation]
    values = [float(v) for v in parts]
    if len(values) not in (6, 7):
        raise ValueError(
            "pose must be a CartesianPose, [x, y, z, roll, pitch, yaw], "
            "[x, y, z, qx, qy, qz, qw], (position, rotation) or a waypoint dict"
        )
    return values


class SpatialPoseCapableMixin:
    """Cartesian pose for locomoting twins (MQTT get/set — not REST editor)."""

    #: ``twin.pose.set()`` places the robot itself, even with an arm on board.
    _pose_set_moves_the_tool = False

    def get_pose(self) -> Optional[Dict[str, Dict[str, float]]]:
        """MQTT pose read — same canonical cache as :attr:`pose`.

        Returns ``None`` before any pose has arrived (see ``PoseView``).
        """
        return self.pose.get().to_legacy_pose()  # type: ignore[attr-defined]

    def set_pose(
        self,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        w: Optional[float] = None,
        rx: Optional[float] = None,
        ry: Optional[float] = None,
        rz: Optional[float] = None,
    ) -> None:
        """MQTT pose write (delegates to :attr:`pose`)."""
        return self.pose.set(  # type: ignore[attr-defined]
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            w=w,
            rx=rx,
            ry=ry,
            rz=rz,
        )


class MotionCapableMixin:
    """Marker: motion flat methods live on :class:`~cyberwave.twin.base.Twin`."""


class NavigationCapableMixin:
    """Waypoint navigation handle."""

    _navigation: Optional["TwinNavigationHandle"] = None

    @property
    def navigation(self) -> "TwinNavigationHandle":
        if self._navigation is None:
            from ..motion import TwinNavigationHandle

            self._navigation = TwinNavigationHandle(self)  # type: ignore[arg-type]
        return self._navigation


class PowerCapableMixin:
    """Battery / power queries."""

    _power: Optional[Any] = None

    @property
    def power(self) -> Any:
        if self._power is None:
            from .capabilities.power import PowerHandle

            self._power = PowerHandle(self)  # type: ignore[arg-type]
        return self._power


class CameraCapableMixin:
    """Imaging — ``twin.camera`` is an indexable family of camera sensors.

    ``camera`` is injected on :class:`~cyberwave.twin.base.Twin` via ``__getattr__``
    (not a class property). ``twin.camera.get_frame()`` uses the first sensor;
    ``twin.camera['depth_camera']`` / ``twin.camera[1]`` select a specific one.

    ``twin.calibration`` holds this camera's hand-eye calibration (see
    :class:`~cyberwave.calibration.persistence.CameraCalibrationHandle`).
    """

    _camera_calibration: Optional[Any] = None

    @property
    def calibration(self) -> Any:
        """Hand-eye calibration for this camera (REST only, mirrors ``joints.calibration``).

        Twin-level rather than per-sensor under ``camera[...]``: the solved transform
        is stored in the twin's ``attach_offset_*`` fields, and there is only one of
        those per twin.
        """
        if self._camera_calibration is None:
            from ..calibration.persistence import CameraCalibrationHandle

            self._camera_calibration = CameraCalibrationHandle(self)
        return self._camera_calibration

    def stream(
        self,
        fps: int = 30,
        *,
        idx: int | str = 0,
        sensor: str | None = None,
    ) -> None:
        if sensor is not None:
            self._imaging_handle(sensor=sensor).stream(  # type: ignore[attr-defined]
                fps=fps, camera_id=idx
            )
        else:
            self._default_imaging_handle().stream(fps=fps, camera_id=idx)  # type: ignore[attr-defined]

    def get_frame(
        self,
        format: str = "bytes",
        *,
        path: Optional[str] = None,
        source: str = "cloud",
        sensor_id: Optional[str] = None,
        mock: bool = False,
        idx: int | str = 0,
        max_age_ms: float | None = None,
        zenoh_timeout_s: float = 3.0,
        edge_timeout_s: float = 5.0,
    ) -> Any | None:
        return self._imaging_handle(sensor_id=sensor_id).get_frame(  # type: ignore[attr-defined]
            format,
            path=path,
            source=source,
            sensor_id=self._resolve_sensor_id(sensor_id),  # type: ignore[attr-defined]
            mock=mock,
            idx=idx,
            max_age_ms=max_age_ms,
            zenoh_timeout_s=zenoh_timeout_s,
            edge_timeout_s=edge_timeout_s,
        )

    def get_frames(
        self,
        count: int,
        *,
        interval_ms: int = 0,
        format: str = "path",
        directory: Optional[str] = None,
        path: Optional[str] = None,
        source: str = "cloud",
        sensor_id: Optional[str] = None,
        mock: bool = False,
        idx: int | str = 0,
        max_age_ms: float | None = None,
        zenoh_timeout_s: float = 3.0,
        edge_timeout_s: float = 5.0,
    ) -> List[Any] | str:
        if count < 1:
            raise ValueError("count must be >= 1")

        frame_kwargs = {
            "source": source,
            "sensor_id": sensor_id,
            "mock": mock,
            "idx": idx,
            "max_age_ms": max_age_ms,
            "zenoh_timeout_s": zenoh_timeout_s,
            "edge_timeout_s": edge_timeout_s,
        }

        if format == "path":
            if count == 1 and path is not None:
                single = self.get_frame("path", path=path, **frame_kwargs)
                if single is None:
                    return None
                return single

            import os
            import tempfile

            from .compat import time

            folder = directory
            if folder is None:
                folder = tempfile.mkdtemp(prefix="cyberwave_frames_")
            else:
                os.makedirs(folder, exist_ok=True)

            for i in range(count):
                dest = os.path.join(folder, f"frame_{i:04d}.jpg")
                saved = self.get_frame("path", path=dest, **frame_kwargs)
                if saved is None:
                    return None
                if i < count - 1 and interval_ms > 0:
                    time.sleep(interval_ms / 1000.0)
            return os.path.abspath(folder)

        from .compat import time

        frames: List[Any] = []
        for i in range(count):
            frame = self.get_frame(format, **frame_kwargs)
            frames.append(frame)
            if i < count - 1 and interval_ms > 0:
                time.sleep(interval_ms / 1000.0)
        return frames

    def capture_frame(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "twin.capture_frame() is deprecated; use twin.get_frame()",
            DeprecationWarning,
            stacklevel=2,
        )
        frame_format = args[0] if args else kwargs.get("format", "path")
        result = self.get_frame(
            frame_format,
            source="local",
            sensor_id=kwargs.get("sensor_id"),
            mock=kwargs.get("mock", False),
        )
        if result is None:
            raise CyberwaveError("No frame available")
        return result

    def capture_frames(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "twin.capture_frames() is deprecated; use twin.get_frames()",
            DeprecationWarning,
            stacklevel=2,
        )
        count = args[0] if args else kwargs["count"]
        interval_ms = args[1] if len(args) > 1 else kwargs.get("interval_ms", 100)
        frame_format = args[2] if len(args) > 2 else kwargs.get("format", "path")
        kwargs.pop("count", None)
        result = self.get_frames(
            count,
            interval_ms=interval_ms,
            format=frame_format,
            sensor_id=kwargs.get("sensor_id"),
            mock=kwargs.get("mock", False),
            source=kwargs.get("source", "cloud"),
        )
        if result is None:
            raise CyberwaveError("No frame available")
        return result
