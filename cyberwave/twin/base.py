from __future__ import annotations

import logging
import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..exceptions import (
    CyberwaveError,
    NotSimulatedError,
    SimulationLevelError,
    SimulationNotRunningError,
)

from ._helpers import (
    _SDK_JOINT_INPUT_DEVICES,
    _build_controller_assignment_metadata,
    _check_controller_ready_live,
    _emit_controller_policy_alert,
    _get_twin_metadata,
    _pick_default_sdk_joint_policy_uuid,
    _policy_is_sdk_joint_teleop_candidate,
    _sdk_auto_attach_controller_enabled,
)
from .commands import TwinCommandsHandle
from .driver import TwinDriverHandle
from .telemetry import TwinTelemetry
from .editor import TwinEditorMixin
from .transport import TwinTransportMixin

if TYPE_CHECKING:
    from ..client import Cyberwave
    from ..alerts import TwinAlertManager
    from .simulation_support import SimLevel
    from ..motion import TwinMotionHandle
    from ..managers.recordings import TwinRecordingsHandle
    from .capability_resolve import HandlerResolution
    from cyberwave.rest.models.twin_joint_calibration_schema import (
        TwinJointCalibrationSchema,
    )

logger = logging.getLogger(__name__)

_SENSOR_FAMILY_ATTRS: tuple[str, ...] = (
    "camera",
    "lidar",
    "gps",
    "compass",
    "imu",
    "flashlight",
)

# Removed plural sensor namespaces mapped to their singular replacement family.
# Accessing the old plural raises a directed AttributeError instead of a bare one.
_REMOVED_PLURAL_SENSOR_ATTRS: dict[str, str] = {
    "cameras": "camera",
    "lidars": "lidar",
    "gpss": "gps",
    "compasses": "compass",
    "imus": "imu",
    "flashlights": "flashlight",
}


class Twin(TwinEditorMixin, TwinTransportMixin):
    """
    High-level abstraction for a digital twin.

    Provides intuitive methods for controlling position, rotation, scale,
    and joint states of a digital twin.

    Example:
        >>> twin = client.twin("the-robot-studio/so101")
        >>> twin.edit_position(x=1, y=0, z=0.5)
        >>> twin.rotate(yaw=90)
        >>> twin.joints.set("joint_1", 45, degrees=True)
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
        self._controller_ensured: bool = False
        self._init_transport_state()
        self._mqtt_catalog_cache: Optional[Dict[str, Any]] = None
        self._driver_catalog_cache: Optional[Dict[str, Any]] = None

        # Cache for current state
        self._position: Optional[Dict[str, float]] = None
        self._rotation: Optional[Dict[str, float]] = None

        # Lazy-initialized handles (capability-specific handles live on mixins)
        self._alerts: Optional["TwinAlertManager"] = None
        self._commands_handle: Optional[TwinCommandsHandle] = None
        self._recordings_handle: Optional["TwinRecordingsHandle"] = None
        self._driver_handle: Optional[TwinDriverHandle] = None
        self._telemetry_handle: Optional[TwinTelemetry] = None
        self._motion: Optional["TwinMotionHandle"] = None
        self._camera_handle: Optional[Any] = None
        self._pose: Optional[Any] = None
        self._scale: Optional[Dict[str, float]] = None

    def _get_workspace_uuid(self) -> Optional[str]:
        """Return the workspace UUID for this twin's environment (non-fatal)."""
        try:
            env = self.client.environments.get(self.environment_id)
            if hasattr(env, "workspace_uuid"):
                return str(env.workspace_uuid) if env.workspace_uuid else None
        except Exception:
            pass
        return None

    def _list_controller_policies(self) -> List[Any]:
        """Fetch controller policies scoped to this twin's asset and workspace."""
        api = getattr(getattr(self.client, "twins", None), "api", None)
        if api is None:
            raise CyberwaveError("Client does not expose a controller-policies API")
        try:
            return api.src_app_api_controller_policies_list_controller_policies(
                asset_uuid=self.asset_id or None,
                workspace_uuid=self._get_workspace_uuid(),
            )
        except Exception as e:
            raise CyberwaveError(f"Failed to list controller policies: {e}") from e

    def _pick_controller_policy(self, policies: List[Any]) -> Any:
        """Return the best policy object to use for SDK joint commands.

        Keeps the twin's current teleop assignment if it is visible in *policies*
        (i.e. passes the workspace filter).  Otherwise picks the best
        sdk/keyboard candidate.
        """
        current: Optional[str] = None
        if hasattr(self._data, "controller_policy_uuid"):
            raw = self._data.controller_policy_uuid
            current = str(raw) if raw else None
        elif isinstance(self._data, dict):
            raw = self._data.get("controller_policy_uuid")
            current = str(raw) if raw else None

        if current:
            cur_policy = next((p for p in policies if str(p.uuid) == current), None)
            if cur_policy is None:
                logger.warning(
                    "Twin %s: assigned controller %r not visible in workspace policy list; "
                    "will replace with a suitable candidate",
                    self.uuid,
                    current,
                )
            elif (
                str(getattr(cur_policy, "controller_type", "") or "").lower()
                == "teleop"
            ):
                return cur_policy

        candidates = [p for p in policies if _policy_is_sdk_joint_teleop_candidate(p)]
        if not candidates:
            raise CyberwaveError(
                "No controller policy suitable for SDK joint commands was found "
                f"(need a teleop policy with input_device in "
                f"{sorted(_SDK_JOINT_INPUT_DEVICES)!r}). "
                "Attach a teleop controller to this twin in the UI, or ensure "
                "this workspace has default sdk/keyboard controller policies."
            )
        chosen_uuid = _pick_default_sdk_joint_policy_uuid(candidates)
        return next(p for p in candidates if str(p.uuid) == chosen_uuid)

    def _apply_controller_policy(self, policy: Any) -> None:
        """PUT the chosen policy onto the twin (FK + metadata), or sync metadata if FK already matches."""
        current: Optional[str] = None
        if hasattr(self._data, "controller_policy_uuid"):
            raw = self._data.controller_policy_uuid
            current = str(raw) if raw else None
        elif isinstance(self._data, dict):
            raw = self._data.get("controller_policy_uuid")
            current = str(raw) if raw else None

        chosen_uuid = str(policy.uuid)
        chosen_name = getattr(policy, "name", chosen_uuid)
        metadata_update = _build_controller_assignment_metadata(self._data, policy)

        if current != chosen_uuid:
            _emit_controller_policy_alert(self, action="assign", policy=policy)
            try:
                self._data = self.client.twins.update(
                    self.uuid,
                    controller_policy_uuid=chosen_uuid,
                    metadata=metadata_update,
                )
            except Exception as e:
                raise CyberwaveError(
                    f"Failed to attach controller policy to twin: {e}"
                ) from e
            logger.info("Twin %s: assigned controller %r", self.uuid, chosen_name)
        elif (
            _get_twin_metadata(self._data).get("controller_policy_uuid") != chosen_uuid
        ):
            try:
                self._data = self.client.twins.update(
                    self.uuid, metadata=metadata_update
                )
                logger.info(
                    "Twin %s: synced controller metadata for %r", self.uuid, chosen_name
                )
            except Exception as exc:
                logger.warning(
                    "Twin %s: metadata sync failed (non-fatal): %s", self.uuid, exc
                )
        else:
            logger.debug(
                "Twin %s: controller %r already assigned", self.uuid, chosen_name
            )

    def _attached_controller_policy_uuid(self) -> Optional[str]:
        """Return the twin's controller policy UUID, if any."""
        if hasattr(self._data, "controller_policy_uuid"):
            raw = self._data.controller_policy_uuid
        elif isinstance(self._data, dict):
            raw = self._data.get("controller_policy_uuid")
        else:
            return None
        return str(raw) if raw else None

    def _unassign_controller_policy(self, *, emit_alert: bool = True) -> None:
        """Clear controller policy FK + metadata (REST)."""
        if emit_alert and self._attached_controller_policy_uuid():
            _emit_controller_policy_alert(self, action="unassign")
        try:
            self._data = self.client.twins.update(
                self.uuid,
                controller_policy_uuid="",
                metadata=_build_controller_assignment_metadata(self._data, None),
            )
        except Exception as e:
            raise CyberwaveError(f"Failed to unassign controller policy: {e}") from e
        self._controller_ensured = False

    def disconnect(self) -> None:
        """Release live-session resources for this twin.

        In ``live`` runtime mode, detaches any assigned controller policy so the
        edge controller can be released. Simulation and other modes are no-ops.
        Publishes a ``Disconnecting`` operator alert before clearing the controller.
        """
        runtime_mode = getattr(
            getattr(self.client, "config", None), "runtime_mode", "live"
        )
        if runtime_mode != "live":
            return
        if not self._attached_controller_policy_uuid():
            self._controller_ensured = False
            return
        _emit_controller_policy_alert(self, action="disconnect")
        self._unassign_controller_policy(emit_alert=False)
        logger.info("Twin %s: disconnected (controller policy unassigned)", self.uuid)

    def _ensure_controller_ready(self) -> None:
        """Auto-attach a teleop controller and stub live readiness check.

        Set ``CYBERWAVE_SDK_AUTO_ATTACH_CONTROLLER=0`` to skip assignment.
        """
        if self._controller_ensured:
            return

        if not _sdk_auto_attach_controller_enabled():
            logger.debug(
                "Twin %s: auto-attach disabled; skipping controller assignment",
                self.uuid,
            )
        elif getattr(getattr(self.client, "twins", None), "api", None) is None:
            logger.debug(
                "Twin %s: client has no controller-policies API; skipping auto-attach",
                self.uuid,
            )
        else:
            policies = self._list_controller_policies()
            policy = self._pick_controller_policy(policies)
            self._apply_controller_policy(policy)

        runtime_mode = getattr(
            getattr(self.client, "config", None), "runtime_mode", "live"
        )
        if runtime_mode == "live" and not _check_controller_ready_live():
            raise CyberwaveError(
                "Robot controller is not ready for live joint commands."
            )

        self._controller_ensured = True

    @property
    def commands(self) -> TwinCommandsHandle:
        """MQTT catalog command invocation (``twin.commands.<name>(...)``)."""
        if self._commands_handle is None:
            self._commands_handle = TwinCommandsHandle(self)
        return self._commands_handle

    @property
    def recordings(self) -> "TwinRecordingsHandle":
        """List and fetch this twin's recordings (``twin.recordings``)."""
        if self._recordings_handle is None:
            from ..managers.recordings import TwinRecordingsHandle

            self._recordings_handle = TwinRecordingsHandle(self)
        return self._recordings_handle

    @property
    def driver(self) -> TwinDriverHandle:
        """Driver interface catalogs (MQTT + Zenoh): getters and ``set_schema``."""
        if self._driver_handle is None:
            self._driver_handle = TwinDriverHandle(self)
        return self._driver_handle

    @property
    def telemetry(self) -> TwinTelemetry:
        """MQTT telemetry publisher for this twin."""
        if self._telemetry_handle is None:
            self._telemetry_handle = TwinTelemetry(self)
        return self._telemetry_handle

    @property
    def pose(self) -> Any:
        if self._pose is None:
            from .capabilities.pose import PoseHandle

            self._pose = PoseHandle(self)
        return self._pose

    @property
    def motion(self) -> "TwinMotionHandle":
        """Saved poses, movements, and animations (scope defaults to ``auto``)."""
        if self._motion is None:
            from ..motion import TwinMotionHandle

            self._motion = TwinMotionHandle(self)
        return self._motion

    def list_movements(
        self, scope: str = "auto", environment_uuid: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.motion.list_movements(
            scope=scope,
            environment_uuid=environment_uuid,
        )

    def run_movement(
        self,
        name: str,
        *,
        scope: str = "auto",
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        transition_ms: Optional[int] = None,
        hold_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.motion.run_movement(
            name,
            scope=scope,
            environment_uuid=environment_uuid,
            preview=preview,
            sync=sync,
            source_type=source_type,
            transition_ms=transition_ms,
            hold_ms=hold_ms,
        )

    def move_to_pose(
        self,
        name: str,
        *,
        scope: str = "auto",
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        transition_ms: Optional[int] = None,
        hold_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.motion.move_to_pose(
            name,
            scope=scope,
            environment_uuid=environment_uuid,
            preview=preview,
            sync=sync,
            source_type=source_type,
            transition_ms=transition_ms,
            hold_ms=hold_ms,
        )

    def get_latest_frame(
        self,
        sensor_id: Optional[str] = None,
        mock: bool = False,
        source_type: Optional[str] = None,
        frame_bucket: Optional[str] = None,
        _request_timeout: Any = None,
    ) -> bytes | None:
        """Fetch the latest cloud JPEG (deprecated — prefer :meth:`get_frame`).

        ``_request_timeout`` is forwarded to the HTTP client (single total
        timeout or ``(connect, read)`` tuple). Background polling callers MUST
        set it so a stalled socket cannot block the fetch thread indefinitely.
        """
        from ._helpers import _decode_frame

        warnings.warn(
            "twin.get_latest_frame() is deprecated; use twin.get_frame(source='cloud')",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            resolved_source_type = source_type
            if isinstance(resolved_source_type, str):
                _norm = resolved_source_type.strip().lower()
                if _norm in {"simulation", "sim"}:
                    resolved_source_type = "sim"
                elif _norm in {"tele", "real-world", "real", "teleoperation", "edge"}:
                    resolved_source_type = "tele"
            if resolved_source_type is None:
                client_config = getattr(self.client, "config", None)
                configured_source_type = getattr(client_config, "source_type", None)
                if isinstance(configured_source_type, str):
                    normalized = configured_source_type.strip().lower()
                    if normalized in {"sim", "simulation"}:
                        resolved_source_type = "sim"
                    elif normalized in {
                        "tele",
                        "real-world",
                        "real",
                        "teleoperation",
                        "edge",
                    }:
                        resolved_source_type = "tele"

            manager_kwargs: Dict[str, Any] = {
                "sensor_id": self._resolve_sensor_id_for_cloud_frame(sensor_id),
                "mock": mock,
            }
            if resolved_source_type in {"sim", "tele"}:
                manager_kwargs["source_type"] = resolved_source_type
            if frame_bucket:
                manager_kwargs["frame_bucket"] = frame_bucket
            if _request_timeout is not None:
                manager_kwargs["_request_timeout"] = _request_timeout

            jpeg = self.client.twins.get_latest_frame(self.uuid, **manager_kwargs)
            if jpeg is None:
                return None
            return _decode_frame(jpeg, "bytes")
        except Exception as e:
            raise CyberwaveError(
                f"Failed to get latest frame for twin {self.uuid}: {e}"
            ) from e

    def resolve_handler_from_capabilities(self, handler: str) -> "HandlerResolution":
        """Whether a grouped handle exists for this twin's capabilities."""
        from .capability_resolve import resolve_handler_from_capabilities

        return resolve_handler_from_capabilities(self.capabilities, handler)

    def _resolve_handler(self, handler: str) -> "HandlerResolution":
        """Internal alias used by sensor helpers and namespaces."""
        return self.resolve_handler_from_capabilities(handler)

    def _default_imaging_sensor_id(self) -> Optional[str]:
        return self._resolve_handler("camera").default_sensor_id

    def _default_lidar_sensor_id(self) -> Optional[str]:
        return self._resolve_handler("lidar").default_sensor_id

    def _resolve_sensor_id_for_cloud_frame(
        self, sensor_id: Optional[str]
    ) -> Optional[str]:
        """Sensor id for cloud frame APIs (first imaging sensor when omitted)."""
        return self._resolve_sensor_id(sensor_id)

    def _resolve_sensor_id(self, sensor_id: Optional[str]) -> Optional[str]:
        """Resolve imaging sensor id (defaults to first when omitted)."""
        from .capability_resolve import resolve_imaging_sensor_id

        return resolve_imaging_sensor_id(self.capabilities, sensor_id)

    def _resolve_imaging_sensor_key(self, sensor: str | None) -> str | None:
        resolution = self._resolve_handler("camera")
        sensors = list(resolution.sensor_entries)
        if not sensors:
            return None
        if sensor is None:
            s0 = sensors[0]
            return str(s0.get("role") or s0.get("name") or s0.get("id") or "default")
        for s in sensors:
            if sensor in {str(s.get("id")), str(s.get("name")), str(s.get("role"))}:
                return str(s.get("role") or s.get("name") or s.get("id"))
        raise ValueError(f"Unknown sensor {sensor!r}")

    def _imaging_handle_for_sid(self, sid: Optional[str]) -> Any:
        """Build the right imaging handle (RGB vs depth) for *sid* by sensor type."""
        from .sensors import sensor_handle_for_key
        from .sensors.camera import TwinCameraHandle

        if sid is None:
            return TwinCameraHandle(self, sensor_id=sid)
        try:
            return sensor_handle_for_key(self, sid)
        except KeyError:
            return TwinCameraHandle(self, sensor_id=sid)

    def _default_imaging_handle(self) -> Any:
        """Cached handle for the default (first) imaging sensor."""
        resolution = self._resolve_handler("camera")
        if not resolution.available:
            raise ValueError("No imaging sensor on this twin")
        sid = resolution.default_sensor_id
        handle = self._camera_handle
        if handle is None or handle._sensor_id != sid:
            handle = self._imaging_handle_for_sid(sid)
            self._camera_handle = handle
        return handle

    def _imaging_handle(
        self,
        *,
        sensor_id: Optional[str] = None,
        sensor: str | None = None,
    ) -> Any:
        """Return a per-sensor camera handle."""
        resolution = self._resolve_handler("camera")
        if not resolution.available:
            raise ValueError("No imaging sensor on this twin")
        if sensor is None and sensor_id is None:
            return self._default_imaging_handle()
        if sensor is not None:
            key = self._resolve_imaging_sensor_key(sensor)
            sid = self._resolve_sensor_id(key)
        else:
            sid = self._resolve_sensor_id(sensor_id)
        default_sid = resolution.default_sensor_id
        if sid == default_sid:
            return self._default_imaging_handle()
        return self._imaging_handle_for_sid(sid)

    def _build_sensor_family(self, attr: str) -> Any:
        from .sensors import (
            CAMERA_HANDLE_PUBLIC_METHODS,
            COMPASS_HANDLE_PUBLIC_METHODS,
            FLASHLIGHT_HANDLE_PUBLIC_METHODS,
            GPS_HANDLE_PUBLIC_METHODS,
            IMU_HANDLE_PUBLIC_METHODS,
            LIDAR_HANDLE_PUBLIC_METHODS,
            SensorFamily,
            compass_handle_for_key,
            flashlight_handle_for_key,
            gps_handle_for_key,
            imu_handle_for_key,
            lidar_handle_for_key,
            sensor_handle_for_key,
        )

        config: dict[str, tuple[Any, tuple[str, ...]]] = {
            "camera": (sensor_handle_for_key, CAMERA_HANDLE_PUBLIC_METHODS),
            "lidar": (lidar_handle_for_key, LIDAR_HANDLE_PUBLIC_METHODS),
            "gps": (gps_handle_for_key, GPS_HANDLE_PUBLIC_METHODS),
            "compass": (compass_handle_for_key, COMPASS_HANDLE_PUBLIC_METHODS),
            "imu": (imu_handle_for_key, IMU_HANDLE_PUBLIC_METHODS),
            "flashlight": (flashlight_handle_for_key, FLASHLIGHT_HANDLE_PUBLIC_METHODS),
        }
        handle_for_key, public_methods = config[attr]
        return SensorFamily(
            self,
            handler_key=attr,
            family_label=attr,
            public_methods=public_methods,
            handle_for_key=handle_for_key,
        )

    def __getattr__(self, name: str) -> Any:
        """Expose each available sensor family as an indexable ``twin.<family>``."""
        if name in _SENSOR_FAMILY_ATTRS:
            resolution = self._resolve_handler(name)
            if resolution.available:
                cache = self.__dict__.setdefault("_sensor_family_cache", {})
                family = cache.get(name)
                if family is None:
                    family = self._build_sensor_family(name)
                    cache[name] = family
                return family
        singular = _REMOVED_PLURAL_SENSOR_ATTRS.get(name)
        if singular is not None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"The plural sensor namespaces were removed; use the singular "
                f"family 'twin.{singular}' "
                f"(indexable: twin.{singular}[0], twin.{singular}['<sensor_id>'])."
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __dir__(self) -> List[str]:
        names = set(super().__dir__())
        for attr in _SENSOR_FAMILY_ATTRS:
            if self._resolve_handler(attr).available:
                names.add(attr)
            else:
                names.discard(attr)
        return sorted(names)

    def describe(self) -> Dict[str, Any]:
        """Agent contract: handles and callable methods on this twin instance.

        Driver catalogs (MQTT + optional Zenoh) are under ``driver`` — introspection
        getters and ``set_schema``. Bound MQTT commands are under ``commands`` —
        ``twin.commands.<name>(...)`` plus ``command_routing`` when present.
        """
        driver_handle = self._driver_handle
        if driver_handle is None:
            driver_handle = self.driver
        driver_info = driver_handle.describe_section()
        commands_handle = self._commands_handle
        if commands_handle is None:
            commands_handle = self.commands
        commands_info = commands_handle.describe_section()
        handles: Dict[str, Any] = {
            "driver": driver_info,
            "commands": commands_info,
        }
        flat_methods: List[str] = ["driver", "commands"]
        if hasattr(type(self), "policy"):
            handles["policy"] = {
                "methods": ["get", "assign", "attached", "ensure_attached", "keyboard"]
            }
        if hasattr(type(self), "locomotion"):
            handles["locomotion"] = {
                "methods": [
                    "move_forward",
                    "move_backward",
                    "turn_left",
                    "turn_right",
                    "stop",
                    "move",
                ]
            }
            flat_methods.extend(
                [
                    "move_forward",
                    "move_backward",
                    "turn_left",
                    "turn_right",
                    "stop",
                ]
            )
        if hasattr(type(self), "flight"):
            handles["flight"] = {"methods": ["takeoff", "land", "hover"]}
        if hasattr(type(self), "gripper"):
            handles["gripper"] = {"methods": ["grip", "release"]}
        if hasattr(type(self), "joints"):
            handles["joints"] = {"methods": ["set", "get", "list"]}
            flat_methods.extend(["get_joints", "set_joints", "get_pose", "set_pose"])
        elif hasattr(type(self), "get_pose") and not hasattr(type(self), "joints"):
            flat_methods.extend(["get_pose", "set_pose"])
        flat_methods.append("get_latest_frame")
        camera_res = self._resolve_handler("camera")
        if camera_res.available:
            flat_methods.extend(["get_frame", "get_frames"])

        from .sensors import CAMERA_HANDLE_PUBLIC_METHODS, READ_SENSOR_METHODS

        family_methods: Dict[str, tuple[str, ...]] = {
            "camera": CAMERA_HANDLE_PUBLIC_METHODS,
            **READ_SENSOR_METHODS,
        }
        for family, methods in family_methods.items():
            res = self._resolve_handler(family)
            if not res.available:
                continue
            first_id = res.sensor_ids[0]
            handles[family] = {
                "keys": list(res.sensor_ids),
                "default_sensor_id": res.default_sensor_id,
                "methods": list(methods),
                "access": [
                    f"{family}[0]",
                    f"{family}['{first_id}']",
                    f"{family}.{first_id}",
                ],
                "per_sensor": f"{family}.describe()",
            }
        handles["motion"] = {
            "methods": ["list_movements", "run_movement", "move_to_pose"],
            "scope_default": "auto",
        }
        flat_methods.extend(["list_movements", "run_movement", "move_to_pose"])
        if hasattr(type(self), "navigation"):
            handles["navigation"] = {"methods": ["goto", "stop", "follow_path"]}
        return {
            "uuid": self.uuid,
            "interfaces": {
                "driver": {
                    "access": "twin.driver",
                    "role": driver_info.get("role"),
                    "transports": driver_info.get("transports", []),
                },
                "commands": {
                    "access": "twin.commands",
                    "role": commands_info.get("role"),
                    "catalog_introspection": commands_info.get(
                        "catalog_introspection", "twin.driver"
                    ),
                },
            },
            "driver": driver_info,
            "commands": commands_info,
            "handles": handles,
            "flat_methods": sorted(set(flat_methods)),
            "class": type(self).__name__,
        }

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
    def slug(self) -> str:
        """Get the twin's unified slug (e.g. ``acme/twins/arm-station-1``)."""
        if hasattr(self._data, "slug"):
            return str(self._data.slug or "")
        if isinstance(self._data, dict):
            return str(self._data.get("slug", ""))
        return ""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Twin metadata (includes seeded ``mqtt`` command catalog from create time)."""
        return _get_twin_metadata(self._data)

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
            from ..alerts import TwinAlertManager

            self._alerts = TwinAlertManager(self)
        return self._alerts

    def refresh(self):
        """Refresh twin data from the server"""
        try:
            self._data = self.client.twins.get_raw(self.uuid)
            self._position = None
            self._rotation = None
            self._scale = None
            self._mqtt_catalog_cache = None
            self._driver_catalog_cache = None
        except Exception as e:
            raise CyberwaveError(f"Failed to refresh twin: {e}")

    def _ensure_simulation_support(self, level: "SimLevel", *, method: str) -> None:
        """Preflight for a simulation-dependent method (see ``@simulation_level``).

        No-op in live runtime mode. In simulation mode: raises
        ``NotSimulatedError`` for ``UNSUPPORTED``; no further check for
        ``PLAYGROUND``. For ``BOTH`` and ``MUJOCO`` raises
        ``SimulationNotRunningError`` when nothing is running (or the running
        instance hasn't finished starting yet); ``MUJOCO`` additionally raises
        ``SimulationLevelError`` when the running backend isn't MuJoCo. ``BOTH``
        accepts any running backend.
        """
        from .runtime_state import RUNTIME_MODE_SIMULATION, active_runtime_mode
        from .simulation_support import SimLevel, backend_sim_level

        if active_runtime_mode(self.client) != RUNTIME_MODE_SIMULATION:
            return

        if level == SimLevel.UNSUPPORTED:
            raise NotSimulatedError(
                f"{method} is not supported in simulation mode."
            )
        if level == SimLevel.PLAYGROUND:
            return

        from ..managers.simulations import running_simulation

        env_id = self.environment_id
        sim = running_simulation(self)
        if sim is None or sim.status != "running":
            status_note = (
                f" (currently {sim.status!r} — not ready yet; wait for it with "
                "sim.wait_until_active())"
                if sim is not None
                else ""
            )
            raise SimulationNotRunningError(
                f"No running simulation for environment {env_id}{status_note}.\n"
                "Start one, either:\n\n"
                f'    cw.environments.simulations.start("{env_id}", '
                'backend="mujoco", duration=300)\n\n'
                "or select a simulation runtime (which auto-starts one for the "
                "client's environment):\n\n"
                '    cw.affect("sim")   # set CYBERWAVE_ENVIRONMENT_ID=... or '
                "pass environment_id= so it knows which environment\n\n"
                "Stop it when finished with sim.stop()."
            )
        if level == SimLevel.MUJOCO and backend_sim_level(sim.backend) < SimLevel.MUJOCO:
            raise SimulationLevelError(
                f"{method} requires a MuJoCo simulation; the running backend "
                f"is {sim.backend!r}."
            )

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
        self._ensure_mqtt_connected()

    def listen(
        self,
        on_update: Callable[[Dict[str, Any]], None] | None = None,
        *,
        handlers: Mapping[str, Any] | None = None,
        filters: Sequence[str] | None = None,
        include_telemetry: bool = False,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> Any:
        """Catalog-driven multi-topic MQTT session (or dry-run spec map).

        Legacy positional ``on_update`` registers a handler on the twin wildcard topic.
        """
        from ..mqtt.listen import (
            TwinListenSession,
            build_listen_specs,
        )

        if on_update is not None:
            warnings.warn(
                "twin.subscribe(on_update) is deprecated; use twin.listen() with "
                "handlers= or subscribe_twin for wildcard updates",
                DeprecationWarning,
                stacklevel=2,
            )
            self._connect_to_mqtt_if_not_connected()
            self.client.mqtt.subscribe_twin(self.uuid, on_update)
            return None

        merged_handlers = dict(handlers or {})
        specs = build_listen_specs(
            self,
            handlers=merged_handlers,
            filters=filters,
            include_telemetry=include_telemetry,
            verbose=verbose,
        )
        if dry_run:
            return specs
        session = TwinListenSession(self, specs)
        session.start()
        return session

    def subscribe_position(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to movement updates (deprecated)."""
        warnings.warn(
            "subscribe_position() is deprecated; use twin.pose.get() or "
            "twin.listen(filters=['pose'])",
            DeprecationWarning,
            stacklevel=2,
        )
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin_position(self.uuid, on_update)

    def subscribe_rotation(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to rotation updates (deprecated)."""
        warnings.warn(
            "subscribe_rotation() is deprecated; use twin.pose.get() or "
            "twin.listen(filters=['pose'])",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """Check if the twin has sensors, optionally of a specific type.

        Uses :meth:`resolve_handler_from_capabilities` for family types
        (``lidar``, ``rgb``, ``camera``, …).
        """
        if sensor_type is None:
            return self.resolve_handler_from_capabilities("sensor").available
        t = sensor_type.lower()
        if t == "lidar" or "lidar" in t:
            return self.resolve_handler_from_capabilities("lidar").available
        if t == "gps":
            return self.resolve_handler_from_capabilities("gps").available
        if t == "compass":
            return self.resolve_handler_from_capabilities("compass").available
        if t == "imu":
            return self.resolve_handler_from_capabilities("imu").available
        if t == "flashlight":
            return self.resolve_handler_from_capabilities("flashlight").available
        if t in {"rgb", "depth", "camera", "imaging"}:
            return self.resolve_handler_from_capabilities("camera").available
        return any(
            isinstance(s, dict) and s.get("type") == sensor_type
            for s in self.capabilities.get("sensors", [])
        )

    # =========================================================================
    # Universal Schema APIs
    # =========================================================================

    def get_controllable_joint_names(self) -> List[str]:
        """
        Deprecated — use :meth:`joints.list` on joint-capable twins.

        Returns the same names as ``twin.joints.list()``.
        """

        from .capabilities.joints import controllable_joint_names

        warnings.warn(
            "twin.get_controllable_joint_names() is deprecated; use twin.joints.list()",
            DeprecationWarning,
            stacklevel=2,
        )
        return controllable_joint_names(self)

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
        """Deprecated — use :meth:`joints.calibration.get` on joint-capable twins."""

        warnings.warn(
            "twin.get_calibration() is deprecated; use twin.joints.calibration.get()",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "joints"):
            return self.joints.calibration.get(robot_type=robot_type)
        return self.client.twins.get_calibration(self.uuid, robot_type=robot_type)

    def update_calibration(
        self,
        joint_calibration: Dict[str, Dict[str, Any]],
        robot_type: str,
    ) -> "TwinJointCalibrationSchema":
        warnings.warn(
            "twin.update_calibration() is deprecated; use twin.joints.calibration.set()",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "joints"):
            return self.joints.calibration.set(joint_calibration, robot_type=robot_type)
        return self.client.twins.update_calibration(
            self.uuid, joint_calibration, robot_type
        )

    def delete_calibration(self, robot_type: Optional[str] = None) -> None:
        warnings.warn(
            "twin.delete_calibration() is deprecated; use twin.joints.calibration.delete()",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "joints"):
            return self.joints.calibration.delete(robot_type=robot_type)
        self.client.twins.delete_calibration(self.uuid, robot_type=robot_type)


# Documented alias — same implementation as :meth:`Twin.listen`.
Twin.subscribe = Twin.listen  # type: ignore[method-assign,assignment]
