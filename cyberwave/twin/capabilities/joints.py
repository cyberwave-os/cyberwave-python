"""Joints handle and nested calibration REST API."""

from __future__ import annotations

import copy
import math
import threading
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from ...exceptions import TwinStateUnavailableError
from ...data.state_representation import parse_joint_mqtt_payload
from ...manifest.driver_config import JOINT_UPDATE_TOPIC_SLUG
from ...mqtt.state import (
    FIRST_READ_TIMEOUT_S,
    attach_topic_listener,
    mqtt_client_for,
)

from .._helpers import _default_control_source_type
from ..runtime_state import (
    active_runtime_mode,
    new_runtime_ready_events,
    runtime_mode_from_mqtt_source_type,
)

if TYPE_CHECKING:
    from ..base import Twin
    from cyberwave.rest.models.twin_joint_calibration_schema import (
        TwinJointCalibrationSchema,
    )

_JOINT_DATA_KINDS = frozenset({"position", "velocity", "acceleration", "effort"})
_CONTROLLABLE_JOINT_TYPES = frozenset({"revolute", "prismatic", "continuous"})
_MQTT_SET_KEY = {
    "position": "positions",
    "velocity": "velocities",
    "acceleration": "efforts",
    "effort": "efforts",
}

# Names that must not be resolved as joint keys via ``__getattr__`` (avoids touching ``_twin`` APIs).
_JOINTS_HANDLE_RESERVED_ATTRS = frozenset(
    {"twin", "client", "uuid", "name", "capabilities", "asset_id", "environment_id"}
)


def _local_universal_schema(twin: Twin) -> dict[str, Any] | None:
    """Read universal schema from twin/asset payloads without a REST round-trip."""
    data = twin._data
    if hasattr(data, "universal_schema"):
        raw = data.universal_schema
        if isinstance(raw, dict):
            return raw
    if isinstance(data, dict):
        raw = data.get("universal_schema")
        if isinstance(raw, dict):
            return raw

    asset_id = twin.asset_id
    if not asset_id:
        return None
    try:
        asset = twin.client.assets.get(asset_id)
    except Exception:
        return None
    if hasattr(asset, "universal_schema") and isinstance(asset.universal_schema, dict):
        return asset.universal_schema
    if isinstance(asset, dict) and isinstance(asset.get("universal_schema"), dict):
        return asset["universal_schema"]
    return None


def controllable_joint_names(twin: Twin) -> List[str]:
    """Joint names from the twin universal schema (revolute, prismatic, continuous)."""
    schema = _local_universal_schema(twin)
    if not schema:
        return []
    joints = schema.get("joints", [])
    controllable = [
        j["name"]
        for j in joints
        if isinstance(j, dict)
        and j.get("name")
        and j.get("type") in _CONTROLLABLE_JOINT_TYPES
    ]
    return sorted(controllable)


def _normalize_what_data(what_data: Sequence[str]) -> List[str]:
    kinds = [str(k).strip().lower() for k in what_data if str(k).strip()]
    if not kinds:
        kinds = ["position"]
    unknown = set(kinds) - _JOINT_DATA_KINDS
    if unknown:
        raise ValueError(
            f"Unknown what_data kind(s): {sorted(unknown)}. "
            f"Use: {sorted(_JOINT_DATA_KINDS)}"
        )
    return kinds


def _resolve_joint_names(
    twin: Twin,
    *,
    what_joints: Optional[Sequence[str]] = None,
) -> List[str]:
    names = controllable_joint_names(twin)
    if what_joints is None:
        return names
    requested = list(what_joints)
    unknown = set(requested) - set(names)
    if names and unknown:
        raise ValueError(
            f"Unknown joint name(s): {sorted(unknown)}. "
            f"Controllable: {names}"
        )
    return requested


class JointsCalibrationHandle:
    """Joint calibration metadata (REST only — no outbound MQTT gate)."""

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def get(self, robot_type: Optional[str] = None) -> Any:
        return self._twin.client.twins.get_calibration(self._twin.uuid, robot_type=robot_type)

    def set(
        self,
        joint_calibration: "TwinJointCalibrationSchema | dict[str, Any]",
        robot_type: str,
    ) -> Any:
        return self._twin.client.twins.update_calibration(
            self._twin.uuid, joint_calibration, robot_type
        )

    def delete(self, robot_type: Optional[str] = None) -> None:
        self._twin.client.twins.delete_calibration(self._twin.uuid, robot_type=robot_type)


class JointsHandle:
    """Grouped joint state — background MQTT listener keeps ``_curr_joints`` fresh."""

    def __init__(self, twin: Twin) -> None:
        object.__setattr__(self, "_twin", twin)
        object.__setattr__(self, "_curr_joints_by_mode", {})
        object.__setattr__(self, "_joint_attached_topics", set())
        object.__setattr__(self, "_joint_listeners_attached", False)
        object.__setattr__(self, "_received_mqtt_by_mode", new_runtime_ready_events())
        object.__setattr__(self, "_joint_lock", threading.Lock())
        object.__setattr__(self, "_calibration", JointsCalibrationHandle(twin))
        self._start_listening()

    @property
    def calibration(self) -> JointsCalibrationHandle:
        return self._calibration

    def list(self) -> List[str]:
        return controllable_joint_names(self._twin)

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _active_runtime_mode(self) -> str:
        return active_runtime_mode(self._twin.client)

    def _zero_joint_entry(self) -> Dict[str, float]:
        return {"position": 0.0, "velocity": 0.0, "acceleration": 0.0, "effort": 0.0}

    def _ensure_curr_joints(self, runtime_mode: str | None = None) -> Dict[str, Dict[str, float]]:
        mode = runtime_mode or self._active_runtime_mode()
        if mode not in self._curr_joints_by_mode:
            self._curr_joints_by_mode[mode] = {
                name: self._zero_joint_entry() for name in self.list()
            }
        return self._curr_joints_by_mode[mode]

    def _merge_list_joints_into(self, curr: Dict[str, Dict[str, float]]) -> None:
        """Ensure every schema joint from :meth:`list` exists in *curr* (missing → 0)."""
        for name in self.list():
            if name not in curr:
                curr[name] = self._zero_joint_entry()

    def _on_joint_payload(self, payload: dict[str, Any]) -> None:
        """MQTT callback — runs on every ``/update`` message while subscribed."""
        mode = runtime_mode_from_mqtt_source_type(
            payload.get("source_type"),
            client=self._twin.client,
        )
        names = frozenset(self.list())
        batch = parse_joint_mqtt_payload(payload, controllable_names=names)
        if not batch.positions and not batch.velocities and not batch.efforts:
            return
        with self._joint_lock:
            curr = self._ensure_curr_joints(mode)
            for name, val in batch.positions.items():
                if name not in curr:
                    curr[name] = self._zero_joint_entry()
                curr[name]["position"] = val
            for name, val in batch.velocities.items():
                if name not in curr:
                    curr[name] = self._zero_joint_entry()
                curr[name]["velocity"] = val
            for name, val in batch.efforts.items():
                if name not in curr:
                    curr[name] = self._zero_joint_entry()
                curr[name]["effort"] = val
                curr[name]["acceleration"] = val
            self._merge_list_joints_into(curr)
            self._received_mqtt_by_mode[mode].set()

    def _joint_update_topic(self) -> str:
        """MQTT topic for joint state — always ``cyberwave/joint/{uuid}/update``."""
        return (
            f"{self._topic_prefix()}"
            f"{JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid=self._twin.uuid)}"
        )

    def _start_listening(self) -> None:
        """Subscribe once to ``/joint/.../update``; callback keeps ``_curr_joints`` updated."""
        if self._joint_listeners_attached:
            return
        try:
            mqtt_client_for(self._twin)
        except TwinStateUnavailableError:
            return
        topic = self._joint_update_topic()
        attach_topic_listener(
            self._twin,
            topic=topic,
            on_payload=self._on_joint_payload,
            attached_topics=self._joint_attached_topics,
        )
        self._joint_listeners_attached = True

    def _await_initial_joint_state(self, *, timeout: float) -> None:
        """Wait for the first joint MQTT update for the active runtime mode.

        On timeout, seeds zeros from :meth:`list` for that mode only.
        """
        mode = self._active_runtime_mode()
        if self._received_mqtt_by_mode[mode].is_set():
            return
        with self._joint_lock:
            if mode in self._curr_joints_by_mode:
                return
        if not self._received_mqtt_by_mode[mode].wait(timeout=timeout):
            with self._joint_lock:
                self._ensure_curr_joints(mode)
                self._received_mqtt_by_mode[mode].set()

    def get(
        self,
        *,
        what_joints: Optional[Sequence[str]] = None,
        what_data: Sequence[str] = ("position",),
        timeout: float = FIRST_READ_TIMEOUT_S,
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Return the latest joint state (background listener updates between calls).

        *what_data* selects which fields to return per joint: ``position`` (default),
        ``velocity``, ``acceleration``, and ``effort`` (effort/torque). A single kind
        yields ``{joint_name: value}``; multiple kinds yield
        ``{kind: {joint_name: value}}``.

        If no MQTT joint update arrives within *timeout* (default 3s), returns every
        controllable joint from :meth:`list` with zero values instead of raising.
        Partial MQTT updates are merged so any schema joint missing from the payload
        is filled with 0. Reads/writes use :attr:`~cyberwave.config.CyberwaveConfig.runtime_mode`
        (``live`` vs ``simulation``); inbound MQTT is bucketed by ``source_type``.

        Prefer :meth:`~cyberwave.twin.mixins.JointsCapableMixin.get_joints` /
        :meth:`~cyberwave.twin.mixins.JointsCapableMixin.set_joints` on the twin for
        the same behavior via stable shortcuts.
        """
        kinds = _normalize_what_data(what_data)
        names = _resolve_joint_names(self._twin, what_joints=what_joints)
        mode = self._active_runtime_mode()
        self._start_listening()
        self._await_initial_joint_state(timeout=timeout)
        with self._joint_lock:
            curr = copy.deepcopy(self._ensure_curr_joints(mode))
            self._merge_list_joints_into(curr)

        if len(kinds) == 1 and kinds[0] == "position":
            return {name: curr.get(name, {}).get("position", 0.0) for name in names}

        result: Dict[str, Dict[str, float]] = {}
        for kind in kinds:
            result[kind] = {name: curr.get(name, {}).get(kind, 0.0) for name in names}
        return result

    def set(
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
        """Publish joint command(s) on the catalog joint-update topic.

        *what_data* is one of ``position``, ``velocity``, ``acceleration``, or
        ``effort`` (effort/torque). Twin shortcuts :meth:`~cyberwave.twin.mixins.JointsCapableMixin.set_joints`
        / :meth:`~cyberwave.twin.mixins.JointsCapableMixin.set_pose` delegate here.
        """
        if isinstance(values, str):
            if position is None:
                raise TypeError(
                    "joints.set(joint_name, value) requires a numeric second argument"
                )
            joint = values
            values = float(position)

        kind = str(what_data).strip().lower()
        if kind not in _JOINT_DATA_KINDS:
            raise ValueError(
                f"Unknown what_data {what_data!r}. Use: {sorted(_JOINT_DATA_KINDS)}"
            )

        if isinstance(values, (int, float)):
            if joint is None:
                raise ValueError("joint name required when values is a scalar")
            payload_values = {joint: float(values)}
        elif isinstance(values, Mapping):
            payload_values = dict(values)
        else:
            raise TypeError(
                "joints.set() expects a mapping {joint_name: value}, "
                f"or joints.set(joint_name, value); got {type(values).__name__}"
            )

        if what_joints is not None:
            allowed = set(_resolve_joint_names(self._twin, what_joints=what_joints))
            payload_values = {k: v for k, v in payload_values.items() if k in allowed}

        controllable = set(self.list())
        unknown = set(payload_values) - controllable
        if controllable and unknown:
            raise ValueError(
                f"Unknown joint name(s): {sorted(unknown)}. "
                f"Controllable: {sorted(controllable)}"
            )

        if degrees and kind == "position":
            payload_values = {
                name: math.radians(val) for name, val in payload_values.items()
            }

        if source_type is None:
            source_type = _default_control_source_type(self._twin.client)

        mqtt_key = _MQTT_SET_KEY[kind]
        data: Dict[str, Any] = {"mode": mode, mqtt_key: payload_values, "timestamp": timestamp}

        resolved = self._twin._resolve_topic_and_payload(
            command="joint_update",
            data=data,
            channel="joint_update",
            source_type=source_type,
        )
        self._twin._publish_resolved(resolved)

        with self._joint_lock:
            curr = self._ensure_curr_joints(self._active_runtime_mode())
            for name, val in payload_values.items():
                if name not in curr:
                    curr[name] = self._zero_joint_entry()
                curr[name][kind] = val

        self._start_listening()

    def set_joints(
        self,
        positions: Mapping[str, float] | float,
        *,
        joint: Optional[str] = None,
        mode: str = "absolute",
        degrees: bool = False,
        source_type: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        warnings.warn(
            "joints.set_joints() is deprecated; use joints.set(..., what_data='position')",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set(
            positions,
            joint=joint,
            what_data="position",
            mode=mode,
            degrees=degrees,
            source_type=source_type,
            timestamp=timestamp,
        )

    def get_all(self) -> Dict[str, float]:
        warnings.warn(
            "joints.get_all() is deprecated; use joints.get()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get()

    def _resolve_joint_key(self, key: str | int) -> str:
        """Map integer index (from :meth:`list` order) or name to a joint name."""
        if isinstance(key, int):
            names = self.list()
            if not names:
                raise IndexError("no controllable joints on this twin")
            index = key + len(names) if key < 0 else key
            if index < 0 or index >= len(names):
                raise IndexError(
                    f"joint index {key} out of range for {len(names)} joint(s)"
                )
            return names[index]
        return str(key)

    def __getitem__(self, key: str | int) -> float:
        name = self._resolve_joint_key(key)
        return self.get(what_joints=[name], what_data=["position"])[name]

    def __setitem__(self, key: str | int, value: float) -> None:
        self.set(value, joint=self._resolve_joint_key(key))

    def __getattr__(self, name: str) -> float:
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name in _JOINTS_HANDLE_RESERVED_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name not in self.list():
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return self.get(what_joints=[name], what_data=["position"])[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            "_twin",
            "_curr_joints_by_mode",
            "_joint_attached_topics",
            "_joint_listeners_attached",
            "_received_mqtt_by_mode",
            "_joint_lock",
            "_calibration",
        ):
            super().__setattr__(name, value)
        else:
            self.set(value, joint=name)
