"""Flight command handle for flying twins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from ...constants import SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE
from ...exceptions import CyberwaveError

from .._helpers import _default_control_source_type, _normalize_locomotion_source_type
from ..simulation_support import SimLevel, simulation_level

if TYPE_CHECKING:
    from ..base import Twin


class FlightHandle:
    """Grouped flight/drone commands."""

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    @simulation_level(SimLevel.PLAYGROUND)
    def takeoff(self, *, source_type: Optional[str] = None, **kwargs: Any) -> None:
        self._send("takeoff", kwargs, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def land(self, *, source_type: Optional[str] = None, **kwargs: Any) -> None:
        self._send("land", kwargs, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def hover(self, *, source_type: Optional[str] = None, **kwargs: Any) -> None:
        self._send("hover", kwargs, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def ascend(self, distance: float, *, source_type: Optional[str] = None) -> None:
        self._send("ascend", {"distance": distance}, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def descend(self, distance: float, *, source_type: Optional[str] = None) -> None:
        self._send("descend", {"distance": distance}, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def gimbal_rotate(
        self,
        *,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None,
        mode: str = "absolute",
        duration: Optional[float] = None,
        source_type: Optional[str] = None,
    ) -> None:
        data: Dict[str, Any] = {"mode": mode}
        if pitch is not None:
            data["pitch"] = float(pitch)
        if roll is not None:
            data["roll"] = float(roll)
        if yaw is not None:
            data["yaw"] = float(yaw)
        if duration is not None:
            data["duration"] = float(duration)
        self._send("gimbal_rotate", data, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def _send(
        self,
        command: str,
        data: Dict[str, Any],
        *,
        source_type: Optional[str] = None,
    ) -> None:
        if source_type is None:
            source_type = _default_control_source_type(self._twin.client)
        source_type = _normalize_locomotion_source_type(source_type)
        if source_type not in {SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE}:
            raise ValueError(f"Invalid source type {source_type!r} for flight command")
        resolved = self._twin._resolve_topic_and_payload(
            command=command,
            data=data,
            source_type=source_type,
        )
        self._twin._publish_resolved(resolved)
