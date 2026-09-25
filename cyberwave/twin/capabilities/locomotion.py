"""Locomotion command handle for locomote-capable twins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..simulation_support import SimLevel, simulation_level
from ..transport import DEFAULT_BURST_DURATION_S, DEFAULT_BURST_RATE_HZ

if TYPE_CHECKING:
    from ..base import Twin


class LocomotionHandle:
    """Grouped locomotion commands (MQTT outbound)."""

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    @simulation_level(SimLevel.PLAYGROUND)
    def move_forward(
        self,
        distance: float = 0.3,
        *,
        duration: float = DEFAULT_BURST_DURATION_S,
        rate_hz: float = DEFAULT_BURST_RATE_HZ,
        source_type: Optional[str] = None,
    ) -> None:
        """Move forward at *distance* m/s for *duration* seconds, then ``stop``.

        The first argument is **linear speed in m/s** (same as keyboard teleop and
        edge drivers), not a travel distance in metres.
        """
        self._twin.publish_command_burst(
            "move_forward",
            {"linear_x": abs(distance), "angular_z": 0.0},
            duration_s=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    @simulation_level(SimLevel.PLAYGROUND)
    def move_backward(
        self,
        distance: float = 0.3,
        *,
        duration: float = DEFAULT_BURST_DURATION_S,
        rate_hz: float = DEFAULT_BURST_RATE_HZ,
        source_type: Optional[str] = None,
    ) -> None:
        """Move backward at *distance* m/s for *duration* seconds, then ``stop``."""
        self._twin.publish_command_burst(
            "move_backward",
            {"linear_x": -abs(distance), "angular_z": 0.0},
            duration_s=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    @simulation_level(SimLevel.PLAYGROUND)
    def turn_left(
        self,
        angle: float = 0.5,
        *,
        duration: float = DEFAULT_BURST_DURATION_S,
        rate_hz: float = DEFAULT_BURST_RATE_HZ,
        source_type: Optional[str] = None,
    ) -> None:
        """Turn left at *angle* rad/s for *duration* seconds, then ``stop``."""
        self._twin.publish_command_burst(
            "turn_left",
            {"linear_x": 0.0, "angular_z": abs(angle)},
            duration_s=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    @simulation_level(SimLevel.PLAYGROUND)
    def turn_right(
        self,
        angle: float = 0.5,
        *,
        duration: float = DEFAULT_BURST_DURATION_S,
        rate_hz: float = DEFAULT_BURST_RATE_HZ,
        source_type: Optional[str] = None,
    ) -> None:
        """Turn right at *angle* rad/s for *duration* seconds, then ``stop``."""
        self._twin.publish_command_burst(
            "turn_right",
            {"linear_x": 0.0, "angular_z": abs(angle)},
            duration_s=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    @simulation_level(SimLevel.PLAYGROUND)
    def stop(self, *, source_type: Optional[str] = None) -> None:
        self._twin.publish_command("stop", {}, source_type=source_type)

    @simulation_level(SimLevel.PLAYGROUND)
    def move(
        self,
        *,
        distance: float | None = None,
        angle: float | None = None,
        linear_x: float | None = None,
        angular_z: float | None = None,
        source_type: Optional[str] = None,
        command: str = "move",
        duration: float = 0.0,
        rate_hz: float = DEFAULT_BURST_RATE_HZ,
    ) -> None:
        """Publish a custom locomotion command (optional burst when ``duration`` > 0)."""
        data: dict[str, float] = {}
        if linear_x is not None:
            data["linear_x"] = linear_x
        elif distance is not None:
            data["linear_x"] = distance
        if angular_z is not None:
            data["angular_z"] = angular_z
        elif angle is not None:
            data["angular_z"] = angle
        if distance is not None and angle is None and angular_z is None:
            data.setdefault("angular_z", 0.0)
        if duration > 0:
            self._twin.publish_command_burst(
                command,
                data,
                duration_s=duration,
                rate_hz=rate_hz,
                source_type=source_type,
            )
        else:
            self._twin.publish_command(command, data, source_type=source_type)
