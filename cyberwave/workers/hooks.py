"""Hook registry and typed decorator factories for channel-driven worker callbacks."""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HookRegistration:
    """One registered hook binding a callback to a channel on a twin."""

    channel: str
    twin_uuid: str
    callback: Callable[..., Any]
    hook_type: str
    sensor_name: str = "default"
    options: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        opts = f", options={self.options}" if self.options else ""
        return (
            f"HookRegistration(channel={self.channel!r}, "
            f"twin_uuid={self.twin_uuid!r}, "
            f"hook_type={self.hook_type!r}, "
            f"sensor_name={self.sensor_name!r}{opts})"
        )


@dataclass(frozen=True)
class SynchronizedGroup:
    """Describes a set of channels to synchronize before dispatching."""

    channels: tuple[str, ...]
    twin_uuid: str
    callback: Callable[..., Any]
    tolerance_ms: float = 50.0
    options: dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """Collects hook registrations.  One instance per ``Cyberwave`` client."""

    def __init__(self) -> None:
        self._hooks: list[HookRegistration] = []
        self._synchronized: list[SynchronizedGroup] = []
        self._lock = threading.Lock()

    @property
    def hooks(self) -> list[HookRegistration]:
        with self._lock:
            return list(self._hooks)

    @property
    def synchronized_groups(self) -> list[SynchronizedGroup]:
        with self._lock:
            return list(self._synchronized)

    def register(self, registration: HookRegistration) -> None:
        with self._lock:
            for existing in self._hooks:
                if (
                    existing.callback is registration.callback
                    and existing.channel == registration.channel
                    and existing.twin_uuid == registration.twin_uuid
                ):
                    warnings.warn(
                        f"Duplicate hook registration: callback "
                        f"{registration.callback.__qualname__!r} is already "
                        f"registered on channel {registration.channel!r} for "
                        f"twin {registration.twin_uuid!r}",
                        stacklevel=3,
                    )
                    break
            self._hooks.append(registration)

    def clear(self) -> None:
        with self._lock:
            self._hooks.clear()
            self._synchronized.clear()

    # ── Private decorator factory ────────────────────────────────

    def _make_decorator(
        self,
        channel: str,
        hook_type: str,
        twin_uuid: str,
        sensor_name: str = "default",
        **options: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        filtered_options = {k: v for k, v in options.items() if v is not None}

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                HookRegistration(
                    channel=channel,
                    twin_uuid=twin_uuid,
                    callback=fn,
                    hook_type=hook_type,
                    sensor_name=sensor_name,
                    options=filtered_options,
                )
            )
            return fn

        return decorator

    # ── Single-channel typed decorators ──────────────────────────

    def on_frame(
        self,
        twin_uuid: str,
        *,
        sensor: str = "default",
        fps: int | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator(
            f"frames/{sensor}", "frame", twin_uuid, sensor, fps=fps
        )

    def on_depth(
        self, twin_uuid: str, *, sensor: str = "default"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator(f"depth/{sensor}", "depth", twin_uuid, sensor)

    def on_audio(
        self, twin_uuid: str, *, sensor: str = "default"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator(f"audio/{sensor}", "audio", twin_uuid, sensor)

    def on_pointcloud(
        self, twin_uuid: str, *, sensor: str = "default"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator(
            f"pointcloud/{sensor}", "pointcloud", twin_uuid, sensor
        )

    def on_imu(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("imu", "imu", twin_uuid)

    def on_force_torque(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("force_torque", "force_torque", twin_uuid)

    def on_joint_states(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("joint_states", "joint_states", twin_uuid)

    def on_attitude(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("attitude", "attitude", twin_uuid)

    def on_gps(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("gps", "gps", twin_uuid)

    def on_end_effector_pose(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("end_effector_pose", "end_effector_pose", twin_uuid)

    def on_gripper_state(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("gripper_state", "gripper_state", twin_uuid)

    def on_map(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("map", "map", twin_uuid)

    def on_battery(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("battery", "battery", twin_uuid)

    def on_temperature(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("temperature", "temperature", twin_uuid)

    def on_lidar(
        self, twin_uuid: str, *, sensor: str = "default"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator(f"lidar/{sensor}", "lidar", twin_uuid, sensor)

    # ── Generic channel decorator ────────────────────────────────

    def on_data(
        self, twin_uuid: str, channel: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Generic hook for any custom data channel."""
        return self._make_decorator(channel, "data", twin_uuid)

    # ── Multi-channel synchronized decorator ─────────────────────

    def on_synchronized(
        self,
        twin_uuid: str,
        channels: list[str],
        *,
        tolerance_ms: float = 50.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a callback that fires when all *channels* have
        a sample within *tolerance_ms* of each other.

        Dispatch logic is implemented in a follow-up issue; this
        method only records the registration.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            with self._lock:
                self._synchronized.append(
                    SynchronizedGroup(
                        channels=tuple(channels),
                        twin_uuid=twin_uuid,
                        callback=fn,
                        tolerance_ms=tolerance_ms,
                    )
                )
            return fn

        return decorator
