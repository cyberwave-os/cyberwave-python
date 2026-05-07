"""Hook registry and typed decorator factories for channel-driven worker callbacks."""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


_HOOK_TYPE_CONTENT_HINTS: dict[str, str] = {
    "frame": "numpy",
    "depth": "numpy",
    "pointcloud": "numpy",
    "audio": "numpy",
    "lidar": "numpy",
}
"""Map hook types that always carry numpy payloads so decode can skip JSON/bytes attempts."""


WILDCARD_SENSOR: str = "*"
"""Marker used in :attr:`HookRegistration.sensor_name` when the hook subscribes
to every sensor under a channel (``frames/*``, ``depth/*``, …).

Drivers publish under the sensor name declared in the twin's asset (e.g.
``color_camera``, ``depth_camera``).  A hook with a wildcard sensor matches
any of those and keeps single-camera twins working without the user having
to know the declared sensor name.
"""


SENSOR_BEARING_CHANNELS: frozenset[str] = frozenset(
    {"frames", "depth", "audio", "pointcloud", "lidar"}
)
"""Channels whose published keys always include a trailing ``/<sensor>``
segment (drivers take that segment from the twin asset).  Used by
:class:`WorkerRuntime` and :class:`HookRegistry` to expand bare channel
names (e.g. ``"frames"``) into wildcard subscriptions (``frames/**``)
so authors don't have to know the sensor name for single-sensor twins.
"""


@dataclass(frozen=True)
class HookRegistration:
    """One registered hook binding a callback to a channel on a twin.

    ``channel`` holds the hook-level channel name.  For sensor channels it
    is either ``"frames/<sensor>"`` (specific sensor) or just ``"frames"``
    (wildcard — matches any sensor on the twin).  ``sensor_name`` is the
    literal sensor qualifier or :data:`WILDCARD_SENSOR` for the wildcard
    form.
    """

    channel: str
    twin_uuid: str
    callback: Callable[..., Any]
    hook_type: str
    sensor_name: str = ""
    """Literal sensor qualifier, :data:`WILDCARD_SENSOR` for a wildcard
    hook, or empty string for channels with no sensor component (``imu``,
    ``joint_states``, …).
    """
    options: dict[str, Any] = field(default_factory=dict)
    content_hint: str = ""
    """``"numpy"`` for binary sensor channels, empty for auto-detect."""

    @property
    def is_wildcard_sensor(self) -> bool:
        """True if this hook matches every sensor under ``channel``."""
        return self.sensor_name == WILDCARD_SENSOR

    def __repr__(self) -> str:
        opts = f", options={self.options}" if self.options else ""
        return (
            f"HookRegistration(channel={self.channel!r}, "
            f"twin_uuid={self.twin_uuid!r}, "
            f"hook_type={self.hook_type!r}, "
            f"sensor_name={self.sensor_name!r}{opts})"
        )


@dataclass(frozen=True)
class ScheduleRegistration:
    """One registered cron schedule callback."""

    cron: str
    timezone: str
    callback: Callable[..., Any]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynchronizedGroup:
    """Describes a set of channels to synchronize before dispatching.

    For single-twin hooks ``twin_channels`` is empty and ``channels``
    contains channel names scoped to ``twin_uuid``.

    For cross-twin hooks ``twin_channels`` is a tuple of
    ``(label, twin_uuid, channel)`` triples and ``channels`` contains
    the corresponding labels.
    """

    channels: tuple[str, ...]
    twin_uuid: str
    callback: Callable[..., Any]
    tolerance_ms: float = 50.0
    options: dict[str, Any] = field(default_factory=dict)
    twin_channels: tuple[tuple[str, str, str], ...] = ()


class HookRegistry:
    """Collects hook registrations.  One instance per ``Cyberwave`` client."""

    def __init__(self) -> None:
        self._hooks: list[HookRegistration] = []
        self._schedules: list[ScheduleRegistration] = []
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

    @property
    def schedule_hooks(self) -> list[ScheduleRegistration]:
        with self._lock:
            return list(self._schedules)

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
            self._schedules.clear()
            self._synchronized.clear()

    # ── Private decorator factory ────────────────────────────────

    def _make_decorator(
        self,
        channel: str,
        hook_type: str,
        twin_uuid: str,
        sensor_name: str = "",
        **options: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        filtered_options = {k: v for k, v in options.items() if v is not None}
        hint = _HOOK_TYPE_CONTENT_HINTS.get(hook_type, "")

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                HookRegistration(
                    channel=channel,
                    twin_uuid=twin_uuid,
                    callback=fn,
                    hook_type=hook_type,
                    sensor_name=sensor_name,
                    options=filtered_options,
                    content_hint=hint,
                )
            )
            return fn

        return decorator

    # ── Single-channel typed decorators ──────────────────────────

    @staticmethod
    def _resolve_sensor(sensor: str | None) -> str:
        """Normalise the user-facing ``sensor=`` kwarg.

        ``None`` or ``"*"`` mean "match any sensor under this channel" and
        are stored as :data:`WILDCARD_SENSOR` on the registration.  Any
        explicit name is passed through verbatim.
        """
        if sensor is None or sensor == WILDCARD_SENSOR:
            return WILDCARD_SENSOR
        return sensor

    def _sensor_channel(self, base: str, sensor_name: str) -> str:
        """Return the hook-level channel for a sensor-bearing hook.

        A wildcard hook is stored as the base channel alone (``"frames"``)
        so it stays greppable and round-trips cleanly through logs and
        stats; specific sensors keep the compound ``"base/<sensor>"`` form.
        """
        if sensor_name == WILDCARD_SENSOR:
            return base
        return f"{base}/{sensor_name}"

    def on_frame(
        self,
        twin_uuid: str,
        *,
        sensor: str | None = None,
        fps: int | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Hook a callback to camera frames on *twin_uuid*.

        Without ``sensor=``, the hook matches every camera under the twin
        (``frames/*``).  That keeps single-camera twins working regardless
        of the sensor name declared in the twin's asset (e.g.
        ``color_camera``).  Pass ``sensor="color_camera"`` (or any other
        name from the twin's sensor list) to narrow to one camera on
        multi-camera twins.
        """
        sensor_name = self._resolve_sensor(sensor)
        channel = self._sensor_channel("frames", sensor_name)
        return self._make_decorator(channel, "frame", twin_uuid, sensor_name, fps=fps)

    def on_depth(
        self, twin_uuid: str, *, sensor: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        sensor_name = self._resolve_sensor(sensor)
        channel = self._sensor_channel("depth", sensor_name)
        return self._make_decorator(channel, "depth", twin_uuid, sensor_name)

    def on_audio(
        self, twin_uuid: str, *, sensor: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        sensor_name = self._resolve_sensor(sensor)
        channel = self._sensor_channel("audio", sensor_name)
        return self._make_decorator(channel, "audio", twin_uuid, sensor_name)

    def on_pointcloud(
        self, twin_uuid: str, *, sensor: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        sensor_name = self._resolve_sensor(sensor)
        channel = self._sensor_channel("pointcloud", sensor_name)
        return self._make_decorator(channel, "pointcloud", twin_uuid, sensor_name)

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

    def on_alert(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("alert", "alert", twin_uuid)

    def on_temperature(
        self, twin_uuid: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._make_decorator("temperature", "temperature", twin_uuid)

    def on_lidar(
        self, twin_uuid: str, *, sensor: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        sensor_name = self._resolve_sensor(sensor)
        channel = self._sensor_channel("lidar", sensor_name)
        return self._make_decorator(channel, "lidar", twin_uuid, sensor_name)

    # ── Generic channel decorator ────────────────────────────────

    def on_data(
        self, twin_uuid: str, channel: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Generic hook for any custom data channel."""
        return self._make_decorator(channel, "data", twin_uuid)

    # ── Schedule decorator ───────────────────────────────────────

    def on_schedule(
        self,
        cron: str,
        *,
        timezone: str = "UTC",
        **options: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a callback that fires when *cron* is due.

        The callback receives a :class:`HookContext` with
        ``channel="schedule"`` and schedule metadata. Cron parsing is
        handled by the worker runtime through ``croniter``.
        """
        filtered_options = {k: v for k, v in options.items() if v is not None}

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            registration = ScheduleRegistration(
                cron=cron,
                timezone=timezone,
                callback=fn,
                options=filtered_options,
            )
            with self._lock:
                for existing in self._schedules:
                    if (
                        existing.callback is fn
                        and existing.cron == cron
                        and existing.timezone == timezone
                    ):
                        warnings.warn(
                            f"Duplicate schedule hook registration: callback "
                            f"{fn.__qualname__!r} is already registered for "
                            f"cron {cron!r} in timezone {timezone!r}",
                            stacklevel=3,
                        )
                        return fn
                self._schedules.append(registration)
            return fn

        return decorator

    # ── Multi-channel synchronized decorator ─────────────────────

    def on_synchronized(
        self,
        twin_uuid: str = "",
        channels: list[str] | None = None,
        *,
        twin_channels: dict[str, tuple[str, str]] | None = None,
        tolerance_ms: float = 50.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a callback that fires when all channels have a sample
        within *tolerance_ms* of each other.

        **Single-twin mode** (backward-compatible)::

            @cw.on_synchronized(twin_uuid, ["frames/front", "depth/default"])

        **Cross-twin mode** — each label maps to a ``(twin_uuid, channel)``
        pair::

            @cw.on_synchronized(twin_channels={
                "left": (TWIN_A, "frames/default"),
                "right": (TWIN_B, "frames/default"),
            })

        The callback receives ``(samples: dict[str, Sample], ctx: HookContext)``
        where *samples* maps channel name (or label) to the most recent sample.

        **Wildcard sensors** — for sensor-bearing channels
        (``frames``/``depth``/``audio``/``pointcloud``/``lidar``) a bare
        channel name or a ``/*`` suffix subscribes to every sensor the
        twin exposes, matching :meth:`on_frame`'s default.  Authors can
        write::

            @cw.on_synchronized(twin_uuid, ["frames", "depth"])

        without knowing the driver's declared sensor names.  Pin a
        specific sensor by using the full form (``"frames/front"``).
        Sensor-less channels (``joint_states``, ``imu``, …) stay
        exact-match.

        Duplicate registrations (same callback + channels) emit a warning and
        are skipped.
        """
        if channels is None and twin_channels is None:
            raise TypeError(
                "on_synchronized() requires either positional 'channels' or "
                "keyword 'twin_channels'."
            )

        if channels is not None and twin_channels is not None:
            raise TypeError(
                "on_synchronized() accepts 'channels' or 'twin_channels', not both."
            )

        if twin_channels is not None:
            labels = tuple(twin_channels.keys())
            tc_tuples = tuple(
                (label, t_uuid, ch) for label, (t_uuid, ch) in twin_channels.items()
            )

            def decorator_cross(fn: Callable[..., Any]) -> Callable[..., Any]:
                group = SynchronizedGroup(
                    channels=labels,
                    twin_uuid="",
                    callback=fn,
                    tolerance_ms=tolerance_ms,
                    twin_channels=tc_tuples,
                )
                with self._lock:
                    for existing in self._synchronized:
                        if (
                            existing.callback is fn
                            and existing.twin_channels == group.twin_channels
                        ):
                            warnings.warn(
                                f"Duplicate synchronized hook registration: callback "
                                f"{fn.__qualname__!r} is already registered with the same "
                                f"twin_channels configuration",
                                stacklevel=3,
                            )
                            return fn
                    self._synchronized.append(group)
                return fn

            return decorator_cross

        assert channels is not None

        def decorator_single(fn: Callable[..., Any]) -> Callable[..., Any]:
            group = SynchronizedGroup(
                channels=tuple(channels),
                twin_uuid=twin_uuid,
                callback=fn,
                tolerance_ms=tolerance_ms,
            )
            with self._lock:
                for existing in self._synchronized:
                    if (
                        existing.callback is fn
                        and existing.channels == group.channels
                        and existing.twin_uuid == twin_uuid
                    ):
                        warnings.warn(
                            f"Duplicate synchronized hook registration: callback "
                            f"{fn.__qualname__!r} is already registered on channels "
                            f"{list(channels)!r} for twin {twin_uuid!r}",
                            stacklevel=3,
                        )
                        return fn
                self._synchronized.append(group)
            return fn

        return decorator_single
