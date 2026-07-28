"""JointController — forward control with optional motion generation.

Pure math, no I/O. ``after_process`` is carried for the mixin (which invokes it
per produced command); ``velocity_limits`` / ``position_limits`` are lazy
provider callables — the opportunistic-kinematics seam. Providers are called at
most once and any failure or empty result means "no limits available".
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace

from .motion import trapezoidal_motion
from .types import (
    JointCommand,
    JointControllerConfig,
    MotionGenerator,
    MotionRequest,
    TrajectoryPlan,
    TrajectoryWaypoint,
)

logger = logging.getLogger(__name__)

_UNSET = object()


def waypoint_command(
    wp: TrajectoryWaypoint, config: JointControllerConfig
) -> JointCommand:
    """Waypoint -> JointCommand, populating only declared command interfaces."""
    velocities = (
        dict(wp.velocities)
        if wp.velocities is not None and "velocity" in config.command_interfaces
        else None
    )
    return JointCommand(positions=dict(wp.positions), velocities=velocities)


class JointController:
    """Config-driven joint command shaping. Computes plans, never applies them."""

    def __init__(
        self,
        config: JointControllerConfig,
        *,
        motion: MotionGenerator | None = trapezoidal_motion,
        after_process: Callable[[JointCommand], None] | None = None,
        velocity_limits: Callable[[], Mapping[str, float]] | None = None,
        position_limits: Callable[[], Mapping[str, tuple[float, float]]] | None = None,
    ) -> None:
        self._config = config
        self._motion = motion
        self.after_process = after_process
        self._velocity_limits_provider = velocity_limits
        self._position_limits_provider = position_limits
        self._velocity_limits_cache: Mapping[str, float] | object = _UNSET
        self._position_limits_cache: Mapping[str, tuple[float, float]] | object = _UNSET

    @property
    def config(self) -> JointControllerConfig:
        return self._config

    # ── lazy limit providers (opportunistic kinematics) ──────────────────────

    def _provider_velocity_limits(self) -> Mapping[str, float]:
        if self._velocity_limits_cache is _UNSET:
            self._velocity_limits_cache = self._call_provider(
                self._velocity_limits_provider
            )
        return self._velocity_limits_cache  # type: ignore[return-value]

    def _provider_position_limits(self) -> Mapping[str, tuple[float, float]]:
        if self._position_limits_cache is _UNSET:
            self._position_limits_cache = self._call_provider(
                self._position_limits_provider
            )
        return self._position_limits_cache  # type: ignore[return-value]

    @staticmethod
    def _call_provider(provider: Callable[[], Mapping] | None) -> Mapping:
        if provider is None:
            return {}
        try:
            return provider() or {}
        except Exception:
            logger.warning("joint limits provider failed; using defaults", exc_info=True)
            return {}

    # ── plan pipeline ─────────────────────────────────────────────────────────

    def plan(
        self,
        current: dict[str, float],
        target: dict[str, float],
        *,
        velocities: dict[str, float] | None = None,
    ) -> TrajectoryPlan:
        cfg = self._config
        if cfg.joints is not None:
            target = {j: v for j, v in target.items() if j in cfg.joints}
        if not target:
            return TrajectoryPlan(waypoints=(), duration=0.0, joints=frozenset())
        target = self._clamp(target)
        joints = frozenset(target)

        passthrough: dict[str, float] = {}
        shaped: dict[str, float] = {}
        for j, v in target.items():
            if j in cfg.passthrough_joints:
                passthrough[j] = v
            elif j not in current:
                # Never shape from an assumed 0.0 pose — apply directly.
                logger.warning(
                    "joint %r has no known current position; applying target directly", j
                )
                passthrough[j] = v
            else:
                shaped[j] = v

        if self._motion is None or not shaped:
            wp = TrajectoryWaypoint(
                positions={**shaped, **passthrough}, velocities=None, time_from_start=0.0
            )
            return TrajectoryPlan(waypoints=(wp,), duration=0.0, joints=joints)

        request = MotionRequest(
            current={j: current[j] for j in shaped},
            target=shaped,
            velocities=self._resolve_velocities(shaped, velocities),
            rate_hz=cfg.rate_hz,
        )
        waypoints = self._motion(request)
        if not waypoints:
            if not passthrough:
                return TrajectoryPlan(waypoints=(), duration=0.0, joints=joints)
            wp = TrajectoryWaypoint(
                positions=dict(passthrough), velocities=None, time_from_start=0.0
            )
            return TrajectoryPlan(waypoints=(wp,), duration=0.0, joints=joints)
        if passthrough:
            waypoints = tuple(
                replace(w, positions={**w.positions, **passthrough}) for w in waypoints
            )
        return TrajectoryPlan(
            waypoints=waypoints,
            duration=waypoints[-1].time_from_start,
            joints=joints,
        )

    def _clamp(self, target: dict[str, float]) -> dict[str, float]:
        provider = self._provider_position_limits()
        out: dict[str, float] = {}
        for j, v in target.items():
            limits = self._config.position_limits.get(j) or provider.get(j)
            if limits is not None:
                lo, hi = limits
                v = max(lo, min(hi, v))
            out[j] = v
        return out

    def _resolve_velocities(
        self, target: dict[str, float], payload: dict[str, float] | None
    ) -> dict[str, float]:
        """payload > config.default_velocities > kinematics provider > default."""
        cfg = self._config
        provider = self._provider_velocity_limits()
        resolved: dict[str, float] = {}
        for j in target:
            v = (payload or {}).get(j)
            if v is None:
                v = cfg.default_velocities.get(j)
            if v is None:
                v = provider.get(j)
            if v is None or v <= 0:
                v = cfg.default_velocity
            resolved[j] = float(v)
        return resolved
