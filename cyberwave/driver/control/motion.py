"""Default motion generator: trapezoidal velocity profile.

scipy is imported lazily so forward-only (``motion=None``) drivers never pay
for it; missing scipy raises with the ``cyberwave[drivers]`` install hint,
mirroring BaseKinematicsManipulator's pinocchio hint.
"""

from __future__ import annotations

import math

from .types import MotionRequest, TrajectoryWaypoint

# Fraction of the duration spent ramping up (and again ramping down).
RAMP_FRACTION = 0.2

_SCIPY_INSTALL_HINT = (
    "scipy is required for trapezoidal motion generation. Install it with: "
    'pip install "cyberwave[drivers]"'
)


def _import_cumtrapz():
    try:
        from scipy.integrate import cumulative_trapezoid  # noqa: PLC0415 (lazy by design)
    except ImportError as exc:
        raise ImportError(_SCIPY_INSTALL_HINT) from exc
    return cumulative_trapezoid


def trapezoidal_motion(req: MotionRequest) -> tuple[TrajectoryWaypoint, ...]:
    """Shape ``req.target`` into a trapezoidal-profile waypoint stream.

    duration = max_j |target_j - current_j| / velocities_j. Returns () when the
    target is empty or already satisfied. The final waypoint lands exactly on
    the target (profile integral normalized); peak velocity slightly exceeds
    the constraint because the duration formula, not the peak, is the contract.
    """
    deltas = {
        j: req.target[j] - req.current.get(j, req.target[j]) for j in req.target
    }
    moving = {j: d for j, d in deltas.items() if abs(d) > 1e-9}
    if not moving:
        return ()
    duration = max(abs(d) / req.velocities[j] for j, d in moving.items())
    if duration <= 0.0:
        return ()

    cumulative_trapezoid = _import_cumtrapz()
    import numpy as np  # noqa: PLC0415 (keep module import-light with scipy)

    n = max(2, math.ceil(duration * req.rate_hz) + 1)
    t = np.linspace(0.0, duration, n)
    ramp = RAMP_FRACTION * duration
    # Unit trapezoid: 0 -> 1 over [0, ramp], 1 over [ramp, T-ramp], 1 -> 0 after.
    shape = np.minimum(1.0, np.minimum(t / ramp, (duration - t) / ramp))
    shape = np.clip(shape, 0.0, 1.0)
    s = cumulative_trapezoid(shape, t, initial=0.0)
    total = s[-1]
    if total <= 0.0:
        # n == 2: both samples land exactly on the ramp endpoints, where the
        # trapezoid shape is 0 by construction — fall back to a linear
        # profile so the move still lands exactly on target.
        s = t / duration
        shape = np.full(n, 1.0 / duration)
        total = 1.0
    else:
        s = s / total  # normalized progress 0 -> 1

    waypoints: list[TrajectoryWaypoint] = []
    for i in range(n):
        positions = {
            j: req.current.get(j, req.target[j]) + deltas[j] * float(s[i])
            for j in req.target
        }
        velocities = {
            j: deltas[j] * float(shape[i]) / total for j in req.target
        }
        waypoints.append(
            TrajectoryWaypoint(
                positions=positions,
                velocities=velocities,
                time_from_start=float(t[i]),
            )
        )
    return tuple(waypoints)
