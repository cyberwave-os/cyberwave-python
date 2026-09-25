"""Simulation capability levels and the ``@simulation_level`` preflight decorator.

Each simulation-dependent method declares the capability it needs. In live
runtime mode the check is a no-op; in simulation mode it fails fast before the
wrapped body runs. Level 0 (``PLAYGROUND``) is the default and is never
annotated. Internal implementation detail — not part of the public API.
"""

from __future__ import annotations

from enum import IntEnum
from functools import wraps
from typing import Any, Callable


class SimLevel(IntEnum):
    """Capability a method needs from the simulation runtime."""

    UNSUPPORTED = -1  # not produced by any simulation backend (live/driver-only)
    PLAYGROUND = 0  # default; no running-simulation requirement at all
    MUJOCO = 1  # requires a running MuJoCo simulation specifically (physics + rendering)
    BOTH = 2  # requires *some* running simulation, either playground or MuJoCo


#: Backend name -> level available while that backend is the running simulation.
_BACKEND_LEVEL: dict[str, SimLevel] = {
    "playground": SimLevel.PLAYGROUND,
    "mujoco": SimLevel.MUJOCO,
}


def backend_sim_level(backend: Any) -> SimLevel:
    """Map a running simulation's ``backend`` string to the level it provides."""
    return _BACKEND_LEVEL.get(str(backend or "").strip().lower(), SimLevel.PLAYGROUND)


def simulation_level(level: SimLevel = SimLevel.PLAYGROUND) -> Callable[..., Any]:
    """Declare the simulation capability a method requires.

    No-op in live runtime mode. In simulation mode the wrapped method's twin
    (``self._twin`` on handles, or ``self`` on the twin itself) runs
    :meth:`Twin._ensure_simulation_support` before the body executes.

    Use ``SimLevel.BOTH`` for methods that need a running simulation but don't
    care which backend — e.g. ``twin.joints`` reads/writes joint state over
    MQTT, which both ``playground`` and ``mujoco`` publish. Use ``SimLevel.MUJOCO``
    only for methods that need physics/rendering that ``playground`` doesn't
    produce (camera frames, depth, point clouds).
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            twin = getattr(self, "_twin", self)
            twin._ensure_simulation_support(level, method=fn.__qualname__)
            return fn(self, *args, **kwargs)

        wrapper.__cw_sim_level__ = level
        return wrapper

    return deco
