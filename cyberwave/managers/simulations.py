"""Simulation resource manager (``cw.environments.simulations``).

Wraps the environment-simulation REST endpoints and returns lightweight
:class:`Simulation` handles. Used directly and reused by
:meth:`cyberwave.twin.classes.CameraTwin.start_streaming` in simulation mode.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, NoReturn, Optional

from ..exceptions import CyberwaveError
from ..resources import BaseResourceManager
from ..twin.runtime_state import RUNTIME_MODE_SIMULATION, active_runtime_mode

logger = logging.getLogger(__name__)

# A running MuJoCo simulation is a billable cloud instance. Surfaced to users
# whenever the SDK starts one so the cost is never hidden.
SIMULATION_CREDITS_PER_MINUTE = 0.01
SIMULATION_CREDITS_PER_HOUR = round(SIMULATION_CREDITS_PER_MINUTE * 60, 2)  # 0.6

_JSON_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


@dataclass
class Simulation:
    """Handle to a single environment simulation run."""

    environment_id: str
    simulation_id: str
    status: str
    stream_data: bool = True
    backend: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    _manager: Optional["SimulationManager"] = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def _from_dict(
        cls, environment_id: str, data: Dict[str, Any], manager: "SimulationManager"
    ) -> "Simulation":
        data = data or {}
        return cls(
            environment_id=str(environment_id),
            simulation_id=str(data.get("simulation_id") or ""),
            status=str(data.get("status") or "unknown"),
            stream_data=bool(data.get("stream_data", True)),
            backend=data.get("backend"),
            raw=dict(data),
            _manager=manager,
        )

    @property
    def is_active(self) -> bool:
        """True while the run is loading or running."""
        return self.status in ("loading", "running")

    @property
    def total_duration_s(self) -> Optional[float]:
        """Configured run duration in seconds, if the backend reported one."""
        value = self.raw.get("total_duration_s")
        return float(value) if isinstance(value, (int, float)) else None

    def refresh(self) -> "Simulation":
        """Re-fetch this run's status from the backend (in place)."""
        if self._manager is None:
            raise CyberwaveError("Simulation handle is detached from its manager")
        latest = self._manager._get_by_id(self.environment_id, self.simulation_id)
        if latest is None:
            # No longer in the active list -> treat as stopped/finished.
            self.status = "stopped"
            return self
        self.status = latest.status
        self.stream_data = latest.stream_data
        self.backend = latest.backend
        self.raw = latest.raw
        return self

    def wait_until_active(
        self, timeout: float = 120.0, poll: float = 2.0
    ) -> "Simulation":
        """Poll until status is ``running``. Raise on a terminal status or timeout.

        A run that ``failed`` or dropped off the active list (``refresh()``
        reports ``stopped``) is terminal — we fail fast instead of polling until
        the full ``timeout``, so a reused ``loading`` sim that errors out surfaces
        promptly.
        """
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            if self.status == "running":
                return self
            if self.status in ("failed", "stopped"):
                raise CyberwaveError(
                    f"Simulation {self.simulation_id} did not become active "
                    f"(status: {self.status})"
                )
            if time.monotonic() >= deadline:
                raise CyberwaveError(
                    f"Simulation {self.simulation_id} not active after "
                    f"{timeout:.0f}s (last status: {self.status})"
                )
            time.sleep(poll)

    def stop(self) -> None:
        """Stop this simulation run."""
        if self._manager is None:
            raise CyberwaveError("Simulation handle is detached from its manager")
        self._manager.stop(self.environment_id, self.simulation_id)
        self.status = "stopped"


@dataclass
class _CacheEntry:
    """One cached, presumed-``running`` simulation for an environment."""

    sim: Simulation
    cached_at: float
    first_seen: float
    duration_s: Optional[float]

    def is_valid(self, *, ttl: float) -> bool:
        # Only "running" counts — unlike Simulation.is_active, a "loading" run
        # is not yet safe to hand to a getter that expects live data.
        if self.sim.status != "running":
            return False
        now = time.monotonic()
        if now - self.cached_at >= ttl:
            return False
        if self.duration_s is not None and now - self.first_seen >= self.duration_s:
            return False
        return True


class _SimReadyCache:
    """Thread-safe per-environment simulation-readiness cache.

    Avoids a REST ``list()`` call on every sim-mode getter tick. See
    :meth:`_CacheEntry.is_valid` for what makes an entry usable.
    """

    REVALIDATE_TTL_S = 2.0

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock_for(self, env_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(env_id, threading.Lock())

    def get_valid(self, env_id: str) -> Optional[Simulation]:
        """Lock-free read of a still-valid cached simulation, if any."""
        entry = self._entries.get(env_id)
        if entry is None or not entry.is_valid(ttl=self.REVALIDATE_TTL_S):
            return None
        return entry.sim

    def put(self, env_id: str, sim: Simulation) -> None:
        now = time.monotonic()
        prev = self._entries.get(env_id)
        # Preserve the original first-seen time across re-caching the *same*
        # simulation, so its duration bound is measured from when it actually
        # started, not from every subsequent cache refresh.
        first_seen = (
            prev.first_seen
            if prev is not None and prev.sim.simulation_id == sim.simulation_id
            else now
        )
        self._entries[env_id] = _CacheEntry(
            sim=sim,
            cached_at=now,
            first_seen=first_seen,
            duration_s=sim.total_duration_s,
        )

    def invalidate(self, env_id: str) -> None:
        self._entries.pop(env_id, None)


def _sim_ready_cache_for(client: Any) -> _SimReadyCache:
    cache = getattr(client, "_sim_ready_cache", None)
    if cache is None:
        cache = _SimReadyCache()
        client._sim_ready_cache = cache
    return cache


def running_simulation(twin: Any) -> Optional[Simulation]:
    """Return the simulation running for *twin*'s environment, or ``None``.

    Passive: reads the readiness cache, then falls back to a single
    ``get_active`` REST call. **Never starts a simulation.** Returns ``None`` in
    live runtime mode or when no simulation is active.
    """
    if active_runtime_mode(twin.client) != RUNTIME_MODE_SIMULATION:
        return None

    client = twin.client
    env_id = str(twin.environment_id)
    cache = _sim_ready_cache_for(client)

    cached = cache.get_valid(env_id)
    if cached is not None:
        return cached

    with cache.lock_for(env_id):
        cached = cache.get_valid(env_id)  # another thread may have just refreshed it
        if cached is not None:
            return cached

        sim = client.environments.simulations.get_active(env_id)
        if sim is None:
            return None
        if sim.status == "running":
            cache.put(env_id, sim)
        return sim


class SimulationManager(BaseResourceManager):
    """Start, inspect, and stop simulations for an environment."""

    @staticmethod
    def _extract_detail(exc: Exception) -> Optional[str]:
        """Pull the backend's ``{"detail": ...}`` message out of an ApiException.

        The simulation endpoints return actionable errors in the response body
        (e.g. "Mujoco simulation is not enabled for: Logitech C920." — raised
        when a twin in the environment does not support the requested backend).
        """
        data = getattr(exc, "data", None)
        if isinstance(data, dict) and data.get("detail"):
            return str(data["detail"])
        body = getattr(exc, "body", None)
        if isinstance(body, (str, bytes)):
            try:
                parsed = json.loads(body)
            except (ValueError, TypeError):
                return None
            if isinstance(parsed, dict) and parsed.get("detail"):
                return str(parsed["detail"])
        return None

    def _raise_api_error(self, exc: Exception, operation: str) -> NoReturn:
        """Raise a clean :class:`CyberwaveError` from a REST failure.

        When the backend supplies a ``detail`` message, surface just that (plus
        the status code) instead of the raw transport dump of status line and
        header blocks. Otherwise fall back to the generic manager handler.
        """
        detail = self._extract_detail(exc)
        if detail:
            status = getattr(exc, "status", None)
            suffix = f" (HTTP {status})" if status else ""
            raise CyberwaveError(f"Failed to {operation}: {detail}{suffix}")
        self._handle_error(exc, operation)
        raise  # For type checker

    def start(
        self,
        environment_id: str,
        *,
        backend: str = "mujoco",
        stream_data: bool = True,
        duration: Optional[float] = None,
        **options: Any,
    ) -> Simulation:
        """Start a simulation for ``environment_id`` and return its handle."""
        body: Dict[str, Any] = {"stream_data": stream_data, "backend": backend}
        if duration is not None:
            body["duration"] = duration
        body.update(options)
        try:
            _param = self.api.api_client.param_serialize(
                method="POST",
                resource_path="/api/v1/environments/{uuid}/simulations",
                path_params={"uuid": str(environment_id)},
                header_params=dict(_JSON_HEADERS),
                body=body,
                auth_settings=["CustomTokenAuthentication"],
            )
            response_data = self.api.api_client.call_api(*_param)
            response_data.read()
            payload = self.api.api_client.response_deserialize(
                response_data=response_data,
                response_types_map={"200": "object"},
            ).data
        except Exception as e:
            self._raise_api_error(
                e, f"start simulation for environment {environment_id}"
            )
        sim_dict = (payload or {}).get("simulation") or {}
        return Simulation._from_dict(environment_id, sim_dict, self)

    def list(self, environment_id: str) -> List[Simulation]:
        """List active simulations for ``environment_id``."""
        try:
            _param = self.api.api_client.param_serialize(
                method="GET",
                resource_path="/api/v1/environments/{uuid}/simulations",
                path_params={"uuid": str(environment_id)},
                header_params={"Accept": "application/json"},
                auth_settings=["CustomTokenAuthentication"],
            )
            response_data = self.api.api_client.call_api(*_param)
            response_data.read()
            payload = self.api.api_client.response_deserialize(
                response_data=response_data,
                response_types_map={"200": "object"},
            ).data
        except Exception as e:
            self._raise_api_error(
                e, f"list simulations for environment {environment_id}"
            )
        entries = (payload or {}).get("active_simulations") or []
        return [Simulation._from_dict(environment_id, entry, self) for entry in entries]

    def get_active(self, environment_id: str) -> Optional[Simulation]:
        """Return the running simulation, else the most recent loading one, else None."""
        sims = self.list(environment_id)
        for status in ("running", "loading"):
            for sim in sims:
                if sim.status == status:
                    return sim
        return None

    def _get_by_id(
        self, environment_id: str, simulation_id: str
    ) -> Optional[Simulation]:
        for sim in self.list(environment_id):
            if sim.simulation_id == str(simulation_id):
                return sim
        return None

    def stop(self, environment_id: str, simulation_id: str) -> None:
        """Stop a specific simulation run."""
        try:
            _param = self.api.api_client.param_serialize(
                method="POST",
                resource_path=(
                    "/api/v1/environments/{uuid}/simulations/{simulation_id}/stop"
                ),
                path_params={
                    "uuid": str(environment_id),
                    "simulation_id": str(simulation_id),
                },
                header_params={"Accept": "application/json"},
                auth_settings=["CustomTokenAuthentication"],
            )
            response_data = self.api.api_client.call_api(*_param)
            response_data.read()
        except Exception as e:
            self._raise_api_error(e, f"stop simulation {simulation_id}")
