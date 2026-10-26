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
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from .._rest import request
from ..exceptions import (
    CyberwaveError,
    CyberwaveValidationError,
    SimulationBackendError,
)
from ..resources import BaseResourceManager
from ..twin.runtime_state import RUNTIME_MODE_SIMULATION, active_runtime_mode

logger = logging.getLogger(__name__)

# A running MuJoCo simulation is a billable cloud instance. Surfaced to users
# whenever the SDK starts one so the cost is never hidden.
SIMULATION_CREDITS_PER_MINUTE = 0.01
SIMULATION_CREDITS_PER_HOUR = round(SIMULATION_CREDITS_PER_MINUTE * 60, 2)  # 0.6

_JSON_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

#: Backends ``start()`` accepts. ``isaac`` is gated per organization server-side.
SIMULATION_BACKENDS: Tuple[str, ...] = ("mujoco", "mujoco_warp", "isaac")
#: Accepted spellings that are sent as their canonical backend.
SIMULATION_BACKEND_ALIASES: Dict[str, str] = {"mjwarp": "mujoco_warp"}
#: Execution profiles for ``backend="isaac"``: the physics engine and where it
#: steps. Every profile runs on, and bills as, an RTX GPU node.
ISAAC_EXECUTION_PROFILES: Tuple[str, ...] = (
    "isaac-cpu-physics",
    "isaac-gpu-physics",
    "isaac-newton-physics",
)
#: Profiles NVIDIA marks experimental (Newton physics).
EXPERIMENTAL_ISAAC_EXECUTION_PROFILES: Tuple[str, ...] = ("isaac-newton-physics",)
#: What the server uses for ``backend="isaac"`` when no profile is sent.
DEFAULT_ISAAC_EXECUTION_PROFILE = "isaac-cpu-physics"
#: Stable codes the server returns when it refuses a backend selection.
SIMULATION_BACKEND_ERROR_CODES = frozenset(
    {
        "simulation_backend_unsupported",
        "simulation_backend_not_enabled",
        "simulation_backend_incompatible",
        "simulation_backend_runtime_mismatch",
        "simulation_backend_capacity_unavailable",
        "simulation_parameter_unsupported",
        "simulation_parameter_invalid",
    }
)


def normalize_simulation_backend(backend: str) -> str:
    """Return the canonical backend for ``backend``, or raise.

    Unknown values fail closed (``CyberwaveValidationError``) before any request
    is made; nothing is ever silently mapped to MuJoCo.
    """
    if not isinstance(backend, str):
        raise CyberwaveValidationError(
            f"Simulation backend must be a string, got {type(backend).__name__}."
        )
    value = backend.strip().lower()
    value = SIMULATION_BACKEND_ALIASES.get(value, value)
    if value not in SIMULATION_BACKENDS:
        raise CyberwaveValidationError(
            f"Unsupported simulation backend {backend!r}. "
            f"Use one of: {', '.join(SIMULATION_BACKENDS)}."
        )
    return value


def _validate_backend_selection(
    backend: Optional[str], execution_profile: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Validate a ``(backend, execution_profile)`` pair client-side."""
    canonical = None if backend is None else normalize_simulation_backend(backend)
    if execution_profile is None:
        return canonical, None
    if canonical != "isaac":
        raise CyberwaveValidationError(
            "execution_profile is only valid with backend='isaac' "
            f"(got backend={backend!r})."
        )
    profile = str(execution_profile).strip().lower()
    if profile not in ISAAC_EXECUTION_PROFILES:
        raise CyberwaveValidationError(
            f"Unsupported Isaac execution profile {execution_profile!r}. "
            f"Use one of: {', '.join(ISAAC_EXECUTION_PROFILES)}."
        )
    return canonical, profile


# Terminal statuses, in the vocabulary the backend's simulation routes and the
# frontend use.
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped"})


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
    def is_terminal(self) -> bool:
        """True once the run has ended (``completed``, ``failed`` or ``stopped``)."""
        return self.status in TERMINAL_STATUSES

    @property
    def execution_profile(self) -> Optional[str]:
        """Execution profile the run was requested with (Isaac only), if reported."""
        value = self.raw.get("execution_profile")
        return str(value) if value else None

    @property
    def runtime_backend(self) -> Optional[str]:
        """Backend the simulation runtime itself reported, once it has."""
        value = self.raw.get("runtime_backend")
        return str(value) if value else None

    @property
    def total_duration_s(self) -> Optional[float]:
        """Configured run duration in seconds, if the backend reported one."""
        value = self.raw.get("total_duration_s")
        return float(value) if isinstance(value, (int, float)) else None

    def refresh(self) -> "Simulation":
        """Re-fetch this run's status from the backend (in place)."""
        if self._manager is None:
            raise CyberwaveError("Simulation handle is detached from its manager")
        latest = self._manager._get_by_id(
            self.environment_id, self.simulation_id
        ) or self._manager._get_detail(self.environment_id, self.simulation_id)
        if latest is None:
            # Neither listed nor readable on its own (a backend without the
            # detail route): treat as stopped/finished.
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

        A run that ended (``completed``, ``failed`` or ``stopped``, see
        :data:`TERMINAL_STATUSES`) is terminal — we fail fast instead of polling
        until the full ``timeout``, so a reused ``loading`` sim that errors out
        surfaces promptly.
        """
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            if self.status == "running":
                return self
            if self.is_terminal:
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
    def _extract_payload(exc: Exception) -> Optional[Dict[str, Any]]:
        """The JSON error body of an ApiException, if it has one."""
        data = getattr(exc, "data", None)
        if isinstance(data, dict):
            return data
        body = getattr(exc, "body", None)
        if isinstance(body, (str, bytes)):
            try:
                parsed = json.loads(body)
            except (ValueError, TypeError):
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    @classmethod
    def _extract_detail(cls, exc: Exception) -> Optional[str]:
        """Pull the backend's ``{"detail": ...}`` message out of an ApiException.

        The simulation endpoints return actionable errors in the response body
        (e.g. "Mujoco simulation is not enabled for: Logitech C920." — raised
        when a twin in the environment does not support the requested backend).
        """
        payload = cls._extract_payload(exc)
        if payload and payload.get("detail"):
            return str(payload["detail"])
        return None

    @staticmethod
    def _format_blockers(blocking_assets: List[Any], reasons: List[Any]) -> str:
        lines = [f"  - {reason}" for reason in reasons if reason]
        for asset in blocking_assets:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name") or asset.get("asset_uuid") or "asset"
            asset_reasons = [str(r) for r in asset.get("reasons") or [] if r]
            suffix = f": {'; '.join(asset_reasons)}" if asset_reasons else ""
            lines.append(f"  - {name}{suffix}")
        return ("\n" + "\n".join(lines)) if lines else ""

    def _raise_api_error(self, exc: Exception, operation: str) -> NoReturn:
        """Raise a clean :class:`CyberwaveError` from a REST failure.

        When the backend supplies a ``detail`` message, surface just that (plus
        the status code) instead of the raw transport dump of status line and
        header blocks. Otherwise fall back to the generic manager handler.
        """
        payload = self._extract_payload(exc) or {}
        detail = payload.get("detail")
        detail = str(detail) if detail else None
        status = getattr(exc, "status", None)
        suffix = f" (HTTP {status})" if status else ""
        code = payload.get("code") or payload.get("error_code")
        if isinstance(code, str) and code in SIMULATION_BACKEND_ERROR_CODES:
            blocking_assets = payload.get("blocking_assets") or []
            reasons = payload.get("reasons") or []
            if not isinstance(blocking_assets, list):
                blocking_assets = []
            if not isinstance(reasons, list):
                reasons = []
            message = (
                f"Failed to {operation}: {detail or code} [{code}]{suffix}"
                f"{self._format_blockers(blocking_assets, reasons)}"
            )
            raise SimulationBackendError(
                message,
                code=str(code),
                backend=payload.get("backend"),
                blocking_assets=blocking_assets,
                reasons=reasons,
                status_code=status,
            ) from exc
        if detail:
            raise CyberwaveError(f"Failed to {operation}: {detail}{suffix}")
        self._handle_error(exc, operation)
        raise  # For type checker

    def start(
        self,
        environment_id: str,
        *,
        backend: Optional[str] = "mujoco",
        execution_profile: Optional[str] = None,
        stream_data: bool = True,
        duration: Optional[float] = None,
        **options: Any,
    ) -> Simulation:
        """Start a simulation for ``environment_id`` and return its handle.

        Args:
            backend: ``"mujoco"`` (default), ``"mujoco_warp"`` (alias
                ``"mjwarp"``) or ``"isaac"``. ``None`` omits the field and lets
                the server pick its default (MuJoCo). Anything else raises
                :class:`~cyberwave.exceptions.CyberwaveValidationError` without
                contacting the server.
            execution_profile: Only with ``backend="isaac"``:
                ``"isaac-cpu-physics"`` (server default; PhysX on the CPU),
                ``"isaac-gpu-physics"`` (PhysX on the GPU) or
                ``"isaac-newton-physics"`` (experimental: NVIDIA Newton on the
                GPU). Every profile runs on, and bills as, an RTX GPU node.

        Raises:
            SimulationBackendError: The server refused the backend (not enabled
                for the organization, environment incompatible, ...). Carries
                ``code``, ``blocking_assets`` and ``reasons``. With code
                ``simulation_backend_capacity_unavailable`` (HTTP 503) no
                qualified Isaac Sim GPU node is free; retry later. The server
                never falls back to another backend.

        Isaac Sim provides PhysX (CPU or GPU) or Newton physics, RTX-rendered
        RGB/depth cameras, thermal cameras, IMU, range sensors, microphones and
        speakers. Thermal images come from a heuristic heat model, not a
        radiometric sensor model, and audio is mixed rather than propagated
        acoustically. A start is refused with a ``blocking_assets`` entry for
        each twin that cannot run on Isaac Sim: its asset does not declare the
        ``"isaac"`` backend or has not passed the Isaac Sim compatibility check
        for the selected execution profile, the twin itself opts out of
        ``"isaac"``, or the scene export rejected it.
        """
        backend, execution_profile = _validate_backend_selection(
            backend, execution_profile
        )
        body: Dict[str, Any] = {"stream_data": stream_data}
        if backend is not None:
            body["backend"] = backend
        if execution_profile is not None:
            body["execution_profile"] = execution_profile
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

    def preflight(
        self,
        environment_id: str,
        *,
        backend: str = "isaac",
        execution_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask whether ``environment_id`` can run on ``backend`` without starting it.

        Returns the server's answer as a dict: ``enabled`` (organization
        rollout), ``eligible``, ``available_backends``, ``execution_profiles``,
        ``experimental_execution_profiles``, ``blocking_assets``, ``reasons``,
        ``bundle`` (``{"status", "digest"}``) and, for organizations with Isaac
        enabled, ``capacity`` (``{"available", "code", "reasons"}``). Capacity
        never changes ``eligible``; a start without it raises
        ``SimulationBackendError`` with ``simulation_backend_capacity_unavailable``.
        """
        backend, execution_profile = _validate_backend_selection(
            backend, execution_profile
        )
        query: List[Tuple[str, str]] = [("backend", str(backend))]
        if execution_profile is not None:
            query.append(("execution_profile", execution_profile))
        try:
            _param = self.api.api_client.param_serialize(
                method="GET",
                resource_path="/api/v1/environments/{uuid}/simulation/preflight",
                path_params={"uuid": str(environment_id)},
                query_params=query,
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
                e, f"check {backend} support for environment {environment_id}"
            )
        return dict(payload or {})

    def request_scene_bundle(self, environment_id: str) -> Dict[str, Any]:
        """Request (or reuse) the Isaac scene bundle of the current revision.

        The build runs in the background and is idempotent per revision.
        Returns ``{"status", "digest", "bundle_id"}``; ``digest`` is set only
        once the bundle is ``ready``. Follow the build with :meth:`preflight`
        (its ``bundle`` entry). Re-requesting a failed bundle starts it again
        once the failure is a few minutes old; before that the failed record is
        returned as is.

        Raises:
            SimulationBackendError: ``simulation_backend_not_enabled`` (HTTP 403)
                when Isaac Sim is not enabled for the organization, or
                ``simulation_backend_incompatible`` (HTTP 409) when the scene
                cannot be bundled; ``blocking_assets`` and ``reasons`` say why.
        """
        try:
            _param = self.api.api_client.param_serialize(
                method="POST",
                resource_path="/api/v1/environments/{uuid}/simulation/scene-bundles",
                path_params={"uuid": str(environment_id)},
                header_params=dict(_JSON_HEADERS),
                body={},
                auth_settings=["CustomTokenAuthentication"],
            )
            response_data = self.api.api_client.call_api(*_param)
            response_data.read()
            payload = self.api.api_client.response_deserialize(
                response_data=response_data,
                response_types_map={"202": "object"},
            ).data
        except Exception as e:
            self._raise_api_error(
                e, f"request the scene bundle of environment {environment_id}"
            )
        return dict(payload or {})

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

    def _get_detail(
        self, environment_id: str, simulation_id: str
    ) -> Optional[Simulation]:
        """One run, active or ended, from the simulation detail route.

        Unlike the listing, which keeps an ended run only briefly, the detail
        reports how any run of the environment ended. ``None`` when the backend
        cannot say (unknown run, or a backend that predates the route). Any
        other failure (5xx, auth, transport) raises: it says nothing about how
        the run ended, so it must not read as "stopped".
        """
        status, payload = request(
            self.api.api_client,
            "GET",
            "/api/v1/environments/{uuid}/simulations/{simulation_id}",
            path_params={
                "uuid": str(environment_id),
                "simulation_id": str(simulation_id),
            },
            allow_statuses=(404, 405),
        )
        if status in (404, 405):
            logger.debug("No detail for simulation %s (HTTP %d)", simulation_id, status)
            return None
        if not isinstance(payload, dict) or not payload.get("status"):
            return None
        return Simulation._from_dict(environment_id, payload, self)

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
