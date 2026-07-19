"""
Main Cyberwave client that integrates REST and MQTT APIs
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

from cyberwave._version import get_version
from cyberwave.config import (
    CyberwaveConfig,
    DEFAULT_BASE_URL,
)
from cyberwave.constants import (
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_SIM,
)
from cyberwave.controller import EdgeController
from cyberwave.data.api import DataBus
from cyberwave.data.backend import DataBackend
from cyberwave.data.utils import (
    close_frame_subscribe_caches_for_backend,
    fetch_twin_frame,
)
from cyberwave.exceptions import (
    CyberwaveAPIError,
    CyberwaveError,
    CyberwaveInsufficientCreditsError,
    UnauthorizedException,
)
from cyberwave.models.manager import ModelManager
from cyberwave.mqtt_client import CyberwaveMQTTClient
from cyberwave.twin import Twin, create_twin
from cyberwave.utils import TimeReference
from cyberwave.workers.hooks import HookRegistry

# Import camera streamers with optional dependency handling
try:
    from cyberwave.sensor import CV2CameraStreamer as CameraStreamer

    _has_camera = True
except ImportError:
    _has_camera = False
    CameraStreamer = None

try:
    from cyberwave.sensor import RealSenseStreamer

    _has_realsense = True
except ImportError:
    _has_realsense = False
    RealSenseStreamer = None

if TYPE_CHECKING:
    from cyberwave.scene import Scene


logger = logging.getLogger(__name__)


# SDK identity headers attached to every outbound REST request. These let the
# backend attribute API traffic to an SDK version cohort (``backend_api_activity``
# in ``src/lib/posthog_tracking.py``) without any per-call payload changes.
_SDK_VERSION = get_version()
_SDK_USER_AGENT = f"cyberwave-python/{_SDK_VERSION}"
_SDK_VERSION_HEADER = "X-Cyberwave-SDK-Version"


def _apply_sdk_identity_headers(header_params: dict[str, Any]) -> dict[str, Any]:
    """Add SDK identity headers without overriding caller-provided values.

    Sets a ``User-Agent`` and ``X-Cyberwave-SDK-Version`` header so backend
    request tracking can report SDK version distribution over REST. Callers that
    set their own ``User-Agent`` (e.g. the CLI) keep it.
    """
    if not any(str(key).lower() == "user-agent" for key in header_params):
        header_params["User-Agent"] = _SDK_USER_AGENT
    if not any(
        str(key).lower() == _SDK_VERSION_HEADER.lower() for key in header_params
    ):
        header_params[_SDK_VERSION_HEADER] = _SDK_VERSION
    return header_params


_RUNTIME_MODE_MAP = {
    "live": "live",
    "real-world": "live",
    "real": "live",
    "tele": "live",
    "teleoperation": "live",
    "simulation": "simulation",
    "sim": "simulation",
    "sim_tele": "simulation",
    "mujoco": "simulation",
    "playground": "simulation",
}

# Simulation profiles that spin up a MuJoCo cloud instance when selected via
# ``affect(...)``. ``playground`` is a simulation runtime too, but a lightweight
# kinematic one with no MuJoCo instance to start (level-0 only), so it is absent.
_AFFECT_MUJOCO_PROFILES = frozenset({"sim", "simulation", "sim_tele", "mujoco"})


def _affect_autostart_backend(mode: Optional[str]) -> Optional[str]:
    """Backend to auto-start for an ``affect(...)`` profile, or ``None``.

    Returns ``"mujoco"`` for MuJoCo simulation profiles; ``None`` for
    ``playground`` and every live profile (nothing to start).
    """
    normalized = (mode or "").lower().strip()
    return "mujoco" if normalized in _AFFECT_MUJOCO_PROFILES else None


def _resolve_runtime_mode(mode: Optional[str]) -> str:
    normalized = (mode or "live").lower().strip()
    if normalized not in _RUNTIME_MODE_MAP:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'live'/'real-world' or 'simulation'."
        )
    return _RUNTIME_MODE_MAP[normalized]


def _default_state_source_type(runtime_mode: str) -> str:
    return SOURCE_TYPE_SIM if runtime_mode == "simulation" else SOURCE_TYPE_EDGE


class Cyberwave:
    """
    Main client for the Cyberwave Digital Twin Platform.

    This client provides access to both REST and MQTT APIs, along with
    high-level abstractions for working with digital twins.

    Example:
        >>> client = Cyberwave(base_url="http://localhost:8000", api_key="your_api_key")
        >>> workspaces = client.workspaces.list()
        >>> twin = client.twin("the-robot-studio/so101")

    Args:
        base_url: Base URL of the Cyberwave backend
        api_key: API key for authentication
        token: Deprecated alias for api_key (kept for backwards compatibility)
        mqtt_host: MQTT broker host (optional, defaults to "mqtt.cyberwave.com")
        mqtt_port: MQTT broker port (default: 8883)
        mqtt_username: MQTT username placeholder (default: "mqttcyb")
        mqtt_use_tls: Enable TLS for MQTT connection
        mqtt_tls_ca_cert: Path to CA cert bundle for MQTT TLS
        mqtt_protocol: MQTT protocol version (default: MQTTv311).
            Pass ``paho.mqtt.client.MQTTv5`` to use MQTT v5 when your broker supports it.
        source_type: Optional explicit default state/telemetry source_type override
        mode: Runtime mode, either live or simulation (defaults to live)
        environment_id: Default environment ID (overrides CYBERWAVE_ENVIRONMENT_ID env var)
        workspace_id: Default workspace ID (overrides CYBERWAVE_WORKSPACE_ID env var)
        **config_kwargs: Additional configuration options
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        mqtt_host: Optional[str] = None,
        mqtt_port: int | None = None,
        mqtt_username: Optional[str] = None,
        mqtt_use_tls: bool = False,
        mqtt_tls_ca_cert: Optional[str] = None,
        mqtt_protocol: Optional[int] = None,
        topic_prefix: Optional[str] = None,
        source_type: Optional[str] = None,
        mode: Optional[str] = "live",
        environment_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **config_kwargs,
    ):
        runtime_mode = _resolve_runtime_mode(mode)

        if not base_url:
            base_url = os.getenv("CYBERWAVE_BASE_URL", DEFAULT_BASE_URL)

        if api_key is None and token is not None:
            warnings.warn(
                "'token' is deprecated and will be removed in a future release. "
                "Use 'api_key' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            api_key = token

        if api_key is None:
            api_key = os.getenv("CYBERWAVE_API_KEY", None)

        if api_key is None:
            raise ValueError(
                "No API key found! Set CYBERWAVE_API_KEY. "
                "Get yours at https://cyberwave.com/profile"
            )

        self.config = CyberwaveConfig(
            base_url=base_url,
            api_key=api_key,
            token=token,
            mqtt_host=mqtt_host,
            mqtt_port=mqtt_port,
            mqtt_username=mqtt_username,
            mqtt_use_tls=mqtt_use_tls,
            mqtt_tls_ca_cert=mqtt_tls_ca_cert,
            mqtt_protocol=mqtt_protocol,
            topic_prefix=topic_prefix,
            environment_id=environment_id or os.getenv("CYBERWAVE_ENVIRONMENT_ID", None),
            workspace_id=workspace_id or os.getenv("CYBERWAVE_WORKSPACE_ID", None),
            source_type=source_type,
            runtime_mode=runtime_mode,
            **config_kwargs,
        )

        if source_type is None and not os.getenv("CYBERWAVE_SOURCE_TYPE"):
            self.config.source_type = _default_state_source_type(runtime_mode)

        self._setup_rest_client()
        self._mqtt_client: Optional[CyberwaveMQTTClient] = None
        self._data_backend: Optional[DataBackend] = None
        self._data_bus: Optional[DataBus] = None
        self._data_twin_uuid_override: Optional[str] = None
        self._data_sensor_name_override: Optional[str] = None
        self._hook_registry = HookRegistry()
        self._init_managers()

    def _setup_rest_client(self):
        """Setup the REST API client with authentication"""
        from cyberwave.rest import DefaultApi, ApiClient, Configuration

        configuration = Configuration(host=self.config.base_url)

        if self.config.api_key:
            configuration.api_key["CustomTokenAuthentication"] = self.config.api_key
            configuration.api_key_prefix["CustomTokenAuthentication"] = "Bearer"

        configuration.verify_ssl = self.config.verify_ssl

        api_client = ApiClient(configuration)

        original_response_deserialize = api_client.response_deserialize
        last_request_headers = {}

        def response_deserialize_with_headers(response_data, response_types_map=None):
            try:
                return original_response_deserialize(response_data, response_types_map)
            except Exception as e:
                if hasattr(e, "__dict__") and not hasattr(e, "request_headers"):
                    e.request_headers = last_request_headers.copy()
                if getattr(e, "status", None) == 402:
                    import json as _json

                    balance: Optional[float] = None
                    manual_block = False
                    manual_block_reason = ""
                    try:
                        body = _json.loads(getattr(e, "body", "") or "{}")
                        detail = body.get("detail", "")
                        if "balance=" in detail:
                            balance = float(detail.split("balance=")[-1])
                        manual_block = bool(body.get("manual_block", False))
                        manual_block_reason = body.get("manual_block_reason", "")
                    except Exception:
                        pass
                    msg = "Insufficient credits"
                    if balance is not None:
                        msg += f" (balance: {balance})"
                    raise CyberwaveInsufficientCreditsError(
                        message=msg,
                        status_code=402,
                        response_data=getattr(e, "body", None),
                        request_headers=last_request_headers.copy(),
                        balance=balance,
                        manual_block=manual_block,
                        manual_block_reason=manual_block_reason,
                    ) from e
                raise

        original_call_api = api_client.call_api

        def call_api_with_header_tracking(
            method,
            url,
            header_params=None,
            body=None,
            post_params=None,
            _request_timeout=None,
            **kwargs,
        ):
            header_params = dict(header_params or {})
            has_authorization = any(
                str(key).lower() == "authorization" for key in header_params
            )
            if self.config.api_key and not has_authorization:
                header_params["Authorization"] = f"Bearer {self.config.api_key}"

            _apply_sdk_identity_headers(header_params)

            last_request_headers.clear()
            if header_params:
                last_request_headers.update(header_params)
            return original_call_api(
                method,
                url,
                header_params,
                body,
                post_params,
                _request_timeout,
                **kwargs,
            )

        api_client.response_deserialize = response_deserialize_with_headers
        api_client.call_api = call_api_with_header_tracking

        self.api = DefaultApi(api_client)
        self._api_client = api_client

        self._wrap_api_methods()

    def _init_managers(self) -> None:
        """(Re)build high-level managers that hold REST client references.

        ``configure()`` can rebuild ``self._api_client`` and ``self.api`` with a
        new base URL or API key. Any manager instantiated before that point
        would otherwise keep talking to the stale backend. Centralizing manager
        setup here lets ``__init__`` and ``configure()`` refresh every surface
        consistently.
        """
        from cyberwave.actions import ActionsClient
        from cyberwave.agents import AgentManager
        from cyberwave.resources import (
            AttachmentManager,
            AssetManager,
            DatasetManager,
            EdgeManager,
            EnvironmentManager,
            ProjectManager,
            TwinManager,
            WorkspaceManager,
        )
        from cyberwave.workflow_executions import WorkflowExecutionManager
        from cyberwave.workflows import WorkflowManager, WorkflowRunManager

        # ``cw.models`` is the unified surface for runtime, catalog, and playground:
        #   cw.models.load("yolov8n")              → edge LoadedModel
        #   cw.models.load("acme/models/sam-3.1")  → CloudLoadedModel (Playground)
        #   cw.models.list(deployment="edge")       → catalog records
        #   cw.models.get("acme/models/yolo26n")    → single catalog record
        #   cw.models.playground("acme/models/gemini-robotics-er").run(image=...)
        self.models = ModelManager(
            data_bus=lambda: self._try_get_data_bus(),
            api_client=self.api,
        )
        self.workspaces = WorkspaceManager(self.api)
        self.projects = ProjectManager(self.api)
        self.environments = EnvironmentManager(self.api)
        self.attachments = AttachmentManager(self.api)
        self.assets = AssetManager(self.api)
        self.datasets = DatasetManager(self.api)
        self.edges = EdgeManager(self.api)
        self.twins = TwinManager(self.api, client=self)
        from cyberwave.managers.policies import PolicyManager

        self.policies = PolicyManager(self)
        self.actions = ActionsClient(self._api_client)
        self.agents = AgentManager(self._api_client)
        self.control = self.agents.control
        self.workflows = WorkflowManager(self)
        self.workflow_runs = WorkflowRunManager(self)
        self.workflow_executions = WorkflowExecutionManager(self)

    def _wrap_api_methods(self):
        """Wrap API methods to provide better error messages for authentication failures"""
        for attr_name in dir(self.api):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self.api, attr_name)
            if callable(attr):
                wrapped = self._create_wrapped_method(attr)
                setattr(self.api, attr_name, wrapped)

    def _create_wrapped_method(self, method):
        """Create a wrapped version of an API method that handles auth errors"""

        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except UnauthorizedException as e:
                error_msg = "Authentication failed: Invalid or missing credentials.\n\n"

                if self.config.api_key:
                    error_msg += "Your API key appears to be invalid or expired.\n"
                else:
                    error_msg += "No authentication credentials were provided.\n"

                error_msg += "  1. Add an API key at https://cyberwave.com/profile\n"
                error_msg += "  2. Copy it to your clipboard\n"
                error_msg += "  3. Set the environment variable:\n\nexport CYBERWAVE_API_KEY=your_api_key\n"
                error_msg += "  4. Run your script again!\n"

                if hasattr(e, "request_headers") and e.request_headers:
                    auth_header = e.request_headers.get("Authorization", "Not present")
                    if auth_header and auth_header != "Not present":
                        parts = auth_header.split(" ")
                        if len(parts) == 2:
                            token_preview = (
                                parts[1][:8] + "..." if len(parts[1]) > 8 else parts[1]
                            )
                            error_msg += (
                                f"Authorization header: {parts[0]} {token_preview}\n"
                            )
                    else:
                        error_msg += "Authorization header: Not present\n"

                raise CyberwaveAPIError(
                    error_msg,
                    status_code=401,
                    response_data=e.body if hasattr(e, "body") else None,
                ) from e

        return wrapped

    def _get_data_backend(self) -> DataBackend:
        """Shared Zenoh/filesystem backend (one session per :class:`Cyberwave` client)."""
        if self._data_backend is None:
            import os

            from cyberwave.data.config import BackendConfig, get_backend

            cfg = BackendConfig()
            if (
                cfg.backend == "zenoh"
                and not cfg.zenoh_connect
                and not cfg.zenoh_listen
            ):
                host = os.environ.get("ZENOH_ROUTER_HOST", "127.0.0.1")
                port = os.environ.get("ZENOH_ROUTER_PORT", "7447")
                cfg.zenoh_connect = [f"tcp/{host}:{port}"]
                logger.info(
                    "ZENOH_CONNECT unset; connecting to edge router at %s",
                    cfg.zenoh_connect[0],
                )
            self._data_backend = get_backend(cfg)
        return self._data_backend

    def data_bus_for(
        self,
        twin_uuid: str,
        *,
        sensor_name: str | None = None,
    ) -> DataBus:
        """Return a :class:`~cyberwave.data.api.DataBus` for any twin on the shared backend."""
        return DataBus(self._get_data_backend(), twin_uuid, sensor_name=sensor_name)

    def use_data_bus_for(
        self,
        twin_uuid: str,
        *,
        sensor_name: str | None = None,
    ) -> None:
        """Pin :attr:`data` to *twin_uuid*, overriding ``CYBERWAVE_TWIN_UUID``.

        Lazy: does not open a transport backend until :attr:`data` is
        first accessed, so calling this in a compiled worker prelude
        costs nothing for workflows that never touch the data bus.
        Idempotent: repeated calls with the same twin/sensor are no-ops,
        which preserves the existing :class:`DataBus` (and its per-channel
        ``HeaderTemplate.seq`` counters) across multi-trigger hook
        invocations that re-seed on every frame.
        """
        if (
            self._data_twin_uuid_override == twin_uuid
            and self._data_sensor_name_override == sensor_name
        ):
            return
        self._data_bus = None
        self._data_twin_uuid_override = twin_uuid
        self._data_sensor_name_override = sensor_name

    def fetch_zenoh_frame(
        self,
        twin_uuid: str,
        *,
        sensor_name: str = "default",
        timeout_s: float = 1.0,
        max_age_ms: float | None = None,
    ) -> Any | None:
        """Return the next camera frame on the Zenoh ``frames`` stream (subscribe path)."""
        return fetch_twin_frame(
            self._get_data_backend(),
            twin_uuid,
            sensor_name=sensor_name,
            timeout_s=timeout_s,
            max_age_ms=max_age_ms,
        )

    def diagnose_zenoh_frames(
        self,
        twin_uuid: str,
        *,
        sensor_name: str = "default",
        timeout_s: float = 0.5,
    ) -> dict[str, Any]:
        """Return Zenoh connectivity hints for ``get_frame(source='zenoh')`` debugging."""
        from cyberwave.data.keys import build_key, build_wildcard

        try:
            backend = self._get_data_backend()
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "hint": (
                    "Install eclipse-zenoh: pip install 'cyberwave[zenoh]'"
                ),
            }

        prefix = getattr(backend, "key_prefix", "cw")
        frame_key = build_key(
            twin_uuid,
            "frames",
            sensor_name,
            prefix=prefix,
        )
        wildcard = build_wildcard(twin_uuid, "frames", prefix=prefix)
        frame = fetch_twin_frame(
            backend,
            twin_uuid,
            sensor_name=sensor_name,
            timeout_s=timeout_s,
        )
        stats: dict[str, Any] = {}
        stats_fn = getattr(backend, "stats", None)
        if stats_fn is not None:
            stats = stats_fn()

        recv = stats.get("recv", {})
        frame_keys = [k for k in recv if "/data/frames" in k]
        return {
            "ok": frame is not None,
            "frame_key": frame_key,
            "wildcard_key": wildcard,
            "recv_counts": recv,
            "frame_keys_seen": frame_keys,
            "stats": stats,
            "hint": (
                "No frame on subscribe within the probe window. Use the same "
                "ZENOH_CONNECT as the camera driver and match sensor_name to "
                "the driver's frames/<sensor> segment."
            ),
        }

    @property
    def data(self) -> DataBus:
        """Data-layer bus (lazy initialization).

        Returns a :class:`~cyberwave.data.api.DataBus` backed by the
        backend selected via ``CYBERWAVE_DATA_BACKEND``. Scoped to the
        twin set by :meth:`use_data_bus_for` if called, otherwise to
        ``CYBERWAVE_TWIN_UUID``.

        Raises:
            CyberwaveError: If no twin has been pinned and
                ``CYBERWAVE_TWIN_UUID`` is not set.
        """
        if self._data_bus is None:
            twin_uuid = self._data_twin_uuid_override or os.getenv(
                "CYBERWAVE_TWIN_UUID"
            )
            if not twin_uuid:
                raise CyberwaveError(
                    "CYBERWAVE_TWIN_UUID environment variable is required "
                    "for cw.data but is not set.  Export it before accessing "
                    "the data bus, e.g.: export CYBERWAVE_TWIN_UUID=<uuid>"
                )
            self._data_bus = self.data_bus_for(
                twin_uuid, sensor_name=self._data_sensor_name_override
            )
        return self._data_bus

    def _try_get_data_bus(self) -> DataBus | None:
        """Return the data bus if available, None otherwise (no exception)."""
        try:
            return self.data
        except (CyberwaveError, ImportError, Exception):
            return None

    @property
    def mlmodels(self) -> ModelManager:
        """Alias for :attr:`models` (used by generated workflow workers)."""
        return self.models

    @property
    def mqtt(self) -> CyberwaveMQTTClient:
        """Get MQTT client instance (lazy initialization)"""
        if self._mqtt_client is None:
            self._mqtt_client = CyberwaveMQTTClient(self.config)
        return self._mqtt_client

    _QUICKSTART_WORKSPACE_NAME = "Quickstart Workspace"
    _QUICKSTART_PROJECT_NAME = "Quickstart Project"
    _QUICKSTART_ENV_NAME = "Quickstart Environment"

    _WEB_BASE_URL = "https://cyberwave.com"

    def _build_environment_url(self, env_id: str) -> str:
        """Build a user-facing URL for an environment.

        Uses the environment's unified slug when available, falling back to
        the UUID-based URL.
        """
        try:
            env = self.environments.get(env_id)
            slug = getattr(env, "slug", None)
            if slug:
                return f"{self._WEB_BASE_URL}/{slug}"
        except Exception:
            pass
        return f"{self._WEB_BASE_URL}/environments/{env_id}"

    def _resolve_environment_id(self, env_id: str) -> str:
        """Resolve an environment slug to its UUID if needed.

        When *env_id* contains slashes (looks like a slug), the environment
        is fetched by slug and its UUID is returned.  Otherwise *env_id* is
        returned unchanged.
        """
        if "/" in env_id:
            try:
                env = self.environments.get_by_slug(env_id)
                if env is not None:
                    return str(env.uuid)
            except Exception:
                pass
        return env_id

    def _build_twin_url(self, twin_data: Any) -> str:
        """Build a user-facing URL for a twin.

        Uses the twin's unified slug when available, falling back to
        the UUID-based URL.
        """
        slug = getattr(twin_data, "slug", None)
        if isinstance(twin_data, dict):
            slug = twin_data.get("slug", slug)
        if slug:
            return f"{self._WEB_BASE_URL}/{slug}"
        twin_uuid = getattr(twin_data, "uuid", None)
        if isinstance(twin_data, dict):
            twin_uuid = twin_data.get("uuid", twin_uuid)
        return f"{self._WEB_BASE_URL}/twins/{twin_uuid}"

    def get_or_create_quickstart_environment(self) -> tuple[str, bool]:
        """Return ``(environment_uuid, created)`` for the quickstart environment.

        Reuses ``"Quickstart Environment"`` in the active workspace when present.
        May persist ``workspace_id`` / ``environment_id`` on the client config if
        they were unset. Assumes one quickstart env per project.
        """
        return self._get_or_create_quickstart_env()

    def _get_or_create_quickstart_env(self) -> tuple[str, bool]:
        """Internal implementation for :meth:`get_or_create_quickstart_environment`."""
        workspace_id_was_set = bool(self.config.workspace_id)
        workspace_id = self.config.workspace_id
        if not workspace_id:
            workspaces = self.workspaces.list()
            existing_workspace = next(
                (
                    ws
                    for ws in workspaces
                    if getattr(ws, "name", None) == self._QUICKSTART_WORKSPACE_NAME
                ),
                None,
            )
            if existing_workspace:
                workspace_id = existing_workspace.uuid
            elif workspaces:
                workspace_id = workspaces[0].uuid
            else:
                workspace_id = self.workspaces.create(
                    name=self._QUICKSTART_WORKSPACE_NAME,
                ).uuid
        if not workspace_id_was_set:
            self.config.workspace_id = workspace_id

        projects = self.projects.list()
        workspace_projects = [
            p
            for p in projects
            if str(
                getattr(p, "workspace_uuid", None)
                or getattr(p, "workspace_id", None)
                or ""
            )
            == str(workspace_id)
        ]
        # Never fall back to global projects when a workspace is known — that
        # re-attaches quickstart envs to another workspace's Edge Project.
        project_pool = workspace_projects
        existing_project = next(
            (
                p
                for p in project_pool
                if getattr(p, "name", None) == self._QUICKSTART_PROJECT_NAME
            ),
            None,
        )
        if existing_project:
            project_id = existing_project.uuid
        elif project_pool:
            project_id = project_pool[0].uuid
        else:
            project_id = self.projects.create(
                name=self._QUICKSTART_PROJECT_NAME,
                workspace_id=workspace_id,
            ).uuid

        # Single API page; steady state is one quickstart env per project.
        environments = self.environments.list(project_id=project_id)
        existing_env = next(
            (
                e
                for e in environments
                if getattr(e, "name", None) == self._QUICKSTART_ENV_NAME
            ),
            None,
        )
        if existing_env:
            return str(existing_env.uuid), False

        new_env = self.environments.create(
            name=self._QUICKSTART_ENV_NAME,
            project_id=project_id,
        )
        return str(new_env.uuid), True

    def twin(
        self,
        asset_key: Optional[str] = None,
        environment_id: Optional[str] = None,
        twin_id: Optional[str] = None,
        **kwargs,
    ) -> Twin:
        """
        Get or create a twin instance (compact API)

        This is a convenience method for quickly creating twins. The returned
        twin will be an appropriate subclass based on the asset's capabilities:

        - CameraTwin: For assets with RGB sensors (has start_streaming(), etc.)
        - DepthCameraTwin: For assets with depth sensors (has get_point_cloud(), etc.)
        - FlyingTwin: For drones/UAVs (has takeoff(), land(), hover())
        - GripperTwin: For manipulators (has grip(), release())
        - Twin: Base class for assets without special capabilities
        - LocomoteTwin: For assets that can locomote (has move(), etc.)

        Args:
            asset_key: Asset identifier — accepts a registry ID
                (e.g. ``"the-robot-studio/so101"``), a full unified slug
                (e.g. ``"acme/catalog/my-robot-arm"``), or a plain alias.
                Required for creation, optional when *twin_id* is provided.
            environment_id: Environment UUID or unified slug
                (e.g. ``"acme/envs/production-floor"``).  Uses the default
                environment when not provided.
            twin_id: Existing twin UUID or unified slug
                (e.g. ``"acme/twins/arm-station-1"``) to fetch (skips creation).
            **kwargs: Additional twin creation parameters

        Returns:
            Twin instance (or appropriate subclass based on capabilities)

        Example:
            >>> robot = client.twin("unitree/go2")  # Create by registry ID
            >>> robot = client.twin("acme/catalog/go2")  # Create by slug
            >>> robot = client.twin(twin_id="acme/twins/my-go2")  # Fetch by slug
            >>> robot = client.twin(twin_id="uuid")  # Fetch by UUID
            >>> robot.edit_position(x=1, y=0, z=0.5)
        """
        if twin_id:
            twin_data = self.twins.get_raw(twin_id)
            return create_twin(self, twin_data, registry_id=asset_key)

        # asset_key is required for twin creation
        if not asset_key:
            raise CyberwaveError(
                "asset_key is required when creating a new twin (twin_id not provided)"
            )

        twin_name = kwargs.get("name", None)

        env_id = environment_id or self.config.environment_id
        environment_id_was_set = bool(env_id)
        if not env_id:
            env_id, created = self.get_or_create_quickstart_environment()
            if not environment_id_was_set and not self.config.environment_id:
                self.config.environment_id = env_id
            env_url = self._build_environment_url(env_id)
            if created:
                print(
                    f"[Cyberwave] No environment specified — created a new '{self._QUICKSTART_ENV_NAME}'.\n"
                    f"  View it at: {env_url}\n"
                    "  Tip: set environment_id= (or CYBERWAVE_ENVIRONMENT_ID) to skip this step."
                )
            else:
                print(
                    f"[Cyberwave] No environment specified — reusing existing '{self._QUICKSTART_ENV_NAME}'.\n"
                    f"  View it at: {env_url}\n"
                    "  Tip: set environment_id= (or CYBERWAVE_ENVIRONMENT_ID) to skip this step."
                )
        else:
            env_id = self._resolve_environment_id(env_id)

        asset = self.assets.get_by_registry_id(asset_key)
        if asset is None:
            raise CyberwaveError(f"Asset '{asset_key}' not found")

        # Get registry_id for capability lookup
        registry_id = getattr(asset, "registry_id", None) or asset_key

        try:
            existing_twins = self.twins.list(environment_id=env_id)
            for twin_data in existing_twins:
                if twin_data.asset_uuid == asset.uuid and (
                    not twin_name or twin_data.name == twin_name
                ):
                    return create_twin(self, twin_data, registry_id=registry_id)

            twin_data = self.twins.create(
                asset_id=asset.uuid, environment_id=env_id, **kwargs
            )
            return create_twin(self, twin_data, registry_id=registry_id)
        except Exception:
            raise

    def affect(
        self,
        mode: str,
        *,
        environment_id: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> "Cyberwave":
        """
        Set whether commands affect the simulation or the real robot.

        This updates the runtime mode used by high-level command helpers such as
        locomotion APIs, and keeps generic state/telemetry publishers aligned with
        the selected runtime (`edge` in live mode, `sim` in simulation mode).

        Selecting a MuJoCo simulation profile (``"sim"`` / ``"simulation"`` /
        ``"mujoco"``) **auto-starts** a MuJoCo simulation for the resolved
        environment — a billable cloud instance. The lighter ``"playground"``
        profile is a simulation runtime with no MuJoCo instance to start.

        Args:
            mode: ``"sim"`` / ``"simulation"`` / ``"mujoco"`` to target (and start)
                  a MuJoCo simulation, ``"playground"`` for the lightweight
                  kinematic simulation runtime, ``"live"`` / ``"real-world"``
                  (or ``"real"`` / ``"tele"``) to target the real robot.
            environment_id: Environment to start the simulation for. Falls back to
                  the client's configured environment when omitted.
            duration: Optional simulation run duration in seconds (defaults to the
                  backend's own default).

        Returns:
            self, for method chaining.

        Note:
            Locomotion, flight, and driver commands (``twin.locomote.*``,
            ``twin.flying.*``, ``twin.commands.*``) are live/driver-only and raise
            ``NotSimulatedError`` in any simulation runtime.

        Example:
            >>> cw = Cyberwave(environment_id="acme/envs/floor")
            >>> cw.affect("simulation")        # starts a MuJoCo sim (logs the cost)
            >>> frame = robot.get_frame()      # reads from the running sim

            >>> cw.affect("real-world")
            >>> rover.move_forward(1.0)        # moves the real robot
        """
        runtime_mode = _resolve_runtime_mode(mode)
        source_type = _default_state_source_type(runtime_mode)

        if not (
            self.config.runtime_mode == runtime_mode
            and self.config.source_type == source_type
        ):
            self.config.runtime_mode = runtime_mode
            self.config.source_type = source_type

            if self._mqtt_client:
                self._mqtt_client.disconnect()
                self._mqtt_client = None

        # A MuJoCo profile spins up a billable cloud instance; playground/live do not.
        if _affect_autostart_backend(mode) == "mujoco":
            self._autostart_simulation(
                backend="mujoco", environment_id=environment_id, duration=duration
            )

        return self

    def _autostart_simulation(
        self,
        *,
        backend: str,
        environment_id: Optional[str],
        duration: Optional[float],
    ) -> Any:
        """Start (or reuse) a simulation selected via :meth:`affect`, logging the cost.

        Resolves the environment from ``environment_id`` or the client's configured
        environment. When neither is available, logs how to start one and returns
        ``None`` rather than raising — the runtime mode is still set. Blocks until
        the simulation reports ``running`` (mirrors every other simulation-start
        path in the SDK) so that a getter called right after ``affect()`` returns
        never races a still-``loading`` instance.
        """
        from cyberwave.managers.simulations import (
            SIMULATION_CREDITS_PER_HOUR,
            SIMULATION_CREDITS_PER_MINUTE,
        )

        env_id = environment_id or self.config.environment_id
        if not env_id:
            logger.warning(
                "affect(%r) selected a MuJoCo simulation runtime, but no environment "
                "is set — no simulation was started. Pass environment_id=... to "
                "affect(), set CYBERWAVE_ENVIRONMENT_ID, or start one explicitly with "
                "cw.environments.simulations.start(environment_id, backend='mujoco').",
                backend,
            )
            return None

        env_id = self._resolve_environment_id(env_id)
        simulations = self.environments.simulations

        sim = simulations.get_active(env_id)
        if sim is not None:
            logger.info(
                "Reusing active simulation %s for environment %s "
                "(MuJoCo simulations are billed at ~%.2f credits/hour).",
                sim.simulation_id,
                env_id,
                SIMULATION_CREDITS_PER_HOUR,
            )
        else:
            sim = simulations.start(env_id, backend=backend, duration=duration)
            logger.warning(
                "Started MuJoCo simulation %s for environment %s — this is a billable "
                "cloud instance consuming credits at ~%.2f credits/hour "
                "(%.2f credits/min). Stop it with sim.stop() when done.",
                sim.simulation_id,
                env_id,
                SIMULATION_CREDITS_PER_HOUR,
                SIMULATION_CREDITS_PER_MINUTE,
            )

        if sim.status != "running":
            sim.wait_until_active()
        return sim

    def configure(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        environment_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Update client configuration

        Args:
            base_url: Base URL of the Cyberwave backend
            api_key: API key for authentication
            token: Deprecated alias for api_key
            environment_id: Default environment ID
            workspace_id: Default workspace ID
            **kwargs: Additional configuration options
        """
        if base_url:
            self.config.base_url = base_url
        if api_key:
            self.config.api_key = api_key
        elif token:
            warnings.warn(
                "'token' is deprecated and will be removed in a future release. "
                "Use 'api_key' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.config.api_key = token
        if environment_id:
            self.config.environment_id = environment_id
        if workspace_id:
            self.config.workspace_id = workspace_id

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._setup_rest_client()
        self._init_managers()

        if self._mqtt_client:
            self._mqtt_client.disconnect()
            self._mqtt_client = None

    @property
    def time_reference(self) -> TimeReference:
        """Get a shared time reference for synchronization."""
        if not hasattr(self, "_time_reference"):
            self._time_reference = TimeReference()
        return self._time_reference

    def video_stream(
        self,
        twin_uuid: str,
        camera_type: str = "cv2",
        camera_id: int | str = 0,
        fps: int = 30,
        resolution: Optional[tuple] = None,
        enable_depth: bool = False,
        depth_fps: int = 30,
        depth_resolution: Optional[tuple] = None,
        auto_detect: bool = True,
        turn_servers: Optional[list] = None,
        time_reference: Optional[TimeReference] = None,
        keyframe_interval: Optional[int] = None,
        frame_callback: Optional[Callable] = None,
        depth_callback: Optional[Callable] = None,
        camera_name: Optional[str] = None,
        fourcc: Optional[str] = None,
    ):
        """
        Create a camera streamer for the specified twin. DEPRECATED: Use the TwinCamera instead

        This method creates a camera streamer instance that's pre-configured with
        the client's MQTT connection, providing a seamless experience for streaming
        video to digital twins.

        Supports:
        - Local cameras: camera_id=0, camera_id=1 (device index)
        - IP cameras: camera_id="http://192.168.1.100/snapshot.jpg"
        - RTSP streams: camera_id="rtsp://192.168.1.100:554/stream"
        - Intel RealSense: camera_type="realsense"

        Args:
            twin_uuid: UUID of the digital twin to stream to
            camera_type: Camera type - "cv2" for USB/webcam/IP, "realsense" for Intel RealSense
            camera_id: Camera device ID (int) or stream URL (str) (default: 0)
            fps: Frames per second (default: 30)
            resolution: Video resolution as (width, height) tuple (default: 640x480)
            enable_depth: Enable depth streaming for RealSense (default: False)
            depth_fps: Depth stream FPS for RealSense (default: 30)
            depth_resolution: Depth resolution as (width, height) tuple (default: same as color)
            auto_detect: Auto-detect RealSense capabilities (default: True)
            turn_servers: Optional list of TURN server configurations
            time_reference: Optional time reference for synchronization
            keyframe_interval: Force a keyframe every N frames for better streaming start.
                If None, uses CYBERWAVE_KEYFRAME_INTERVAL env var, or disables forced keyframes.
                Recommended: fps * 2 (e.g., 60 for 30fps = keyframe every 2 seconds)
            frame_callback: Optional per-frame color callback (ML inference, etc.).
                Signature: callback(frame: np.ndarray, frame_count: int) -> None
            depth_callback: Optional per-frame depth callback (RealSense only).
                Signature: callback(depth: np.ndarray, frame_count: int) -> None
            camera_name: Optional sensor identifier for multi-stream twins.
            fourcc: Optional FOURCC for local V4L2/USB cameras (e.g. ``'MJPG'``, ``'YUYV'``).
                Passed to :class:`~cyberwave.sensor.camera_cv2.CV2VideoTrack`. If omitted for a
                local device, the SDK tries ``MJPG`` by default. Ignored for RealSense and IP/RTSP cameras.

        Returns:
            Camera streamer instance (CV2CameraStreamer or RealSenseStreamer)

        Example:
            >>> client = Cyberwave(api_key="your_api_key")
            >>>
            >>> # Local USB camera
            >>> streamer = client.video_stream(
            ...     twin_uuid="your_twin_uuid",
            ...     camera_type="cv2",
            ...     camera_id=0,
            ...     resolution=(1280, 720),
            ...     fps=30
            ... )
            >>>
            >>> # IP camera / RTSP stream
            >>> streamer = client.video_stream(
            ...     twin_uuid="your_twin_uuid",
            ...     camera_type="cv2",
            ...     camera_id="rtsp://192.168.1.100:554/stream",
            ...     fps=15
            ... )
            >>>
            >>> # RealSense camera with depth
            >>> streamer = client.video_stream(
            ...     twin_uuid="your_twin_uuid",
            ...     camera_type="realsense",
            ...     resolution=(1280, 720),
            ...     enable_depth=True,
            ...     auto_detect=True
            ... )
            >>>
            >>> await streamer.start()

        Raises:
            ImportError: If camera dependencies are not installed
            CyberwaveError: If camera type is not supported
        """
        if not _has_camera:
            raise ImportError(
                "Camera streaming requires additional dependencies. "
                "Install them with: pip install cyberwave[camera]"
            )

        if self._mqtt_client is None:
            self.mqtt.connect()

        self.mqtt.connect()
        self.mqtt._client._handle_twin_update_with_telemetry(twin_uuid)

        # Use shared time reference if not provided
        if time_reference is None:
            time_reference = self.time_reference

        # Default resolution
        if resolution is None:
            resolution = (640, 480)

        camera_type_lower = camera_type.lower()

        if camera_type_lower == "cv2":
            return CameraStreamer(
                client=self.mqtt,
                camera_id=camera_id,
                fps=fps,
                resolution=resolution,
                turn_servers=turn_servers,
                twin_uuid=twin_uuid,
                time_reference=time_reference,
                keyframe_interval=keyframe_interval,
                frame_callback=frame_callback,
                camera_name=camera_name,
                fourcc=fourcc,
            )
        elif camera_type_lower == "realsense":
            if not _has_realsense:
                raise ImportError(
                    "RealSense camera support requires additional dependencies. "
                    "Install them with: pip install cyberwave[realsense]"
                )

            # Import Resolution for RealSense
            from cyberwave.sensor import Resolution

            # Convert tuple to Resolution enum for from_device
            def to_resolution(res):
                if isinstance(res, tuple):
                    return Resolution.from_size(res[0], res[1]) or Resolution.closest(
                        res[0], res[1]
                    )
                return res

            # Depth resolution defaults to color resolution
            if depth_resolution is None:
                depth_resolution = resolution

            if auto_detect:
                return RealSenseStreamer.from_device(
                    client=self.mqtt,
                    prefer_resolution=to_resolution(resolution),
                    prefer_fps=fps,
                    enable_depth=enable_depth,
                    turn_servers=turn_servers,
                    twin_uuid=twin_uuid,
                    time_reference=time_reference,
                    camera_name=camera_name,
                    frame_callback=frame_callback,
                    depth_callback=depth_callback,
                )
            else:
                return RealSenseStreamer(
                    client=self.mqtt,
                    color_fps=fps,
                    depth_fps=depth_fps,
                    color_resolution=resolution,
                    depth_resolution=depth_resolution,
                    enable_depth=enable_depth,
                    turn_servers=turn_servers,
                    twin_uuid=twin_uuid,
                    time_reference=time_reference,
                    camera_name=camera_name,
                    frame_callback=frame_callback,
                    depth_callback=depth_callback,
                )
        else:
            raise CyberwaveError(
                f"Unsupported camera type: {camera_type}. "
                "Supported types: 'cv2', 'realsense'"
            )

    def controller(
        self,
        twin_uuid: str,
    ) -> "EdgeController":
        """
        Create an edge controller for the specified twin. DEPRECATED

        This method creates an EdgeController instance that's pre-configured with
        the client's MQTT connection, providing a seamless experience for sending
        commands to edge devices.

        Args:
            twin_uuid: UUID of the digital twin to control

        Returns:
            EdgeController instance ready to start

        Example:
            >>> client = Cyberwave(api_key="your_api_key")
            >>> controller = client.controller(twin_uuid="your_twin_uuid")
            >>> await controller.start()

        """
        if self._mqtt_client is None:
            self.mqtt.connect()

        return EdgeController(
            client=self.mqtt,
            twin_uuid=twin_uuid,
        )

    def get_scene(self, environment_id: str) -> "Scene":
        """Get a scene builder for the specified environment."""
        from cyberwave.scene import Scene

        return Scene(self, environment_id)

    # ── Hook decorator delegation ────────────────────────────────

    @property
    def on_frame(self) -> Callable:
        return self._hook_registry.on_frame

    @property
    def on_depth(self) -> Callable:
        return self._hook_registry.on_depth

    @property
    def on_audio(self) -> Callable:
        return self._hook_registry.on_audio

    @property
    def on_pointcloud(self) -> Callable:
        return self._hook_registry.on_pointcloud

    @property
    def on_imu(self) -> Callable:
        return self._hook_registry.on_imu

    @property
    def on_force_torque(self) -> Callable:
        return self._hook_registry.on_force_torque

    @property
    def on_joint_states(self) -> Callable:
        return self._hook_registry.on_joint_states

    @property
    def on_attitude(self) -> Callable:
        return self._hook_registry.on_attitude

    @property
    def on_gps(self) -> Callable:
        return self._hook_registry.on_gps

    @property
    def on_end_effector_pose(self) -> Callable:
        return self._hook_registry.on_end_effector_pose

    @property
    def on_gripper_state(self) -> Callable:
        return self._hook_registry.on_gripper_state

    @property
    def on_map(self) -> Callable:
        return self._hook_registry.on_map

    @property
    def on_battery(self) -> Callable:
        return self._hook_registry.on_battery

    @property
    def on_alert(self) -> Callable:
        return self._hook_registry.on_alert

    @property
    def on_mqtt(self) -> Callable:
        return self._hook_registry.on_mqtt

    @property
    def on_manual_trigger(self) -> Callable:
        return self._hook_registry.on_manual_trigger

    @property
    def on_temperature(self) -> Callable:
        return self._hook_registry.on_temperature

    @property
    def on_lidar(self) -> Callable:
        return self._hook_registry.on_lidar

    @property
    def on_data(self) -> Callable:
        return self._hook_registry.on_data

    @property
    def on_schedule(self) -> Callable:
        return self._hook_registry.on_schedule

    @property
    def on_synchronized(self) -> Callable:
        return self._hook_registry.on_synchronized

    # ── Worker runtime helpers ───────────────────────────────────

    def publish_event(
        self,
        twin_uuid: str,
        event_type: str,
        data: dict[str, Any],
        *,
        source: str = "edge_node",
    ) -> None:
        """Publish a business event via MQTT.

        Published payload shape::

            {"event_type": ..., "source": ..., "data": ..., "timestamp": ...}
        """
        logger.debug(
            "publish_event: twin=%s type=%s data=%s",
            twin_uuid[:8] + "...",
            event_type,
            data,
        )
        prefix = self.mqtt.topic_prefix
        self.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/event",
            {
                "event_type": event_type,
                "source": source,
                "data": data,
                "timestamp": time.time(),
            },
        )

    def publish_alert(
        self,
        twin_uuid: str,
        name: str,
        *,
        description: str = "",
        alert_type: str = "",
        severity: str = "info",
        category: str = "business",
        force: bool = False,
        source_type: str = "edge",
        metadata: Optional[dict[str, Any]] = None,
        workflow_uuid: Optional[str] = None,
        workflow_node_uuid: Optional[str] = None,
        workflow_execution_uuid: Optional[str] = None,
    ) -> None:
        """Create a business alert via the REST API.

        This is a fire-and-forget convenience used by generated edge workers
        to surface detection events as operator-visible alerts.  The backend
        handles deduplication, MQTT relay, and notification dispatch.

        Args:
            twin_uuid: UUID of the twin the alert is attached to.
            name: Human-readable alert title.
            description: Optional details (model name, confidence, etc.).
            alert_type: Machine-readable type code (e.g. ``person_detected``).
            severity: One of ``info``, ``warning``, ``error``, ``critical``.
            category: ``business`` (default) or ``technical``.
            force: If True, bypass backend alert deduplication.
            source_type: Alert origin (``edge``, ``cloud``, ``simulation``, etc.).
            metadata: Optional dict of extra data stored on the alert.
            workflow_uuid: UUID of the workflow that produced the alert. When
                provided, the alert becomes queryable via
                ``GET /api/v1/alerts?workflow_uuid=...``.
            workflow_node_uuid: UUID of the workflow node (e.g. ``send_alert``)
                that produced the alert. Stored under ``metadata`` for
                provenance — does not require a schema change on the backend.
            workflow_execution_uuid: UUID of the workflow execution that
                produced the alert. Stored under ``metadata`` for provenance.
        """
        from cyberwave.alerts import _create_alert

        payload: dict[str, Any] = {
            "name": name,
            "description": description,
            "alert_type": alert_type,
            "severity": severity.lower(),
            "source_type": source_type,
            "category": category,
            "twin_uuid": twin_uuid,
        }
        if self.config.workspace_id:
            payload["workspace_uuid"] = self.config.workspace_id
        if workflow_uuid:
            payload["workflow_uuid"] = workflow_uuid
        if force:
            payload["force"] = True

        # Auto-merge workflow node/execution provenance into metadata so callers
        # don't have to remember; explicit user-supplied keys win.
        merged_metadata: dict[str, Any] = {}
        if workflow_uuid:
            merged_metadata["workflow_uuid"] = workflow_uuid
        if workflow_node_uuid:
            merged_metadata["workflow_node_uuid"] = workflow_node_uuid
        if workflow_execution_uuid:
            merged_metadata["workflow_execution_uuid"] = workflow_execution_uuid
        if metadata is not None:
            merged_metadata.update(metadata)
        if merged_metadata:
            payload["metadata"] = merged_metadata

        try:
            _create_alert(self, payload)
            logger.info(
                "publish_alert: created alert type=%s for twin=%s",
                alert_type,
                twin_uuid[:8] + "...",
            )
        except Exception:
            logger.exception(
                "publish_alert: failed to create alert type=%s for twin=%s",
                alert_type,
                twin_uuid[:8] + "...",
            )

    def run_edge_workers(self, workers_dir: str | None = None) -> None:
        """Start the edge worker runtime: load workers, activate hooks, block.

        Creates a :class:`~cyberwave.workers.runtime.WorkerRuntime`, loads
        worker modules from *workers_dir*, wires hooks to data-layer
        subscriptions, and blocks until ``stop()`` or a signal is received.

        The runtime is stored as ``self._runtime`` so that external code
        (e.g. a health-check thread or edge-core shutdown message) can call
        ``client._runtime.stop()`` without relying on signals.
        """
        from cyberwave.workers.runtime import WorkerRuntime

        try:
            self.mqtt.connect()
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "MQTT connect failed; events will be dropped until reconnected",
                exc_info=True,
            )
        self._runtime = WorkerRuntime(self)
        self._runtime.load(workers_dir)
        self._runtime.start()
        self._runtime.run()

    def disconnect(self):
        """Disconnect all connections (REST, MQTT, and data bus)."""
        if self._mqtt_client:
            self._mqtt_client.disconnect()
        self._data_bus = None
        if self._data_backend is not None:
            close_frame_subscribe_caches_for_backend(self._data_backend)
            self._data_backend.close()
            self._data_backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
