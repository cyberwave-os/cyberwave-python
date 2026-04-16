"""Backend configuration and factory for the data layer.

The :func:`get_backend` factory reads ``CYBERWAVE_DATA_BACKEND`` and related
environment variables to select and configure the appropriate
:class:`~.backend.DataBackend`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .backend import DataBackend
from .exceptions import BackendConfigError

SUPPORTED_BACKENDS = ("zenoh", "filesystem")
PUBLISH_MODES = ("dual", "zenoh_only", "mqtt_only")


def _parse_bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BackendConfig:
    """Configuration for the data backend.

    All fields fall back to environment variables when left at their defaults.
    """

    backend: str = ""
    """``"zenoh"`` or ``"filesystem"``.  Env: ``CYBERWAVE_DATA_BACKEND``."""

    zenoh_connect: list[str] = field(default_factory=list)
    """Zenoh router endpoints.  Env: ``ZENOH_CONNECT`` (comma-separated)."""

    zenoh_listen: list[str] = field(default_factory=list)
    """Zenoh listener endpoints.  Env: ``ZENOH_LISTEN`` (comma-separated).

    When set, the Zenoh session binds a TCP listener so external peers
    (e.g. the CLI monitor) can connect without multicast discovery.
    Example: ``tcp/0.0.0.0:7447``.
    """

    zenoh_shared_memory: bool | None = None
    """Enable Zenoh shared-memory transport.  Env: ``ZENOH_SHARED_MEMORY``.

    Pass ``True`` or ``False`` to override the env var.  Leave ``None`` to
    read ``ZENOH_SHARED_MEMORY`` at construction time.
    """

    filesystem_base_dir: str | None = None
    """Root for filesystem backend.  Env: ``CYBERWAVE_DATA_DIR``."""

    filesystem_ring_buffer_size: int = 100
    """Max samples per channel (filesystem only)."""

    key_prefix: str = "cw"
    """Key prefix for Zenoh key expressions (used by ``DataBus``, not the backend)."""

    publish_mode: str = ""
    """Controls which transport paths are active.

    * ``"dual"`` (default) — publish on both MQTT and Zenoh.
    * ``"zenoh_only"`` — publish on Zenoh only (no MQTT cloud path).
    * ``"mqtt_only"`` — publish on MQTT only (legacy mode, Zenoh disabled).

    Env: ``CYBERWAVE_PUBLISH_MODE``.
    """

    def __post_init__(self) -> None:
        if not self.backend:
            self.backend = os.environ.get("CYBERWAVE_DATA_BACKEND", "zenoh")

        if not self.zenoh_connect:
            connect_env = os.environ.get("ZENOH_CONNECT", "")
            if connect_env:
                self.zenoh_connect = [
                    e.strip() for e in connect_env.split(",") if e.strip()
                ]

        if not self.zenoh_listen:
            listen_env = os.environ.get("ZENOH_LISTEN", "")
            if listen_env:
                self.zenoh_listen = [
                    e.strip() for e in listen_env.split(",") if e.strip()
                ]

        if self.zenoh_shared_memory is None:
            self.zenoh_shared_memory = _parse_bool_env(
                os.environ.get("ZENOH_SHARED_MEMORY"),
            )

        if not self.filesystem_base_dir:
            self.filesystem_base_dir = os.environ.get("CYBERWAVE_DATA_DIR")

        if not self.publish_mode:
            import warnings

            raw = os.environ.get("CYBERWAVE_PUBLISH_MODE", "dual").strip().lower()
            if raw not in PUBLISH_MODES:
                warnings.warn(
                    f"Unknown CYBERWAVE_PUBLISH_MODE '{raw}'; falling back to 'dual'. "
                    f"Valid values: {', '.join(PUBLISH_MODES)}.",
                    stacklevel=2,
                )
                raw = "dual"
            self.publish_mode = raw


def is_zenoh_publish_enabled(config: BackendConfig | None = None) -> bool:
    """Return True when the publish mode includes a Zenoh path."""
    cfg = config or BackendConfig()
    return cfg.publish_mode in ("dual", "zenoh_only")


def is_mqtt_publish_enabled(config: BackendConfig | None = None) -> bool:
    """Return True when the publish mode includes an MQTT path."""
    cfg = config or BackendConfig()
    return cfg.publish_mode in ("dual", "mqtt_only")


def get_backend(config: BackendConfig | None = None) -> DataBackend:
    """Create the appropriate :class:`DataBackend` from *config* / env vars.

    Raises:
        BackendConfigError: If the requested backend name is not recognised.
        BackendUnavailableError: If the backend cannot be initialised (e.g.
            ``eclipse-zenoh`` is not installed).
    """
    cfg = config or BackendConfig()

    if cfg.backend == "zenoh":
        from .zenoh_backend import ZenohBackend

        return ZenohBackend(
            connect=cfg.zenoh_connect or None,
            listen=cfg.zenoh_listen or None,
            shared_memory=bool(cfg.zenoh_shared_memory),
        )

    if cfg.backend == "filesystem":
        from .filesystem_backend import FilesystemBackend

        return FilesystemBackend(
            base_dir=cfg.filesystem_base_dir,
            ring_buffer_size=cfg.filesystem_ring_buffer_size,
        )

    raise BackendConfigError(
        f"Unknown data backend: '{cfg.backend}'.  "
        f"Supported values: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}.  "
        f"Set CYBERWAVE_DATA_BACKEND to a supported value."
    )
