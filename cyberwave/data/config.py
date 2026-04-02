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
    """Key prefix prepended to channel names in Zenoh key expressions."""

    def __post_init__(self) -> None:
        if not self.backend:
            self.backend = os.environ.get("CYBERWAVE_DATA_BACKEND", "zenoh")

        if not self.zenoh_connect:
            connect_env = os.environ.get("ZENOH_CONNECT", "")
            if connect_env:
                self.zenoh_connect = [
                    e.strip() for e in connect_env.split(",") if e.strip()
                ]

        if self.zenoh_shared_memory is None:
            self.zenoh_shared_memory = _parse_bool_env(
                os.environ.get("ZENOH_SHARED_MEMORY"),
            )

        if not self.filesystem_base_dir:
            self.filesystem_base_dir = os.environ.get("CYBERWAVE_DATA_DIR")


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
            shared_memory=bool(cfg.zenoh_shared_memory),
            key_prefix=cfg.key_prefix,
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
