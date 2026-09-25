"""Runtime-mode partitioning for MQTT inbound twin state (live vs simulation)."""

from __future__ import annotations

import threading
from typing import Any

from ..constants import (
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_EDGE_FOLLOWER,
    SOURCE_TYPE_EDGE_LEADER,
    SOURCE_TYPE_SIM,
    SOURCE_TYPE_SIM_TELE,
    SOURCE_TYPE_TELE,
)

RUNTIME_MODE_LIVE = "live"
RUNTIME_MODE_SIMULATION = "simulation"
RUNTIME_MODES = (RUNTIME_MODE_LIVE, RUNTIME_MODE_SIMULATION)

_SIM_MQTT_SOURCE_TYPES = frozenset(
    {
        SOURCE_TYPE_SIM,
        SOURCE_TYPE_SIM_TELE,
        "simulation",
        "edit",
    }
)
_LIVE_MQTT_SOURCE_TYPES = frozenset(
    {
        SOURCE_TYPE_TELE,
        SOURCE_TYPE_EDGE,
        SOURCE_TYPE_EDGE_FOLLOWER,
        SOURCE_TYPE_EDGE_LEADER,
    }
)


def active_runtime_mode(client: Any) -> str:
    """Return ``config.runtime_mode`` as a stable bucket key (``live`` or ``simulation``)."""
    config = getattr(client, "config", None)
    mode = getattr(config, "runtime_mode", RUNTIME_MODE_LIVE) if config else RUNTIME_MODE_LIVE
    if isinstance(mode, str) and mode.strip().lower() == RUNTIME_MODE_SIMULATION:
        return RUNTIME_MODE_SIMULATION
    return RUNTIME_MODE_LIVE


def runtime_mode_from_mqtt_source_type(
    source_type: Any,
    *,
    client: Any | None = None,
) -> str:
    """Map inbound MQTT ``source_type`` to ``live`` or ``simulation``.

    When *source_type* is missing or unrecognized, falls back to
    :func:`active_runtime_mode` so legacy payloads without the field still
    populate the bucket the SDK is configured for.
    """
    if source_type is not None:
        normalized = str(source_type).strip().lower()
        if normalized in _SIM_MQTT_SOURCE_TYPES:
            return RUNTIME_MODE_SIMULATION
        if normalized in _LIVE_MQTT_SOURCE_TYPES:
            return RUNTIME_MODE_LIVE
    if client is not None:
        return active_runtime_mode(client)
    return RUNTIME_MODE_LIVE


def new_runtime_ready_events() -> dict[str, threading.Event]:
    """Fresh per-mode readiness events for MQTT first-read gating."""
    return {mode: threading.Event() for mode in RUNTIME_MODES}
