"""Per-sample context passed to every hook callback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    """Per-sample context passed as the second argument to every hook callback.

    Attributes:
        timestamp: Sample time from the data layer (not wall-clock at dispatch).
        channel: Resolved data channel (e.g. ``"frames/default"``).
        sensor_name: Extracted from the channel suffix (e.g. ``"front"``
            from ``"frames/front"``).
        twin_uuid: Set by the runtime from ``cw.config.twin_uuid``.
            ``None`` when not yet bound to a twin.
        metadata: Extra fields the data layer attached (e.g. frame
            dimensions, encoding).
    """

    timestamp: float
    channel: str
    sensor_name: str = "default"
    twin_uuid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
