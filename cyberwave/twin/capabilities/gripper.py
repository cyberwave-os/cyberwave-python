"""Gripper command handle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..base import Twin


class GripperHandle:
    """Grouped gripper commands."""

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def grip(self, force: float = 1.0, *, source_type: Optional[str] = None) -> None:
        resolved = self._twin._resolve_topic_and_payload(
            command="grip",
            data={"force": max(0.0, min(1.0, force))},
            source_type=source_type,
        )
        self._twin._publish_resolved(resolved)

    def release(self, *, source_type: Optional[str] = None) -> None:
        resolved = self._twin._resolve_topic_and_payload(
            command="release",
            data={},
            source_type=source_type,
        )
        self._twin._publish_resolved(resolved)
