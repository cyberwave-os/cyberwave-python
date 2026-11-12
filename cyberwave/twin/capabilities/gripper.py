"""Gripper command handle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..base import Twin


#: Grip and release set absolute jaw endpoints, so a replayed copy is harmless
#: and at-least-once delivery is the right trade for a one-shot command.
_ABSOLUTE_TARGET_QOS = 1


class GripperHandle:
    """Grouped gripper commands."""

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def grip(
        self,
        force: float = 1.0,
        *,
        max_effort_nm: Optional[float] = None,
        source_type: Optional[str] = None,
    ) -> None:
        """Close the gripper.

        ``max_effort_nm`` caps the gripper-motor torque (N.m) so a grasp does not
        crush the object; the valid range is driver-specific (Piper: 0.5-3.0).
        Omit it for the driver's configured default.
        """
        data: dict[str, float] = {"force": max(0.0, min(1.0, force))}
        if max_effort_nm is not None:
            data["max_effort_nm"] = float(max_effort_nm)
        resolved = self._twin._resolve_topic_and_payload(
            command="grip",
            data=data,
            source_type=source_type,
            qos=_ABSOLUTE_TARGET_QOS,
        )
        self._twin._publish_resolved(resolved)

    def release(self, *, source_type: Optional[str] = None) -> None:
        resolved = self._twin._resolve_topic_and_payload(
            command="release",
            data={},
            source_type=source_type,
            qos=_ABSOLUTE_TARGET_QOS,
        )
        self._twin._publish_resolved(resolved)
