"""Flashlight namespace — ``twin.flashlight`` or ``twin.flashlights[<id>]``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sensors import flashlight_handle_for_key
from .base import SensorFamilyNamespace

if TYPE_CHECKING:
    from ..base import Twin

FLASHLIGHT_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "set")


class FlashlightsNamespace(SensorFamilyNamespace):
    """Keyed access to per-sensor flashlight handles."""

    def __init__(self, twin: "Twin") -> None:
        super().__init__(
            twin,
            handler_key="flashlight",
            family_label="flashlight",
            public_methods=FLASHLIGHT_HANDLE_PUBLIC_METHODS,
            handle_for_key=flashlight_handle_for_key,
        )
