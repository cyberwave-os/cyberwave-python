"""Compass namespace — ``twin.compass`` or ``twin.compasses[<id>]``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sensors import compass_handle_for_key
from .base import SensorFamilyNamespace

if TYPE_CHECKING:
    from ..base import Twin

COMPASS_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get_heading")


class CompassesNamespace(SensorFamilyNamespace):
    """Keyed access to per-sensor compass handles."""

    def __init__(self, twin: "Twin") -> None:
        super().__init__(
            twin,
            handler_key="compass",
            family_label="compass",
            public_methods=COMPASS_HANDLE_PUBLIC_METHODS,
            handle_for_key=compass_handle_for_key,
        )
