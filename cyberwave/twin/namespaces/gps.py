"""GPS namespace — ``twin.gps`` or ``twin.gpss[<id>]``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sensors import gps_handle_for_key
from .base import SensorFamilyNamespace

if TYPE_CHECKING:
    from ..base import Twin

GPS_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get_fix")


class GpssNamespace(SensorFamilyNamespace):
    """Keyed access to per-sensor GPS handles."""

    def __init__(self, twin: "Twin") -> None:
        super().__init__(
            twin,
            handler_key="gps",
            family_label="gps",
            public_methods=GPS_HANDLE_PUBLIC_METHODS,
            handle_for_key=gps_handle_for_key,
        )
