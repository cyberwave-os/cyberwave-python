"""LiDAR namespace — ``twin.lidar`` or ``twin.lidars[<id>]``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sensors import lidar_handle_for_key
from .base import SensorFamilyNamespace

if TYPE_CHECKING:
    from ..base import Twin

LIDAR_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get_scan")


class LidarsNamespace(SensorFamilyNamespace):
    """Keyed access to per-sensor LiDAR handles."""

    def __init__(self, twin: "Twin") -> None:
        super().__init__(
            twin,
            handler_key="lidar",
            family_label="lidar",
            public_methods=LIDAR_HANDLE_PUBLIC_METHODS,
            handle_for_key=lidar_handle_for_key,
        )
