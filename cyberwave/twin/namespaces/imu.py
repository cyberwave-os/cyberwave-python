"""IMU namespace — ``twin.imu`` or ``twin.imus[<id>]``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sensors import imu_handle_for_key
from .base import SensorFamilyNamespace

if TYPE_CHECKING:
    from ..base import Twin

IMU_HANDLE_PUBLIC_METHODS: tuple[str, ...] = ("metadata", "get", "get_sample")


class ImusNamespace(SensorFamilyNamespace):
    """Keyed access to per-sensor IMU handles."""

    def __init__(self, twin: "Twin") -> None:
        super().__init__(
            twin,
            handler_key="imu",
            family_label="imu",
            public_methods=IMU_HANDLE_PUBLIC_METHODS,
            handle_for_key=imu_handle_for_key,
        )
