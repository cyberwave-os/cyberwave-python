from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cyberwave.driver import BaseDriver, DriverOperationMode
from cyberwave.driver.base import resolve_twin_attached_controller


def _make_minimal_driver(twin: object) -> BaseDriver:
    class _D(BaseDriver):
        REGISTRY_ID = "test/driver"

        async def on_configure(self) -> None:
            pass

        async def on_connect_to_device(self) -> None:
            pass

        async def on_register_callbacks(self) -> None:
            pass

        async def on_activate(self) -> None:
            pass

        async def on_shutdown(self) -> None:
            pass

        @classmethod
        def create(cls) -> _D:
            return cls.__new__(cls)

    driver = _D.__new__(_D)
    driver._twin = twin
    driver._cw = None
    driver._operation_mode = DriverOperationMode.NO_OP
    driver._set_operation_mode = AsyncMock()
    return driver


class _SdkTwin:
    def __init__(
        self,
        *,
        controller_policy_uuid: str | None,
        metadata: dict | None = None,
        uuid: str = "twin-uuid",
    ) -> None:
        self.uuid = uuid
        self._data = SimpleNamespace(
            controller_policy_uuid=controller_policy_uuid,
            metadata=metadata or {},
        )

    def _attached_controller_policy_uuid(self) -> str | None:
        raw = self._data.controller_policy_uuid
        return str(raw) if raw else None

    @property
    def metadata(self) -> dict:
        return dict(self._data.metadata)


@pytest.mark.asyncio
async def test_no_controller_sets_no_op() -> None:
    twin = _SdkTwin(controller_policy_uuid=None)
    driver = _make_minimal_driver(twin)
    await driver._derive_initial_operation_mode()
    driver._set_operation_mode.assert_awaited_once_with(DriverOperationMode.NO_OP)


@pytest.mark.asyncio
async def test_attached_remote_controller_sets_teleop_remote() -> None:
    twin = _SdkTwin(
        controller_policy_uuid="policy-uuid",
        metadata={"controller_type": "keyboard"},
    )
    driver = _make_minimal_driver(twin)
    await driver._derive_initial_operation_mode()
    driver._set_operation_mode.assert_awaited_once_with(
        DriverOperationMode.TELEOP_REMOTE
    )


@pytest.mark.asyncio
async def test_attached_localop_controller_sets_teleop_local() -> None:
    twin = _SdkTwin(
        controller_policy_uuid="policy-uuid",
        metadata={"controller_type": "localop"},
    )
    driver = _make_minimal_driver(twin)
    await driver._derive_initial_operation_mode()
    driver._set_operation_mode.assert_awaited_once_with(
        DriverOperationMode.TELEOP_LOCAL
    )


@pytest.mark.asyncio
async def test_startup_refreshes_twin_via_twins_get() -> None:
    refreshed = _SdkTwin(
        controller_policy_uuid="refreshed-policy",
        metadata={"controller_type": "gamepad"},
    )
    stale = _SdkTwin(controller_policy_uuid=None)
    driver = _make_minimal_driver(stale)
    driver._cw = MagicMock()
    driver._cw.twins.get.return_value = refreshed

    await driver._derive_initial_operation_mode()

    driver._cw.twins.get.assert_called_once_with("twin-uuid")
    assert driver._twin is refreshed
    driver._set_operation_mode.assert_awaited_once_with(
        DriverOperationMode.TELEOP_REMOTE
    )


def test_resolve_twin_attached_controller_reads_data_not_top_level_attr() -> None:
    twin = _SdkTwin(
        controller_policy_uuid="from-data",
        metadata={"controller_type": "localop"},
    )
    assert resolve_twin_attached_controller(twin) == ("from-data", "localop")
    assert getattr(twin, "controller_policy_uuid", None) is None
