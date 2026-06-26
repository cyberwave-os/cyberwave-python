"""Lifecycle transitions raise driver lifecycle alerts on BaseDriver."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cyberwave.driver.base import BaseDriver, DriverLifecycleState
from cyberwave.driver.cloud.alerts import AlertCode


def _twin_with_alerts() -> SimpleNamespace:
    return SimpleNamespace(
        uuid="twin-uuid",
        environment_id="env-uuid",
        name="test-twin",
        alerts=SimpleNamespace(
            create=MagicMock(return_value=SimpleNamespace(uuid="alert-1"))
        ),
    )


class _MinimalDriver(BaseDriver):
    REGISTRY_ID = "vendor/robot"

    def on_configure(self) -> None:
        pass

    def on_connect_to_device(self) -> None:
        pass

    def on_register_callbacks(self) -> None:
        pass

    def on_activate(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass

    def on_tick(self) -> None:
        pass

    @classmethod
    def create(cls) -> _MinimalDriver:
        return cls(twin=_twin_with_alerts())


def test_transition_to_raises_lifecycle_and_active_alerts() -> None:
    driver = _MinimalDriver(twin=_twin_with_alerts())

    with patch.object(driver._alert_manager, "raise_alert") as raise_alert:
        with patch.object(driver._alert_manager, "resolve_alert") as resolve_alert:
            driver._transition_to(DriverLifecycleState.CONFIGURING)
            driver._transition_to(DriverLifecycleState.ACTIVE)

    assert raise_alert.call_count == 3
    configuring_alert = raise_alert.call_args_list[0][0][0]
    active_lifecycle_alert = raise_alert.call_args_list[1][0][0]
    active_alert = raise_alert.call_args_list[2][0][0]
    assert configuring_alert.alert_code == AlertCode.DRIVER_LIFECYCLE
    assert configuring_alert.details["to_state"] == "configuring"
    assert active_lifecycle_alert.alert_code == AlertCode.DRIVER_LIFECYCLE
    assert active_lifecycle_alert.details["to_state"] == "active"
    assert active_alert.alert_code == AlertCode.DRIVER_ACTIVE
    resolve_alert.assert_any_call("_MinimalDriver", AlertCode.DRIVER_LIFECYCLE)


def test_transition_to_creates_twin_alert_when_twin_bound() -> None:
    twin = _twin_with_alerts()
    driver = _MinimalDriver(twin=twin)

    # CONFIGURING/CONNECTING/INACTIVE no longer emit twin notices (trimmed);
    # ACTIVE still does.
    driver._transition_to(DriverLifecycleState.ACTIVE)

    twin.alerts.create.assert_called_once()
    kwargs = driver._twin.alerts.create.call_args.kwargs
    assert kwargs["name"] == "Driver active"
    assert kwargs["alert_type"] == "driver_lifecycle"
    assert driver._lifecycle_twin_pending_notice is False


def test_sync_lifecycle_alerts_after_connect_pushes_pending_twin_notice() -> None:
    twin = _twin_with_alerts()
    driver = _MinimalDriver(twin=twin)
    # ACTIVE is one of the states that still emits a twin notice after the trim.
    driver._lifecycle_state = DriverLifecycleState.ACTIVE
    driver._lifecycle_twin_pending_notice = True

    with patch.object(
        driver._alert_manager, "sync_lifecycle_alerts_to_backend"
    ) as sync_backend:
        driver._sync_lifecycle_alerts_after_connect()

    sync_backend.assert_called_once_with("_MinimalDriver")
    twin.alerts.create.assert_called_once()


def test_enable_backend_integration_pushes_lifecycle_info() -> None:
    from cyberwave.driver.cloud.alerts import (
        AlertManager,
        create_driver_lifecycle_alert,
    )

    manager = AlertManager()
    manager.raise_alert(
        create_driver_lifecycle_alert("TestDriver", from_state="a", to_state="b")
    )

    with patch.object(manager, "_push_alert_to_backend") as push:
        manager.enable_backend_integration(
            sdk_client=MagicMock(),
            twin_uuid="twin-1",
        )

    push.assert_called_once()
