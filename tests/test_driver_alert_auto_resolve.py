"""Auto-resolving twin alerts + trimmed lifecycle twin notices."""

from __future__ import annotations

from unittest.mock import MagicMock

from cyberwave.driver.cloud import alert_api
from cyberwave.driver.cloud.alert_api import DriverAlertsMixin
from cyberwave.driver.cloud.alerts import (
    AlertCode,
    AlertManager,
    AlertSeverity,
    DriverAlert,
    _format_alert_description,
)
from cyberwave.driver.cloud.lifecycle_alerts import (
    _LIFECYCLE_TWIN_ALERTS,
    _LIFECYCLE_TWIN_AUTO_RESOLVE_S,
    LifecycleAlertsMixin,
)
from cyberwave.driver.status import DriverLifecycleState


class _ImmediateTimer:
    """Stand-in for threading.Timer that runs the callback on start()."""

    def __init__(self, delay: float, fn) -> None:
        self._fn = fn
        self.daemon = False

    def start(self) -> None:
        self._fn()


class _Host(DriverAlertsMixin):
    pass


def _host() -> _Host:
    return _Host()


# --- _schedule_alert_auto_resolve --------------------------------------------


def test_auto_resolve_resolves_info_alert(monkeypatch) -> None:
    monkeypatch.setattr(alert_api.threading, "Timer", _ImmediateTimer)
    created = MagicMock(uuid="a1")
    _host()._schedule_alert_auto_resolve(created, "info", 5.0)
    created.resolve.assert_called_once()


def test_auto_resolve_resolves_warning_alert(monkeypatch) -> None:
    monkeypatch.setattr(alert_api.threading, "Timer", _ImmediateTimer)
    created = MagicMock(uuid="a1")
    _host()._schedule_alert_auto_resolve(created, "warning", 5.0)
    created.resolve.assert_called_once()


def test_auto_resolve_skips_error_and_critical(monkeypatch) -> None:
    monkeypatch.setattr(alert_api.threading, "Timer", _ImmediateTimer)
    for severity in ("error", "critical"):
        created = MagicMock(uuid="a1")
        _host()._schedule_alert_auto_resolve(created, severity, 5.0)
        created.resolve.assert_not_called()


def test_auto_resolve_noop_without_delay() -> None:
    created = MagicMock(uuid="a1")
    _host()._schedule_alert_auto_resolve(created, "info", None)
    created.resolve.assert_not_called()


def test_create_twin_alert_schedules_auto_resolve(monkeypatch) -> None:
    monkeypatch.setattr(alert_api.threading, "Timer", _ImmediateTimer)
    created = MagicMock(uuid="a1")
    alerts_api = MagicMock()
    alerts_api.create.return_value = created
    host = _host()
    host._twin = MagicMock(alerts=alerts_api)

    out = host.create_twin_alert(
        "n", severity="info", auto_resolve_after=5.0, _async_dispatch=False
    )
    assert out is created
    created.resolve.assert_called_once()


def test_create_twin_alert_error_not_auto_resolved(monkeypatch) -> None:
    monkeypatch.setattr(alert_api.threading, "Timer", _ImmediateTimer)
    created = MagicMock(uuid="a1")
    alerts_api = MagicMock()
    alerts_api.create.return_value = created
    host = _host()
    host._twin = MagicMock(alerts=alerts_api)

    host.create_twin_alert(
        "n", severity="error", auto_resolve_after=5.0, _async_dispatch=False
    )
    created.resolve.assert_not_called()


# --- trimmed lifecycle twin notices ------------------------------------------


class _LHost(LifecycleAlertsMixin):
    registry_id = "vendor/x"

    def _lifecycle_alert_component(self) -> str:
        return "X"


def _lhost() -> _LHost:
    host = _LHost()
    host._twin = object()
    host._lifecycle_twin_pending_notice = False
    host.create_twin_alert = MagicMock()  # type: ignore[method-assign]
    return host


def test_lifecycle_twin_notices_trimmed_to_meaningful_states() -> None:
    assert set(_LIFECYCLE_TWIN_ALERTS) == {
        DriverLifecycleState.ACTIVE,
        DriverLifecycleState.RECONNECTING,
        DriverLifecycleState.FINALIZED,
        DriverLifecycleState.ERROR,
    }


def test_lifecycle_transient_states_emit_no_twin_notice() -> None:
    host = _lhost()
    for state in (
        DriverLifecycleState.CONFIGURING,
        DriverLifecycleState.CONNECTING,
        DriverLifecycleState.INACTIVE,
        DriverLifecycleState.DEACTIVATING,
    ):
        host._notify_lifecycle_twin_alert(DriverLifecycleState.UNCONFIGURED, state)
    host.create_twin_alert.assert_not_called()


def test_lifecycle_active_notice_auto_resolves() -> None:
    host = _lhost()
    host._notify_lifecycle_twin_alert(
        DriverLifecycleState.INACTIVE, DriverLifecycleState.ACTIVE
    )
    _, kwargs = host.create_twin_alert.call_args
    assert kwargs["auto_resolve_after"] == _LIFECYCLE_TWIN_AUTO_RESOLVE_S


def test_lifecycle_error_notice_persists() -> None:
    host = _lhost()
    host._notify_lifecycle_twin_alert(
        DriverLifecycleState.ACTIVE, DriverLifecycleState.ERROR
    )
    _, kwargs = host.create_twin_alert.call_args
    assert kwargs["auto_resolve_after"] is None


# --- AlertManager backend push: suppress lifecycle/active, clean description ---


def _manager_with_twin():
    mgr = AlertManager()
    mgr._sdk_client = object()
    mgr._twin_uuid = "twin-1"
    mgr._enable_backend_push = True
    twin = MagicMock()
    mgr._get_twin = MagicMock(return_value=twin)  # type: ignore[method-assign]
    return mgr, twin


def test_push_skips_lifecycle_and_active_codes() -> None:
    for code in (AlertCode.DRIVER_LIFECYCLE, AlertCode.DRIVER_ACTIVE):
        mgr, twin = _manager_with_twin()
        alert = DriverAlert(
            alert_code=code,
            severity=AlertSeverity.INFO,
            component="PiperDriver",
            message="x",
            details={"from_state": "inactive", "to_state": "active"},
        )
        mgr._push_alert_to_backend(alert, f"PiperDriver_{code.name}")
        twin.alerts.create.assert_not_called()


def test_push_creates_backend_alert_for_other_codes() -> None:
    mgr, twin = _manager_with_twin()
    alert = DriverAlert(
        alert_code=AlertCode.BATTERY_WARNING,
        severity=AlertSeverity.WARNING,
        component="PiperDriver",
        message="Battery low",
        details={"percentage": 12},
    )
    mgr._push_alert_to_backend(alert, "k")
    twin.alerts.create.assert_called_once()
    kwargs = twin.alerts.create.call_args.kwargs
    assert kwargs["name"] == "Battery low"
    # Clean, readable description — no raw JSON blob.
    assert kwargs["description"] == "Component: PiperDriver\npercentage: 12"


def test_format_alert_description_is_readable() -> None:
    alert = DriverAlert(
        alert_code=AlertCode.DRIVER_ACTIVE,
        severity=AlertSeverity.INFO,
        component="PiperDriver",
        message="Driver active",
        details={"registry_id": "agilex/piper"},
    )
    desc = _format_alert_description(alert)
    assert desc == "Component: PiperDriver\nregistry_id: agilex/piper"
    assert "{" not in desc and "}" not in desc
