"""Base on_controller_assigned/removed home + emit a standard twin alert."""
import asyncio

from cyberwave.driver.interface.registry_mixin import InterfaceRegistryMixin


class _Probe(InterfaceRegistryMixin):
    def __init__(self, with_home: bool):
        self.homed = 0
        self.alerts = []
        if with_home:
            self.request_home = self._request_home  # type: ignore[assignment]

    def _request_home(self):
        self.homed += 1

    def create_twin_alert(self, name, **kwargs):
        self.alerts.append((name, kwargs.get("alert_type"), kwargs.get("severity")))

    def get_logger(self):  # registry_mixin logs via module logger, not this
        raise AssertionError("unused")


def test_assigned_homes_and_alerts():
    p = _Probe(with_home=True)
    asyncio.run(p.on_controller_assigned("keyboard", "policy-1"))
    assert p.homed == 1
    assert p.alerts and p.alerts[0][1] == "controller_assigned"


def test_removed_homes_and_alerts():
    p = _Probe(with_home=True)
    asyncio.run(p.on_controller_removed())
    assert p.homed == 1
    assert p.alerts and p.alerts[0][1] == "controller_removed"


def test_no_home_seam_is_safe():
    p = _Probe(with_home=False)
    asyncio.run(p.on_controller_assigned("keyboard", None))  # must not raise
    assert p.homed == 0
    assert p.alerts and p.alerts[0][1] == "controller_assigned"
