"""controller_policy_snapshot() returns {} without a twin, dict otherwise."""
from types import SimpleNamespace

from cyberwave.driver.base import BaseDriver


class _Probe(BaseDriver):
    REGISTRY_ID = "test/probe"
    async def on_configure(self): ...
    async def on_connect_to_device(self): ...
    async def on_register_callbacks(self): ...
    async def on_activate(self): ...
    async def on_shutdown(self): ...
    @classmethod
    def create(cls): return cls()


def test_snapshot_empty_without_twin():
    d = _Probe(twin=None, client=None, auto_register_interface=False)
    assert d.controller_policy_snapshot() == {}


def test_snapshot_keys_present_with_twin(monkeypatch):
    d = _Probe(twin=None, client=None, auto_register_interface=False)
    d._twin = SimpleNamespace(uuid="t-1")
    monkeypatch.setattr(
        "cyberwave.driver.base.resolve_twin_attached_controller",
        lambda twin: ("policy-9", "keyboard"),
    )
    assert d.controller_policy_snapshot() == {
        "controller_policy_uuid": "policy-9",
        "controller_type": "keyboard",
    }
