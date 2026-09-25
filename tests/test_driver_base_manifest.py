"""BaseDriver manifest export from interface registry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.driver import CallbackGroup, CommandArgs, BaseDriver, TopicSpec


_TEST_TWIN = SimpleNamespace(
    uuid="00000000-0000-0000-0000-000000000001",
    environment_id="env-00000000-0000-0000-0000-000000000099",
    name="manifest-test",
    client=None,
)


class _ManifestDriver(BaseDriver):
    REGISTRY_ID = "acme/test-driver"
    driver_family = "python"

    def define_interface(self, iface) -> None:
        cmd = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
        )
        iface.add_listener(
            cmd,
            CallbackGroup(callback=lambda _e: None),
            command=CommandArgs(name="ping"),
        )

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
    def create(cls) -> _ManifestDriver:
        return cls(_TEST_TWIN)


def test_get_driver_manifest_exports_cw_driver_root() -> None:
    driver = _ManifestDriver(_TEST_TWIN)
    root = driver.get_driver_manifest(compiled=False)
    assert root["mqtt"]["twin"]["command"]["payload_schema_ref"] == "TwinCommandPayload"
    assert "ping" in root["mqtt"]["commands"]["supported"]


def test_get_driver_manifest_compiled_raises() -> None:
    driver = _ManifestDriver(_TEST_TWIN)
    with pytest.raises(ValueError, match="backend"):
        driver.get_driver_manifest(compiled=True)
