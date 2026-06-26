"""BaseDriver cloud connect uses Cyberwave() SDK defaults (config.py unchanged)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cyberwave.config as config_module
from cyberwave.config import CyberwaveConfig
from cyberwave.constants import SOURCE_TYPE_EDGE
from cyberwave.driver.base import BaseDriver


class _MinimalDriver(BaseDriver):
    REGISTRY_ID = "vendor/robot"

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
    def create(cls) -> _MinimalDriver:
        return cls(twin=SimpleNamespace(uuid="twin-uuid", environment_id="env-uuid"))


class _DriverWithoutAssetKey(BaseDriver):
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
    def create(cls) -> _DriverWithoutAssetKey:
        return cls()


def _twin_with_client() -> SimpleNamespace:
    client = MagicMock()
    client.mqtt.connected = True
    client.config = CyberwaveConfig(
        api_key="key",
        base_url="https://api.example.com",
        mqtt_host="mqtt.example.com",
    )
    return SimpleNamespace(
        uuid="twin-uuid",
        environment_id="env-uuid",
        name="test-twin",
        client=client,
        alerts=SimpleNamespace(create=MagicMock()),
    )


@pytest.fixture(autouse=True)
def _isolate_global_config() -> None:
    prev = config_module._global_config
    config_module._global_config = None
    yield
    config_module._global_config = prev


def test_registry_id_requires_subclass_constant() -> None:
    driver = _DriverWithoutAssetKey()
    with pytest.raises(RuntimeError, match="REGISTRY_ID"):
        _ = driver.registry_id


def test_registry_id_returns_class_constant() -> None:
    assert _MinimalDriver().registry_id == "vendor/robot"


def test_connect_mqtt_async_uses_cyberwave_with_edge_source_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYBERWAVE_API_KEY", "test-api-key")
    driver = _MinimalDriver()

    mock_client = MagicMock()
    mock_client.mqtt.connected = True
    mock_client.config = CyberwaveConfig(api_key="test-api-key")

    with patch("cyberwave.Cyberwave", return_value=mock_client) as mock_cw:
        result = asyncio.run(driver._connect_mqtt_async())

    mock_cw.assert_called_once_with(source_type=SOURCE_TYPE_EDGE)
    assert result is mock_client


def test_connect_mqtt_async_raises_system_exit_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYBERWAVE_API_KEY", "test-api-key")
    driver = _MinimalDriver()

    mock_client = MagicMock()
    mock_client.mqtt.connected = False

    with patch("cyberwave.Cyberwave", return_value=mock_client):
        with patch("asyncio.sleep", return_value=None):
            with patch.object(driver._alert_manager, "raise_alert") as raise_alert:
                with pytest.raises(SystemExit) as exc:
                    asyncio.run(driver._connect_mqtt_async())

    assert exc.value.code == 1
    raise_alert.assert_called_once()


def test_enable_backend_alerts_always_enabled() -> None:
    twin = _twin_with_client()
    driver = _MinimalDriver(twin=twin)
    driver._cw = twin.client
    driver._twin = twin

    with patch.object(
        driver._alert_manager, "enable_backend_integration"
    ) as enable_backend:
        with patch.object(driver._alert_manager, "start_alert_listener"):
            with patch.object(driver, "_sync_lifecycle_alerts_after_connect"):
                driver._enable_backend_alerts()

    enable_backend.assert_called_once()
    assert enable_backend.call_args.kwargs["sdk_client"] is twin.client


def test_connect_cloud_async_reuses_prebound_client_mqtt() -> None:
    twin = _twin_with_client()
    driver = _MinimalDriver(twin=twin)
    driver._cw = twin.client

    with patch.object(driver, "_connect_mqtt_async") as connect_mqtt:
        with patch.object(driver, "_fetch_twin") as fetch_twin:
            asyncio.run(driver._connect_cloud_async())

    connect_mqtt.assert_not_called()
    fetch_twin.assert_not_called()
