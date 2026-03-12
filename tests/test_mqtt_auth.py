"""Tests for MQTT auth credential selection and validation."""

from unittest.mock import patch

import pytest

from cyberwave.config import CyberwaveConfig
from cyberwave.mqtt import CyberwaveMQTTClient as BaseMQTTClient
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient


@pytest.fixture
def clean_api_key_env(monkeypatch):
    """Ensure tests don't inherit API key credentials from environment."""
    monkeypatch.delenv("CYBERWAVE_API_KEY", raising=False)


def test_base_client_uses_api_key_when_no_mqtt_password(clean_api_key_env):
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
        )

    mqtt_client_cls.return_value.username_pw_set.assert_called_once_with(
        username="user",
        password="api_key_secret",
    )


def test_base_client_accepts_mqtt_password_without_api_key(clean_api_key_env):
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            mqtt_password="mqtt_secret",
            auto_connect=False,
        )

    mqtt_client_cls.return_value.username_pw_set.assert_called_once_with(
        username="user",
        password="mqtt_secret",
    )


def test_base_client_prefers_mqtt_password_over_api_key(clean_api_key_env):
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            mqtt_password="explicit_mqtt_secret",
            auto_connect=False,
        )

    mqtt_client_cls.return_value.username_pw_set.assert_called_once_with(
        username="user",
        password="explicit_mqtt_secret",
    )


def test_base_client_requires_api_key_or_mqtt_password(clean_api_key_env):
    with pytest.raises(ValueError, match="api_key or mqtt_password is required"):
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            auto_connect=False,
        )


def test_wrapper_client_accepts_explicit_mqtt_password_without_api_key(clean_api_key_env):
    config = CyberwaveConfig(api_key=None, token=None, mqtt_username="user")

    with patch("cyberwave.mqtt_client.BaseMQTTClient") as base_client_cls:
        WrapperMQTTClient(config=config, mqtt_password="explicit_mqtt_secret")

    kwargs = base_client_cls.call_args.kwargs
    assert kwargs["api_key"] is None
    assert kwargs["mqtt_password"] == "explicit_mqtt_secret"


def test_wrapper_client_prefers_explicit_mqtt_password_over_api_key(clean_api_key_env):
    config = CyberwaveConfig(api_key="api_key_secret", mqtt_username="user")

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        WrapperMQTTClient(config=config, mqtt_password="explicit_mqtt_secret")

    mqtt_client_cls.return_value.username_pw_set.assert_called_once_with(
        username="user",
        password="explicit_mqtt_secret",
    )


def test_wrapper_client_requires_api_key_or_mqtt_password(clean_api_key_env):
    config = CyberwaveConfig(api_key=None, token=None)

    with pytest.raises(ValueError, match="API key or mqtt_password is required"):
        WrapperMQTTClient(config=config)


def test_config_defaults_to_tls_mqtt_port(clean_api_key_env):
    config = CyberwaveConfig(api_key="api_key_secret")

    assert config.mqtt_port == 8883
    assert config.mqtt_use_tls is True
