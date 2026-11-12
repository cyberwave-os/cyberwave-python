"""Tests for MQTT auth credential selection and validation."""

from unittest.mock import patch

import pytest

from cyberwave.config import CyberwaveConfig, DEFAULT_MQTT_USERNAME
from cyberwave.mqtt import CyberwaveMQTTClient as BaseMQTTClient
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient
from cyberwave.mqtt_identity import mqtt_username_for_token


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


def test_wrapper_client_prefers_config_mqtt_password_over_api_key(clean_api_key_env):
    config = CyberwaveConfig(
        api_key="api_key_secret",
        mqtt_username="user",
        mqtt_password="env_mqtt_secret",
    )

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        WrapperMQTTClient(config=config)

    mqtt_client_cls.return_value.username_pw_set.assert_called_once_with(
        username="user",
        password="env_mqtt_secret",
    )


def test_wrapper_client_requires_api_key_or_mqtt_password(clean_api_key_env):
    config = CyberwaveConfig(api_key=None, token=None)

    with pytest.raises(ValueError, match="API key or mqtt_password is required"):
        WrapperMQTTClient(config=config)


def test_config_defaults_to_tls_mqtt_port(clean_api_key_env):
    config = CyberwaveConfig(api_key="api_key_secret")

    assert config.mqtt_port == 8883
    assert config.mqtt_use_tls is True


def test_the_placeholder_username_is_replaced_by_one_derived_from_the_token(
    clean_api_key_env,
):
    """Authorization checks reach the backend without the password, so a
    username every client shares leaves it unable to tell them apart. Derived
    from the token it names exactly one (CYB-4031)."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            api_key="api_key_secret",
            auto_connect=False,
        )

    username = mqtt_client_cls.return_value.username_pw_set.call_args.kwargs["username"]
    assert username == mqtt_username_for_token("api_key_secret")
    assert "api_key_secret" not in username


def test_an_explicit_username_is_left_alone(clean_api_key_env):
    """A caller that named an account meant it -- the ROS 2 drivers pass the
    Django username, and legacy broker credentials are an account too."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="ada",
            api_key="api_key_secret",
            auto_connect=False,
        )

    assert (
        mqtt_client_cls.return_value.username_pw_set.call_args.kwargs["username"]
        == "ada"
    )


def test_legacy_broker_credentials_keep_the_placeholder(clean_api_key_env):
    """An explicit mqtt_password is the legacy static path, where "mqttcyb" is
    the broker account and hashing the password would deny the CONNECT."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_password="mqttcyb231",
            auto_connect=False,
        )

    assert (
        mqtt_client_cls.return_value.username_pw_set.call_args.kwargs["username"]
        == DEFAULT_MQTT_USERNAME
    )


def test_a_config_built_client_derives_the_username_from_the_api_key(
    clean_api_key_env,
):
    """The wrapper is the path edge-core and the high-level SDK take. It used to
    forward the API key as mqtt_password, which the base client reads as legacy
    broker credentials -- so the placeholder survived and the ACL callback, which
    gets no password, could not tell one client from another."""
    config = CyberwaveConfig(api_key="api_key_secret", mqtt_host="localhost")

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        WrapperMQTTClient(config)

    call = mqtt_client_cls.return_value.username_pw_set.call_args.kwargs
    assert call["username"] == mqtt_username_for_token("api_key_secret")
    assert call["password"] == "api_key_secret"


def test_a_config_built_client_keeps_the_placeholder_for_legacy_credentials(
    clean_api_key_env,
):
    """An explicit password still means the legacy static account, where the
    username is the account and must reach the broker unchanged."""
    config = CyberwaveConfig(
        api_key="api_key_secret",
        mqtt_host="localhost",
        mqtt_password="mqttcyb231",
    )

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        WrapperMQTTClient(config)

    call = mqtt_client_cls.return_value.username_pw_set.call_args.kwargs
    assert call["username"] == DEFAULT_MQTT_USERNAME
    assert call["password"] == "mqttcyb231"


def test_an_api_token_under_mqtt_password_still_derives_the_username(
    clean_api_key_env,
):
    """The cloud node scopes a workload token by setting CYBERWAVE_API_KEY and
    CYBERWAVE_MQTT_PASSWORD to it, and edge-core passes every CYBERWAVE_* var
    into the worker container. Read as legacy credentials that left the worker
    on the shared placeholder, which the ACL callback cannot resolve a principal
    from -- the subscribe came back SUBACK 128 and the worker sat idle."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_password="cw_" + "a" * 64,
            auto_connect=False,
        )

    call = mqtt_client_cls.return_value.username_pw_set.call_args.kwargs
    assert call["username"] == mqtt_username_for_token("cw_" + "a" * 64)
    assert call["password"] == "cw_" + "a" * 64


def test_a_config_built_client_derives_from_an_api_token_mqtt_password(
    clean_api_key_env,
):
    """The worker's own path: both names carry the same workload token."""
    token = "cw_" + "b" * 64
    config = CyberwaveConfig(
        api_key=token,
        mqtt_host="localhost",
        mqtt_password=token,
    )

    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        WrapperMQTTClient(config)

    call = mqtt_client_cls.return_value.username_pw_set.call_args.kwargs
    assert call["username"] == mqtt_username_for_token(token)
    assert call["username"] != DEFAULT_MQTT_USERNAME


def test_config_resolves_raw_paho_credentials_like_the_client(
    clean_api_key_env, monkeypatch
):
    """Edge drivers on a raw paho client connect exactly as the SDK client would."""
    for name in ("CYBERWAVE_MQTT_USERNAME", "CYBERWAVE_MQTT_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    token = "cw_" + "c" * 64
    assert CyberwaveConfig(api_key=token).mqtt_credentials == (
        mqtt_username_for_token(token),
        token,
    )
    assert CyberwaveConfig(
        mqtt_username="edge-user", mqtt_password="edge-pass"
    ).mqtt_credentials == ("edge-user", "edge-pass")
    assert CyberwaveConfig(mqtt_password="static").mqtt_credentials == (
        DEFAULT_MQTT_USERNAME,
        "static",
    )
    assert CyberwaveConfig().mqtt_credentials == (DEFAULT_MQTT_USERNAME, None)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, ""),
        ({"CYBERWAVE_ENVIRONMENT": "production"}, ""),
        ({"CYBERWAVE_ENVIRONMENT": "Production"}, ""),
        ({"CYBERWAVE_ENVIRONMENT": "dev"}, "dev"),
        ({"CYBERWAVE_ENVIRONMENT": " staging "}, "staging"),
        # Simulation nodes export CYBERWAVE_TOPIC_PREFIX; Edge Core forwards it.
        ({"CYBERWAVE_TOPIC_PREFIX": "local", "CYBERWAVE_ENVIRONMENT": "dev"}, "local"),
        (
            {
                "CYBERWAVE_MQTT_TOPIC_PREFIX": "explicit",
                "CYBERWAVE_TOPIC_PREFIX": "local",
                "CYBERWAVE_ENVIRONMENT": "dev",
            },
            "explicit",
        ),
    ],
)
def test_config_topic_prefix_resolution(monkeypatch, environment, expected):
    for name in (
        "CYBERWAVE_MQTT_TOPIC_PREFIX",
        "CYBERWAVE_TOPIC_PREFIX",
        "CYBERWAVE_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert CyberwaveConfig().topic_prefix == expected


@pytest.mark.parametrize(
    ("environment", "port", "tls"),
    [
        ({}, 8883, True),
        ({"CYBERWAVE_MQTT_PORT": "1883"}, 1883, False),
        ({"CYBERWAVE_MQTT_PORT": "8883"}, 8883, True),
        ({"CYBERWAVE_MQTT_PORT": "1883", "CYBERWAVE_MQTT_USE_TLS": "true"}, 1883, True),
    ],
)
def test_config_port_and_tls_resolution(monkeypatch, environment, port, tls):
    for name in ("CYBERWAVE_MQTT_PORT", "CYBERWAVE_MQTT_USE_TLS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    config = CyberwaveConfig()
    assert (config.mqtt_port, config.mqtt_use_tls) == (port, tls)


@pytest.mark.parametrize(
    ("host", "environment", "warns"),
    [
        ("staging.mqtt.cyberwave.com", None, True),
        ("staging.mqtt.cyberwave.com", "dev", True),
        ("staging.mqtt.cyberwave.com", "staging", False),
        ("dev.mqtt.cyberwave.com", "dev", False),
        ("mqtt.cyberwave.com", None, False),
        ("mqtt.cyberwave.com", "production", False),
        ("mqtt.cyberwave.com", "staging", True),
        ("localhost", "local", False),
        ("custom.example.com", None, False),
    ],
)
def test_topic_prefix_broker_mismatch_warns(
    monkeypatch, caplog, host, environment, warns
):
    for name in (
        "CYBERWAVE_MQTT_TOPIC_PREFIX",
        "CYBERWAVE_TOPIC_PREFIX",
        "CYBERWAVE_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    if environment is not None:
        monkeypatch.setenv("CYBERWAVE_ENVIRONMENT", environment)
    with caplog.at_level("WARNING", logger="cyberwave.config"):
        config = CyberwaveConfig(mqtt_host=host)
    # The prefix is never inferred from the broker host.
    expected_prefix = "" if environment in (None, "production") else environment
    assert config.topic_prefix == expected_prefix
    assert ("does not match broker" in caplog.text) is warns
