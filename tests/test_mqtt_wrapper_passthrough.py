"""Regression: the CyberwaveMQTTClient *wrapper* must forward ``as_targets``.

``Cyberwave.mqtt`` returns the compatibility wrapper in ``cyberwave.mqtt_client``
(not the base client in ``cyberwave.mqtt``). Controllers publish commanded joint
setpoints via ``mqtt.update_joints_state(..., as_targets=True)`` so the payload
carries ``target_*`` field names. If the wrapper omits ``as_targets`` from its
signature/forwarding, that call raises ``TypeError`` on every control step (the
robot never moves) — or silently publishes commands as measured state. These
tests pin the passthrough so it cannot regress.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

from cyberwave.config import CyberwaveConfig
from cyberwave.mqtt import _UNSET
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient


def test_wrapper_signature_accepts_as_targets() -> None:
    params = inspect.signature(WrapperMQTTClient.update_joints_state).parameters
    assert "as_targets" in params, (
        "wrapper update_joints_state must accept as_targets so controller "
        "target-field publishes don't raise TypeError"
    )


def _make_wrapper_with_mock_base():
    config = CyberwaveConfig(api_key="api_key_secret", mqtt_username="user")
    with patch("cyberwave.mqtt_client.BaseMQTTClient") as base_cls:
        wrapper = WrapperMQTTClient(config=config)
    return wrapper, base_cls.return_value


def test_wrapper_forwards_as_targets_true_to_base() -> None:
    wrapper, base = _make_wrapper_with_mock_base()

    wrapper.update_joints_state(
        twin_uuid="twin-uuid",
        joint_positions={"_1": 0.5, "_2": -0.3},
        source_type="sim_tele",
        as_targets=True,
    )

    base.update_joints_state.assert_called_once()
    # The wrapper forwards positionally with as_targets as the final argument.
    assert base.update_joints_state.call_args.args[-1] is True


def test_wrapper_defaults_as_targets_false() -> None:
    wrapper, base = _make_wrapper_with_mock_base()

    wrapper.update_joints_state(
        twin_uuid="twin-uuid",
        joint_positions={"_1": 0.5},
        source_type="sim",
    )

    # Measured publishes (the default) must still forward as_targets=False so
    # plant/edge state is never relabelled as a command.
    assert base.update_joints_state.call_args.args[-1] is False


def test_wrapper_signature_accepts_subscriber_key() -> None:
    # WebRTC streamers call self.client.subscribe(..., subscriber_key=...) on
    # the *wrapper*; if it omits the kwarg, _subscribe_to_answer() raises
    # TypeError and the camera/mic stream never starts.
    subscribe_params = inspect.signature(WrapperMQTTClient.subscribe).parameters
    assert "subscriber_key" in subscribe_params, (
        "wrapper subscribe must accept subscriber_key so streamer answer "
        "subscriptions don't raise TypeError"
    )
    unsubscribe_params = inspect.signature(WrapperMQTTClient.unsubscribe).parameters
    assert "subscriber_key" in unsubscribe_params, (
        "wrapper unsubscribe must accept subscriber_key to remove a single "
        "subscriber without tearing down a shared topic"
    )


def test_wrapper_forwards_subscriber_key_on_subscribe() -> None:
    wrapper, base = _make_wrapper_with_mock_base()

    def handler(_data):  # pragma: no cover - never invoked
        pass

    wrapper.subscribe("twin/topic", handler, subscriber_key="video:cam:live:default")

    base.subscribe.assert_called_once()
    assert (
        base.subscribe.call_args.kwargs["subscriber_key"] == "video:cam:live:default"
    )


def test_wrapper_defaults_subscriber_key_none_on_subscribe() -> None:
    wrapper, base = _make_wrapper_with_mock_base()

    wrapper.subscribe("twin/topic", lambda _d: None)

    # The default single-slot key preserves replace semantics for callers that
    # don't opt into multi-subscriber coexistence.
    assert base.subscribe.call_args.kwargs["subscriber_key"] is None


def test_wrapper_forwards_subscriber_key_on_unsubscribe() -> None:
    wrapper, base = _make_wrapper_with_mock_base()

    wrapper.unsubscribe("twin/topic", "video:cam:live:default")
    assert base.unsubscribe.call_args.args[-1] == "video:cam:live:default"


def test_wrapper_unsubscribe_defaults_to_remove_all() -> None:
    wrapper, base = _make_wrapper_with_mock_base()

    wrapper.unsubscribe("twin/topic")
    # Omitting subscriber_key must forward the _UNSET sentinel so the base
    # client removes every handler and tears down the broker subscription.
    assert base.unsubscribe.call_args.args[-1] is _UNSET
