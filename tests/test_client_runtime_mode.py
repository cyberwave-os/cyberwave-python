from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cyberwave import Cyberwave
from cyberwave.constants import SOURCE_TYPE_EDGE, SOURCE_TYPE_SIM
from cyberwave.twin import LocomoteTwin


def test_client_defaults_to_live_mode() -> None:
    client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")

    assert client.config.runtime_mode == "live"
    assert client.config.source_type == SOURCE_TYPE_EDGE


def test_client_simulation_mode_uses_sim_defaults() -> None:
    client = Cyberwave(
        base_url="http://localhost:8000",
        api_key="test_key",
        mode="simulation",
    )

    assert client.config.runtime_mode == "simulation"
    assert client.config.source_type == SOURCE_TYPE_SIM


def test_client_explicit_source_type_overrides_mode_default() -> None:
    client = Cyberwave(
        base_url="http://localhost:8000",
        api_key="test_key",
        mode="simulation",
        source_type=SOURCE_TYPE_EDGE,
    )

    assert client.config.runtime_mode == "simulation"
    assert client.config.source_type == SOURCE_TYPE_EDGE


def test_client_respects_env_source_type_when_arg_omitted() -> None:
    with patch.dict("os.environ", {"CYBERWAVE_SOURCE_TYPE": SOURCE_TYPE_EDGE}):
        client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")

    assert client.config.runtime_mode == "live"
    assert client.config.source_type == SOURCE_TYPE_EDGE


def test_mqtt_client_uses_runtime_mode_client_id_prefix() -> None:
    with patch("cyberwave.mqtt.mqtt.Client"):
        live_client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")
        simulation_client = Cyberwave(
            base_url="http://localhost:8000",
            api_key="test_key",
            mode="simulation",
        )

        assert live_client.mqtt._client.client_id.startswith("sdk_")
        assert not live_client.mqtt._client.client_id.startswith("sdk_sim_")
        assert simulation_client.mqtt._client.client_id.startswith("sdk_sim_")


def test_affect_updates_state_source_type_to_match_runtime() -> None:
    client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")

    client.affect("simulation")
    assert client.config.runtime_mode == "simulation"
    assert client.config.source_type == SOURCE_TYPE_SIM

    client.affect("live")
    assert client.config.runtime_mode == "live"
    assert client.config.source_type == SOURCE_TYPE_EDGE


def test_affect_changes_emitted_command_and_state_source_types() -> None:
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")
        twin = LocomoteTwin(client, SimpleNamespace(uuid="twin-uuid", name="Twin"))

        client.affect("simulation")
        simulation_mqtt = client.mqtt
        simulation_mqtt._client.connected = True
        simulation_mqtt._client.publish = MagicMock()

        twin.move_forward(1.0)
        simulation_command_payload = simulation_mqtt._client.publish.call_args.args[1]
        assert simulation_command_payload["source_type"] == "sim_tele"

        simulation_mqtt._client.publish.reset_mock()
        simulation_mqtt.update_twin_position(
            "twin-uuid", {"x": 1.0, "y": 2.0, "z": 3.0}
        )
        simulation_state_payload = simulation_mqtt._client.publish.call_args.args[1]
        assert simulation_state_payload["source_type"] == "sim"

        client.affect("live")
        live_mqtt = client.mqtt
        live_mqtt._client.connected = True
        live_mqtt._client.publish = MagicMock()

        twin.move_forward(1.0)
        live_command_payload = live_mqtt._client.publish.call_args.args[1]
        assert live_command_payload["source_type"] == "tele"

        live_mqtt._client.publish.reset_mock()
        live_mqtt.update_twin_position("twin-uuid", {"x": 4.0, "y": 5.0, "z": 6.0})
        live_state_payload = live_mqtt._client.publish.call_args.args[1]
        assert live_state_payload["source_type"] == "edge"
