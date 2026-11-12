"""Navigation must subscribe before dispatching a fast physical waypoint."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.motion import TwinNavigationHandle


def test_follow_path_connects_mqtt_before_subscribing_and_dispatching() -> None:
    calls = []

    class Mqtt:
        connected = False
        topic_prefix = "staging"

        def connect(self) -> None:
            calls.append("connect")
            self.connected = True

        def subscribe(self, topic, handler) -> None:
            assert self.connected
            assert topic == "stagingcyberwave/twin/twin-uuid/navigate/status"
            assert callable(handler)
            calls.append("subscribe")

    api_client = MagicMock()
    api_client.param_serialize.return_value = ()
    response = MagicMock()
    response.data = b"{}"
    def dispatch(*args, **kwargs):
        calls.append("dispatch")
        return response

    api_client.call_api.side_effect = dispatch
    client = SimpleNamespace(
        config=SimpleNamespace(source_type="edge"),
        api=SimpleNamespace(api_client=api_client),
        mqtt=Mqtt(),
    )
    twin = SimpleNamespace(uuid="twin-uuid", client=client)

    TwinNavigationHandle(twin).follow_path([[0, 0, 0], [0.1, 0, 0]])

    assert calls == ["connect", "subscribe", "dispatch"]

    for command in ("stop", "pause", "resume"):
        calls.clear()
        getattr(TwinNavigationHandle(twin), command)()
        assert calls == ["subscribe", "dispatch"], command
