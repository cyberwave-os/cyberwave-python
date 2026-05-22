"""End-to-end integration test for the worker pipeline."""

import builtins
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.runtime import WorkerRuntime


class MockBackend:
    """Minimal backend stub that records raw-key subscriptions."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, list] = {}

    def subscribe(self, key: str, callback, **_kwargs):
        self._subscriptions.setdefault(key, []).append(callback)
        return MagicMock()

    def push(self, key: str, sample) -> None:
        for cb in self._subscriptions.get(key, []):
            cb(sample)


class MockDataBus:
    """In-memory data bus facade that exposes a MockBackend."""

    KEY_PREFIX = "cw"

    def __init__(self) -> None:
        self._backend = MockBackend()

    @property
    def backend(self) -> MockBackend:
        return self._backend

    @property
    def key_prefix(self) -> str:
        return self.KEY_PREFIX

    def push(self, key: str, sample) -> None:
        self._backend.push(key, sample)


TEST_TWIN_UUID = "00000000-0000-0000-0000-000000000002"


class FakeCw:
    def __init__(self, *, data_bus=None):
        self._hook_registry = HookRegistry()
        self.config = type("Config", (), {"twin_uuid": TEST_TWIN_UUID})()
        self._data_bus = data_bus
        self.mqtt = MagicMock()
        self.mqtt.topic_prefix = ""

    @property
    def data(self):
        if self._data_bus is None:
            raise Exception("Data backend not available")
        return self._data_bus

    def on_frame(self, *a, **kw):
        return self._hook_registry.on_frame(*a, **kw)

    def publish_event(self, twin_uuid, event_type, data, *, source="edge_node"):
        import time

        self.mqtt.publish(
            f"cyberwave/twin/{twin_uuid}/event",
            {
                "event_type": event_type,
                "source": source,
                "data": data,
                "timestamp": time.time(),
            },
        )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if hasattr(builtins, "cw"):
        delattr(builtins, "cw")
    for key in list(sys.modules):
        if key.startswith("cyberwave_worker_"):
            del sys.modules[key]


def test_full_pipeline(tmp_path):
    """register hook -> load worker -> start runtime -> push sample -> callback fires -> event emitted."""
    data_bus = MockDataBus()
    fake_cw = FakeCw(data_bus=data_bus)

    call_log = []

    (tmp_path / "detect.py").write_text(
        "call_log = None\n"
        "\n"
        "@cw.on_frame(cw.config.twin_uuid)\n"
        "def on_frame(sample, ctx):\n"
        "    import sys\n"
        "    mod = sys.modules[__name__]\n"
        "    if mod.call_log is not None:\n"
        "        mod.call_log.append((sample, ctx))\n"
        "    cw.publish_event(ctx.twin_uuid, 'detected', {'raw': sample})\n"
    )

    runtime = WorkerRuntime(fake_cw)
    runtime.load(str(tmp_path))

    # Inject call_log into the loaded module for assertion
    worker_mod = sys.modules["cyberwave_worker_detect"]
    worker_mod.call_log = call_log

    runtime.start()

    # With no explicit sensor=, @cw.on_frame(twin) now subscribes to a
    # wildcard so the hook matches whatever sensor name the driver
    # publishes under (``color_camera``, ``depth_camera``, …).
    expected_key = f"cw/{TEST_TWIN_UUID}/data/frames/**"
    assert expected_key in data_bus.backend._subscriptions
    assert len(data_bus.backend._subscriptions[expected_key]) == 1

    # Push a sample
    mock_sample = MagicMock()
    mock_sample.payload = b"frame-bytes"
    mock_sample.timestamp = 99999.0
    mock_sample.metadata = {}

    data_bus.push(expected_key, mock_sample)

    # Wait for the dispatch thread to process the sample
    deadline = time.monotonic() + 2.0
    while not call_log and time.monotonic() < deadline:
        time.sleep(0.01)

    # Verify the hook callback was invoked
    assert len(call_log) == 1
    payload, ctx = call_log[0]
    assert payload == b"frame-bytes"
    assert ctx.twin_uuid == TEST_TWIN_UUID
    assert ctx.timestamp == 99999.0

    # Verify the event was published
    fake_cw.mqtt.publish.assert_called_once()
    topic = fake_cw.mqtt.publish.call_args[0][0]
    event_payload = fake_cw.mqtt.publish.call_args[0][1]
    assert topic == f"cyberwave/twin/{TEST_TWIN_UUID}/event"
    assert event_payload["event_type"] == "detected"
    assert event_payload["data"] == {"raw": b"frame-bytes"}

    runtime.stop()


def test_multiple_workers_multiple_hooks(tmp_path):
    """Multiple worker modules registering different hooks."""
    data_bus = MockDataBus()
    fake_cw = FakeCw(data_bus=data_bus)

    (tmp_path / "worker_a.py").write_text(
        "@cw.on_frame(cw.config.twin_uuid)\ndef on_frame(s, c):\n    pass\n"
    )
    (tmp_path / "worker_b.py").write_text(
        "reg = cw._hook_registry\n"
        "@reg.on_depth(cw.config.twin_uuid)\n"
        "def on_depth(s, c):\n"
        "    pass\n"
    )

    runtime = WorkerRuntime(fake_cw)
    count = runtime.load(str(tmp_path))
    assert count == 2

    runtime.start()

    # Wildcard subscription matches whatever sensor name the driver uses.
    frame_key = f"cw/{TEST_TWIN_UUID}/data/frames/**"
    depth_key = f"cw/{TEST_TWIN_UUID}/data/depth/**"
    assert frame_key in data_bus.backend._subscriptions
    assert depth_key in data_bus.backend._subscriptions

    runtime.stop()


def test_real_client_property_hook_delegation():
    """The real Cyberwave client's @property hook delegates register correctly."""
    from cyberwave.client import Cyberwave

    client = object.__new__(Cyberwave)
    client._hook_registry = HookRegistry()

    @client.on_frame("test-uuid", sensor="front")
    def handle_frame(sample, ctx):
        pass

    @client.on_depth("test-uuid")
    def handle_depth(sample, ctx):
        pass

    @client.on_data("test-uuid", "custom_ch")
    def handle_data(sample, ctx):
        pass

    hooks = client._hook_registry.hooks
    assert len(hooks) == 3
    assert hooks[0].hook_type == "frame"
    assert hooks[0].channel == "frames/front"
    assert hooks[0].callback is handle_frame
    assert hooks[1].hook_type == "depth"
    assert hooks[1].callback is handle_depth
    assert hooks[2].hook_type == "data"
    assert hooks[2].channel == "custom_ch"
    assert hooks[2].callback is handle_data
