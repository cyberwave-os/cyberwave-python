"""Tests for WorkerRuntime lifecycle."""

import builtins
import sys
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

import cyberwave.workers.runtime as worker_runtime
from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.runtime import WorkerRuntime


TEST_TWIN_UUID = "00000000-0000-0000-0000-000000000001"


def _always_due_prev_fire(_cron, local_now):
    """Mock ``_previous_cron_fire`` so the runtime always sees a brand
    new cron tick on every poll after first sighting.

    The runtime seeds ``_schedule_last_fire[key] = local_now`` on the
    first poll and only fires on subsequent polls when
    ``prev_fire > last_fire``. Returning a value strictly greater than
    any ``local_now`` we'll ever see guarantees the second poll fires.
    """
    return local_now + timedelta(days=1)


class FakeCw:
    """Minimal stub of the Cyberwave client for runtime tests."""

    def __init__(self, *, data_bus=None):
        self._hook_registry = HookRegistry()
        self.config = type("Config", (), {"twin_uuid": TEST_TWIN_UUID})()
        self._data_bus = data_bus
        self.published_alerts: list[dict] = []
        self._publish_alert_lock = threading.Lock()

    def publish_alert(
        self,
        twin_uuid,
        name,
        *,
        description="",
        alert_type="",
        severity="info",
        category="business",
        force=False,
        metadata=None,
        workflow_uuid=None,
        workflow_node_uuid=None,
        workflow_execution_uuid=None,
    ):
        with self._publish_alert_lock:
            self.published_alerts.append(
                {
                    "twin_uuid": twin_uuid,
                    "name": name,
                    "description": description,
                    "alert_type": alert_type,
                    "severity": severity,
                    "category": category,
                }
            )

    @property
    def data(self):
        if self._data_bus is None:
            raise Exception("Data backend not available")
        return self._data_bus

    @property
    def on_schedule(self):
        return self._hook_registry.on_schedule


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if hasattr(builtins, "cw"):
        delattr(builtins, "cw")
    for key in list(sys.modules):
        if key.startswith("cyberwave_worker_"):
            del sys.modules[key]


def test_runtime_load_and_start(tmp_path):
    fake_cw = FakeCw()
    (tmp_path / "worker.py").write_text(
        "import builtins\n"
        f"reg = builtins.cw._hook_registry\n"
        f"reg.on_frame('{TEST_TWIN_UUID}')(lambda s, c: None)\n"
    )

    runtime = WorkerRuntime(fake_cw)
    count = runtime.load(str(tmp_path))
    assert count == 1
    assert len(fake_cw._hook_registry.hooks) == 1

    runtime.start()
    # No subscriptions because data bus is not available
    assert len(runtime._subscriptions) == 0


def test_runtime_stop_sets_event():
    fake_cw = FakeCw()
    runtime = WorkerRuntime(fake_cw)
    assert not runtime._stop_event.is_set()

    runtime.stop()
    assert runtime._stop_event.is_set()


def test_runtime_stop_cleans_up_builtin_cw():
    fake_cw = FakeCw()
    builtins.cw = fake_cw  # type: ignore[attr-defined]

    runtime = WorkerRuntime(fake_cw)
    runtime.stop()

    assert not hasattr(builtins, "cw")


def test_runtime_run_blocks_and_stop_unblocks():
    fake_cw = FakeCw()
    runtime = WorkerRuntime(fake_cw)

    stopped = threading.Event()

    def run_runtime():
        runtime.run()
        stopped.set()

    t = threading.Thread(target=run_runtime, daemon=True)
    t.start()

    # Give it a moment to start blocking
    assert not stopped.wait(timeout=0.1)

    runtime.stop()
    assert stopped.wait(timeout=2.0)


def test_runtime_with_mock_data_bus():
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    mock_sub = MagicMock()
    mock_bus.backend.subscribe.return_value = mock_sub

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(lambda s, c: None)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    mock_bus.backend.subscribe.assert_called_once()
    assert len(runtime._subscriptions) == 1

    runtime.stop()
    mock_sub.close.assert_called_once()
    assert len(runtime._subscriptions) == 0


def test_runtime_hook_dispatch_calls_callback():
    """Verify the dispatch thread delivers samples to the hook callback."""
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    call_log = []
    dispatched = threading.Event()

    def capture(sample_payload, ctx):
        call_log.append((sample_payload, ctx))
        dispatched.set()

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(capture)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    mock_sample = MagicMock()
    mock_sample.payload = b"raw-frame-bytes"
    mock_sample.timestamp = 12345.0
    mock_sample.metadata = {"width": 640}

    mock_bus._last_cb(mock_sample)
    assert dispatched.wait(timeout=2.0)

    assert len(call_log) == 1
    payload, ctx = call_log[0]
    assert payload == b"raw-frame-bytes"
    assert ctx.timestamp == 12345.0
    # Wildcard hook channel is just "frames"; the actual wire key would
    # travel on Sample.channel, but the MagicMock doesn't set it, so the
    # runtime falls back to the "default" sensor_name placeholder.
    assert ctx.channel == "frames"
    assert ctx.sensor_name == "default"
    assert ctx.twin_uuid == TEST_TWIN_UUID
    assert ctx.metadata == {"width": 640}

    runtime.stop()


def test_runtime_hook_error_does_not_crash():
    """A failing hook callback should not crash the dispatch thread."""
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    dispatched = threading.Event()

    def bad_handler(sample, ctx):
        dispatched.set()
        raise ValueError("boom")

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(bad_handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    mock_sample = MagicMock()
    mock_sample.payload = b""
    mock_sample.timestamp = 0.0
    mock_sample.metadata = {}

    mock_bus._last_cb(mock_sample)
    assert dispatched.wait(timeout=2.0)

    # Dispatch thread should still be alive after the error
    assert len(runtime._dispatch_threads) == 1
    assert runtime._dispatch_threads[0].is_alive()

    runtime.stop()


def test_runtime_drop_oldest_under_backpressure():
    """When inference is slow, intermediate samples are dropped."""
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    received_payloads = []
    processing = threading.Event()
    first_entered = threading.Event()

    def slow_handler(sample_payload, ctx):
        first_entered.set()
        processing.wait(timeout=5.0)
        received_payloads.append(sample_payload)

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(slow_handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    def make_sample(payload: bytes) -> MagicMock:
        s = MagicMock()
        s.payload = payload
        s.timestamp = 0.0
        s.metadata = {}
        return s

    # Push first sample — dispatch thread picks it up and blocks
    mock_bus._last_cb(make_sample(b"frame-1"))
    assert first_entered.wait(timeout=2.0)

    # Push two more while the handler is blocked — only the last survives
    mock_bus._last_cb(make_sample(b"frame-2"))
    mock_bus._last_cb(make_sample(b"frame-3"))

    # Unblock the handler
    processing.set()

    # Wait for the dispatch thread to process the queued sample
    time.sleep(0.2)

    runtime.stop()

    # frame-1 was processed first; frame-2 was replaced by frame-3
    assert received_payloads == [b"frame-1", b"frame-3"]


def test_runtime_fps_throttles_frame_hook_dispatch(monkeypatch):
    """``@cw.on_frame(..., fps=N)`` must skip samples arriving faster than 1/N s.

    Samples that beat the wall-clock floor are counted as drops on the
    hook's stats entry and never reach the user callback.
    """
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    received_payloads = []
    dispatched = threading.Event()

    def handler(sample_payload, ctx):
        received_payloads.append(sample_payload)
        dispatched.set()

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID, fps=2)(handler)

    runtime = WorkerRuntime(fake_cw)

    # Drive ``time.monotonic`` deterministically so the test does not
    # depend on real wall-clock timing. The dispatch loop reads it once
    # per sample; ticks of 0.1 s with fps=2 (min_interval=0.5 s) means
    # only every fifth sample passes the gate.
    fake_now = {"t": 0.0}

    def fake_monotonic():
        return fake_now["t"]

    monkeypatch.setattr(worker_runtime.time, "monotonic", fake_monotonic)

    runtime.start()

    def make_sample(payload: bytes) -> MagicMock:
        s = MagicMock()
        s.payload = payload
        s.timestamp = 0.0
        s.metadata = {}
        return s

    def push_and_wait(payload: bytes) -> None:
        dispatched.clear()
        mock_bus._last_cb(make_sample(payload))
        # Give the dispatch thread a chance to either consume or drop it.
        # Either path is fast — we just need the loop to run.
        dispatched.wait(timeout=0.2)
        time.sleep(0.02)

    fake_now["t"] = 0.0
    push_and_wait(b"frame-1")
    fake_now["t"] = 0.1
    push_and_wait(b"frame-2")
    fake_now["t"] = 0.2
    push_and_wait(b"frame-3")
    fake_now["t"] = 0.5
    push_and_wait(b"frame-4")
    fake_now["t"] = 0.7
    push_and_wait(b"frame-5")
    fake_now["t"] = 1.1
    push_and_wait(b"frame-6")

    runtime.stop()

    assert received_payloads == [b"frame-1", b"frame-4", b"frame-6"]
    hook_stats = runtime._hook_stats["handler"]
    assert hook_stats["frames"] == 3
    assert hook_stats["drops"] == 3


def test_runtime_no_fps_does_not_throttle():
    """Without ``fps=`` the dispatcher delivers every sample (drop-oldest
    backpressure aside)."""
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    received_payloads = []
    dispatched = threading.Event()

    def handler(sample_payload, ctx):
        received_payloads.append(sample_payload)
        dispatched.set()

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    def make_sample(payload: bytes) -> MagicMock:
        s = MagicMock()
        s.payload = payload
        s.timestamp = 0.0
        s.metadata = {}
        return s

    for i in range(3):
        dispatched.clear()
        mock_bus._last_cb(make_sample(f"frame-{i}".encode()))
        assert dispatched.wait(timeout=2.0)

    runtime.stop()

    assert received_payloads == [b"frame-0", b"frame-1", b"frame-2"]


def test_runtime_dispatches_generated_schedule_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_runtime, "_previous_cron_fire", _always_due_prev_fire)
    fake_cw = FakeCw()
    fake_cw.scheduled_run = threading.Event()
    fake_cw.schedule_calls = []
    (tmp_path / "wf_scheduled.py").write_text(
        "SCHEDULE_TRIGGERS = ["
        "{'node_uuid': 'schedule-node', 'cron': '* * * * *', 'timezone': 'UTC'}"
        "]\n"
        "def run(client=None):\n"
        "    client.schedule_calls.append('schedule-node')\n"
        "    client.scheduled_run.set()\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1
    runtime.start()

    try:
        assert fake_cw.scheduled_run.wait(timeout=3.0)
        assert fake_cw.schedule_calls == ["schedule-node"]
    finally:
        runtime.stop()


def test_runtime_dispatches_on_schedule_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_runtime, "_previous_cron_fire", _always_due_prev_fire)
    fake_cw = FakeCw()
    fake_cw.scheduled_run = threading.Event()
    fake_cw.schedule_contexts = []
    (tmp_path / "scheduled_worker.py").write_text(
        "@cw.on_schedule('* * * * *', timezone='UTC')\n"
        "def every_minute(ctx):\n"
        "    cw.schedule_contexts.append(ctx)\n"
        "    cw.scheduled_run.set()\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1
    runtime.start()

    try:
        assert fake_cw.scheduled_run.wait(timeout=3.0)
        assert len(fake_cw.schedule_contexts) == 1
        ctx = fake_cw.schedule_contexts[0]
        assert ctx.channel == "schedule"
        assert ctx.metadata["cron"] == "* * * * *"
        assert ctx.metadata["timezone"] == "UTC"
    finally:
        runtime.stop()


def test_previous_cron_fire_handles_six_field_subminute_cron():
    """6-field cron must parse with ``second_at_beginning=True`` so the
    leading field is seconds, not the trailing seconds slot.

    Without that flag, ``croniter`` would interpret ``"*/15 * * * * *"``
    as 5 fields with an unused tail and the previous tick would land
    on a minute boundary instead of a 15 s boundary — silently
    converting every sub-minute schedule into a once-per-minute one.

    Skipped when ``croniter`` isn't installed: it's an optional extra
    (``cyberwave[schedule]``) and the base SDK test env doesn't pull
    it in. The runtime itself logs a warning and disables schedule
    dispatch in that case — covered by inspection, not by this test.
    """
    pytest.importorskip("croniter")
    now = datetime(2030, 1, 1, 12, 0, 47, tzinfo=ZoneInfo("UTC"))
    prev = worker_runtime._previous_cron_fire("*/15 * * * * *", now)
    assert prev is not None
    assert prev == datetime(2030, 1, 1, 12, 0, 45, tzinfo=ZoneInfo("UTC"))


def test_runtime_fires_sub_minute_schedule_multiple_times_per_minute(
    tmp_path, monkeypatch
):
    """Sub-minute (6-field) schedules must fire more than once per minute.

    Pre-edge-sub-minute, the runtime had a ``"%Y-%m-%dT%H:%M"`` dedup
    key that capped every registration at one fire per minute. With
    the instant-level dedup, advancing the cron tick on each poll
    must produce a fresh dispatch on every poll cycle.
    """
    # Each call returns a tick strictly newer than the last so the
    # runtime sees a fresh fire on every poll after seeding.
    call_count = {"n": 0}

    def advancing_prev_fire(_cron, local_now):
        call_count["n"] += 1
        return local_now + timedelta(seconds=call_count["n"])

    monkeypatch.setattr(
        worker_runtime, "_previous_cron_fire", advancing_prev_fire
    )

    fake_cw = FakeCw()
    fake_cw.fire_count = 0
    fake_cw.fire_lock = threading.Lock()
    fake_cw.two_fires = threading.Event()
    (tmp_path / "scheduled_worker.py").write_text(
        "@cw.on_schedule('*/5 * * * * *', timezone='UTC')\n"
        "def every_five_seconds(ctx):\n"
        "    with cw.fire_lock:\n"
        "        cw.fire_count += 1\n"
        "        if cw.fire_count >= 2:\n"
        "            cw.two_fires.set()\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1
    runtime.start()

    try:
        # Poll cadence is 1 s; first poll seeds, second poll fires,
        # third poll fires again → 2 fires inside ~3 s wall time.
        assert fake_cw.two_fires.wait(timeout=5.0), (
            "Sub-minute schedule did not fire twice within 5 s — "
            "the runtime is probably back on per-minute dedup."
        )
    finally:
        runtime.stop()


def test_runtime_stop_waits_for_scheduled_run_before_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_runtime, "_previous_cron_fire", _always_due_prev_fire)
    fake_cw = FakeCw()
    fake_cw.scheduled_run_started = threading.Event()
    fake_cw.release_scheduled_run = threading.Event()
    fake_cw.disconnect_called = threading.Event()

    def disconnect():
        fake_cw.disconnect_called.set()

    fake_cw.disconnect = disconnect
    (tmp_path / "scheduled_worker.py").write_text(
        "@cw.on_schedule('* * * * *', timezone='UTC')\n"
        "def every_minute(ctx):\n"
        "    cw.scheduled_run_started.set()\n"
        "    cw.release_scheduled_run.wait(timeout=3.0)\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1
    runtime.start()
    assert fake_cw.scheduled_run_started.wait(timeout=3.0)

    stop_returned = threading.Event()

    def stop_runtime():
        runtime.stop()
        stop_returned.set()

    stop_thread = threading.Thread(target=stop_runtime, daemon=True)
    stop_thread.start()

    assert not fake_cw.disconnect_called.wait(timeout=0.2)
    assert not stop_returned.is_set()

    fake_cw.release_scheduled_run.set()
    assert stop_returned.wait(timeout=2.0)
    assert fake_cw.disconnect_called.is_set()
    stop_thread.join(timeout=1.0)


def test_runtime_subscribes_mqtt_hook_via_client_mqtt():
    """``@cw.on_mqtt`` hooks must subscribe directly through ``client.mqtt``.

    The MQTT trigger emitter assumes the runtime owns the subscription
    so the worker module stays declarative (mirroring how ``on_alert``
    flows through the Zenoh-MQTT bridge). Edge workflows whose trigger
    is ``mqtt`` rely on this path being separate from the data-bus
    subscription used by every other hook type.
    """

    class FakeMQTT:
        def __init__(self):
            self.connected = True
            self.topic_prefix = "local"
            self.subscriptions = []

        def connect(self):
            self.connected = True

        def subscribe(self, topic, handler, qos=0):
            self.subscriptions.append((topic, handler, qos))

    fake_mqtt = FakeMQTT()
    fake_cw = FakeCw()
    fake_cw.mqtt = fake_mqtt
    received = []

    def handler(payload, topic, ctx):
        received.append((payload, topic, ctx))

    fake_cw._hook_registry.on_mqtt(
        TEST_TWIN_UUID, subtopic="status", qos=1
    )(handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    try:
        assert len(fake_mqtt.subscriptions) == 1
        topic, registered_handler, qos = fake_mqtt.subscriptions[0]
        assert topic == f"localcyberwave/twin/{TEST_TWIN_UUID}/status"
        assert qos == 1

        registered_handler({"alert_type": "motor_overheat"})
        assert len(received) == 1
        payload, recv_topic, ctx = received[0]
        assert payload == {"alert_type": "motor_overheat"}
        assert recv_topic == topic
        assert ctx.channel == "mqtt/status"
        assert ctx.twin_uuid == TEST_TWIN_UUID
        assert ctx.metadata["subtopic"] == "status"
        assert ctx.metadata["qos"] == 1

        # ``on_mqtt`` hooks should not double-up via the data bus path.
        assert len(runtime._subscriptions) == 0
        assert len(runtime._dispatch_threads) == 0
    finally:
        runtime.stop()


def test_runtime_start_warms_up_before_subscribing_hooks(tmp_path):
    fake_cw = FakeCw()
    (tmp_path / "worker.py").write_text(
        "import builtins\n"
        f"reg = builtins.cw._hook_registry\n"
        f"reg.on_frame('{TEST_TWIN_UUID}')(lambda s, c: None)\n"
    )

    runtime = WorkerRuntime(fake_cw)
    runtime.load(str(tmp_path))
    call_order: list[str] = []
    original_warm = runtime._warm_up_models
    original_subscribe = runtime._subscribe_hook

    def tracked_warm() -> None:
        call_order.append("warm_up")
        original_warm()

    def tracked_subscribe(hook) -> None:
        call_order.append("subscribe")
        original_subscribe(hook)

    runtime._warm_up_models = tracked_warm  # type: ignore[method-assign]
    runtime._subscribe_hook = tracked_subscribe  # type: ignore[method-assign]
    runtime.start()
    try:
        assert call_order == ["warm_up", "subscribe"]
    finally:
        runtime.stop()


def test_runtime_start_count_includes_schedule_hooks(tmp_path, caplog):
    fake_cw = FakeCw()
    (tmp_path / "scheduled_worker.py").write_text(
        "@cw.on_schedule('* * * * *', timezone='UTC')\n"
        "def every_minute(ctx):\n"
        "    pass\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1

    with caplog.at_level("INFO", logger="cyberwave.workers.runtime"):
        runtime.start()

    try:
        assert "Worker runtime started with 1 hook(s)" in caplog.text
    finally:
        runtime.stop()


def test_runtime_hook_error_sends_alert():
    """A failing hook callback should publish a worker_runtime_error alert."""
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    dispatched = threading.Event()

    def bad_handler(sample, ctx):
        dispatched.set()
        raise RuntimeError("Caller node did not produce field 'detection_count'")

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(bad_handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    mock_sample = MagicMock()
    mock_sample.payload = b""
    mock_sample.timestamp = 0.0
    mock_sample.metadata = {}

    mock_bus._last_cb(mock_sample)
    assert dispatched.wait(timeout=2.0)
    time.sleep(0.1)

    runtime.stop()

    assert len(fake_cw.published_alerts) == 1
    alert = fake_cw.published_alerts[0]
    assert alert["twin_uuid"] == TEST_TWIN_UUID
    assert alert["alert_type"] == "worker_runtime_error"
    assert alert["severity"] == "error"
    assert alert["category"] == "technical"
    assert "bad_handler" in alert["name"]
    assert "RuntimeError" in alert["description"]
    assert "detection_count" in alert["description"]


def test_runtime_hook_error_alert_cooldown(monkeypatch):
    """Repeated hook errors within the cooldown window should not send duplicate alerts."""
    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    call_count = {"n": 0}
    second_dispatched = threading.Event()

    def bad_handler(sample, ctx):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            second_dispatched.set()
        raise ValueError("repeated error")

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(bad_handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    mock_sample = MagicMock()
    mock_sample.payload = b""
    mock_sample.timestamp = 0.0
    mock_sample.metadata = {}

    mock_bus._last_cb(mock_sample)
    time.sleep(0.1)
    mock_bus._last_cb(mock_sample)
    assert second_dispatched.wait(timeout=2.0)
    time.sleep(0.1)

    runtime.stop()

    assert len(fake_cw.published_alerts) == 1


def test_runtime_hook_error_alert_after_cooldown_expires(monkeypatch):
    """After the cooldown expires, a new alert should be sent."""
    monkeypatch.setattr(worker_runtime, "HOOK_ERROR_ALERT_COOLDOWN_S", 0.0)

    mock_bus = MagicMock()
    mock_bus.key_prefix = "cw"
    call_count = {"n": 0}
    second_dispatched = threading.Event()

    def bad_handler(sample, ctx):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            second_dispatched.set()
        raise ValueError("error again")

    mock_bus.backend.subscribe.side_effect = lambda key, cb: (
        setattr(mock_bus, "_last_cb", cb) or MagicMock()
    )

    fake_cw = FakeCw(data_bus=mock_bus)
    fake_cw._hook_registry.on_frame(TEST_TWIN_UUID)(bad_handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    mock_sample = MagicMock()
    mock_sample.payload = b""
    mock_sample.timestamp = 0.0
    mock_sample.metadata = {}

    mock_bus._last_cb(mock_sample)
    time.sleep(0.1)
    mock_bus._last_cb(mock_sample)
    assert second_dispatched.wait(timeout=2.0)
    time.sleep(0.1)

    runtime.stop()

    assert len(fake_cw.published_alerts) == 2


def test_runtime_mqtt_hook_error_sends_alert():
    """A failing MQTT hook callback should publish a worker_runtime_error alert."""

    class FakeMQTT:
        def __init__(self):
            self.connected = True
            self.topic_prefix = ""
            self.subscriptions = []

        def subscribe(self, topic, handler, qos=0):
            self.subscriptions.append((topic, handler, qos))

    fake_mqtt = FakeMQTT()
    fake_cw = FakeCw()
    fake_cw.mqtt = fake_mqtt

    def bad_handler(payload, topic, ctx):
        raise RuntimeError("mqtt boom")

    fake_cw._hook_registry.on_mqtt(
        TEST_TWIN_UUID, subtopic="custom/event", qos=0
    )(bad_handler)

    runtime = WorkerRuntime(fake_cw)
    runtime.start()

    try:
        assert len(fake_mqtt.subscriptions) == 1
        _topic, registered_handler, _qos = fake_mqtt.subscriptions[0]
        registered_handler({"test": True})
        time.sleep(0.1)

        assert len(fake_cw.published_alerts) == 1
        alert = fake_cw.published_alerts[0]
        assert alert["twin_uuid"] == TEST_TWIN_UUID
        assert alert["alert_type"] == "worker_runtime_error"
        assert alert["severity"] == "error"
        assert "bad_handler" in alert["name"]
        assert "RuntimeError" in alert["description"]
    finally:
        runtime.stop()


def test_runtime_scheduled_workflow_error_sends_alert(tmp_path, monkeypatch):
    """A failing scheduled workflow should publish a worker_runtime_error alert."""
    monkeypatch.setattr(worker_runtime, "_previous_cron_fire", _always_due_prev_fire)
    fake_cw = FakeCw()
    fake_cw.scheduled_run = threading.Event()
    (tmp_path / "wf_failing.py").write_text(
        "SCHEDULE_TRIGGERS = ["
        "{'node_uuid': 'fail-node', 'cron': '* * * * *', 'timezone': 'UTC'}"
        "]\n"
        "def run(client=None):\n"
        "    client.scheduled_run.set()\n"
        "    raise RuntimeError('schedule boom')\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1
    runtime.start()

    try:
        assert fake_cw.scheduled_run.wait(timeout=3.0)
        time.sleep(0.2)

        assert len(fake_cw.published_alerts) == 1
        alert = fake_cw.published_alerts[0]
        assert alert["twin_uuid"] == TEST_TWIN_UUID
        assert alert["alert_type"] == "worker_runtime_error"
        assert alert["severity"] == "error"
        assert "schedule:fail-node" in alert["name"]
        assert "RuntimeError" in alert["description"]
    finally:
        runtime.stop()
