"""Tests for WorkerRuntime lifecycle."""

import builtins
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

import cyberwave.workers.runtime as worker_runtime
from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.runtime import WorkerRuntime


TEST_TWIN_UUID = "00000000-0000-0000-0000-000000000001"


class FakeCw:
    """Minimal stub of the Cyberwave client for runtime tests."""

    def __init__(self, *, data_bus=None):
        self._hook_registry = HookRegistry()
        self.config = type("Config", (), {"twin_uuid": TEST_TWIN_UUID})()
        self._data_bus = data_bus

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


def test_runtime_dispatches_generated_schedule_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_runtime, "_cron_matches", lambda _cron, _now: True)
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
        assert fake_cw.scheduled_run.wait(timeout=2.0)
        assert fake_cw.schedule_calls == ["schedule-node"]
    finally:
        runtime.stop()


def test_runtime_dispatches_on_schedule_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_runtime, "_cron_matches", lambda _cron, _now: True)
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
        assert fake_cw.scheduled_run.wait(timeout=2.0)
        assert len(fake_cw.schedule_contexts) == 1
        ctx = fake_cw.schedule_contexts[0]
        assert ctx.channel == "schedule"
        assert ctx.metadata["cron"] == "* * * * *"
        assert ctx.metadata["timezone"] == "UTC"
    finally:
        runtime.stop()


def test_runtime_stop_waits_for_scheduled_run_before_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_runtime, "_cron_matches", lambda _cron, _now: True)
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
        "    cw.release_scheduled_run.wait(timeout=2.0)\n"
    )

    runtime = WorkerRuntime(fake_cw)
    assert runtime.load(str(tmp_path)) == 1
    runtime.start()
    assert fake_cw.scheduled_run_started.wait(timeout=2.0)

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
