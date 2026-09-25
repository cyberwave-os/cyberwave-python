"""Runtime boundary tests: importing a worker must NOT start the event loop."""

import builtins
import sys
import threading

import pytest

from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.loader import load_workers


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if hasattr(builtins, "cw"):
        delattr(builtins, "cw")
    for key in list(sys.modules):
        if key.startswith("cyberwave_worker_"):
            del sys.modules[key]


class FakeCw:
    """Minimal stub that mimics the ``Cyberwave`` client for worker imports."""

    def __init__(self):
        self._hook_registry = HookRegistry()
        self.config = type("Config", (), {"twin_uuid": "test-uuid"})()

    def on_frame(self, *args, **kwargs):
        return self._hook_registry.on_frame(*args, **kwargs)


def test_import_worker_does_not_start_loop(tmp_path):
    """Importing a worker module must not create subscriptions or threads."""
    fake_cw = FakeCw()

    (tmp_path / "detector.py").write_text(
        "@cw.on_frame(cw.config.twin_uuid)\ndef handler(sample, ctx):\n    pass\n"
    )

    initial_threads = threading.active_count()
    count = load_workers(tmp_path, cw_instance=fake_cw)

    assert count == 1
    assert len(fake_cw._hook_registry.hooks) == 1
    assert threading.active_count() <= initial_threads + 1


def test_hooks_registered_but_not_active(tmp_path):
    """Hooks are only registered during import; activation is the runtime's job."""
    fake_cw = FakeCw()

    (tmp_path / "worker.py").write_text(
        "@cw.on_frame(cw.config.twin_uuid)\n"
        "def on_frame(sample, ctx):\n"
        "    pass\n"
        "\n"
        "@cw.on_frame(cw.config.twin_uuid, sensor='back')\n"
        "def on_back(sample, ctx):\n"
        "    pass\n"
    )

    load_workers(tmp_path, cw_instance=fake_cw)
    hooks = fake_cw._hook_registry.hooks

    assert len(hooks) == 2
    # Default is wildcard: channel is just "frames"; explicit sensor keeps "frames/back".
    assert hooks[0].channel == "frames"
    assert hooks[1].channel == "frames/back"
