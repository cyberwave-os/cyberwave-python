import importlib.util
from pathlib import Path

import pytest
import os
import cyberwave as cw


class DummyClient:
    async def login(self):
        pass

    async def aclose(self):
        pass


class DummyRobot:
    def __init__(self):
        self.events = []

    def connect(self, ip):
        self.events.append(f"connect:{ip}")

    def initialize_sensors(self, sensors):
        self.events.append(f"init:{sensors}")

    def takeoff(self):
        self.events.append("takeoff")

    def get_status(self):
        return {"battery": 95}

    def land(self):
        self.events.append("land")

    def disconnect(self):
        self.events.append("disconnect")


dummy_robot = DummyRobot()


def test_tello_example_runs(monkeypatch):
    # Mock HTTP POSTs from the SDK to avoid real backend calls
    import cyberwave.http as httpmod

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    calls: list[str] = []

    def _post(url, *a, **kw):
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(httpmod.requests, "post", _post)
    monkeypatch.setattr(cw, "Client", lambda *a, **kw: DummyClient())
    monkeypatch.setattr(cw, "Robot", lambda *a, **kw: dummy_robot)
    monkeypatch.setenv("CYBERWAVE_TOKEN", "dummy-token")
    monkeypatch.setenv("CYBERWAVE_TELLO_TWIN_UUID", "tello-twin-uuid")

    spec = importlib.util.spec_from_file_location(
        "tello_example",
        Path(__file__).resolve().parents[1] / "examples" / "tello_sdk_example.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()

    # Expect at least teleop start + two commands (takeoff, land)
    assert len(calls) >= 3
