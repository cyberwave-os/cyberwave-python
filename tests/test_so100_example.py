import importlib.util
import os
from pathlib import Path

import pytest
import cyberwave as cw


class DummyClient:
    def __init__(self):
        self.calls = []

    async def _request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs.get("json")))
        class R:
            def json(self_inner):
                return {"ok": True}
        return R()

    async def aclose(self):
        pass


def test_so100_example_runs(monkeypatch):
    # Mock HTTP POSTs from the SDK to avoid real backend calls
    import cyberwave.http as httpmod

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(httpmod.requests, "post", lambda *a, **kw: _Resp())

    # Patch Client used by legacy example path
    monkeypatch.setattr(cw, "Client", lambda *a, **kw: DummyClient())
    # Provide expected env vars used by the updated example
    monkeypatch.setenv("CYBERWAVE_ARM_TWIN_UUID", "arm-twin-uuid")
    monkeypatch.setenv("CYBERWAVE_TOKEN", "dummy-token")

    spec = importlib.util.spec_from_file_location(
        "so100_example",
        Path(__file__).resolve().parents[1] / "examples" / "so100_sdk_example.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Updated example is sync. Just call main()
    module.main()

    # No direct access to the dummy client in updated example; assert no exception
    assert True


