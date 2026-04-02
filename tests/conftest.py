"""Shared test configuration.

When the auto-generated ``cyberwave.rest`` package is absent (it requires the
backend running + ``python-sdk-gen.sh``), we inject lightweight stubs into
``sys.modules`` so that importing ``cyberwave.data.*`` does not fail.

This file is loaded by pytest before any test module is collected, so the
stubs are in place before the import chain is triggered.
"""

import importlib
import sys
import types


def _rest_module_is_real() -> bool:
    """Return True if the auto-generated REST client is available."""
    try:
        mod = importlib.import_module("cyberwave.rest")
        return hasattr(mod, "DefaultApi")
    except (ImportError, ModuleNotFoundError):
        return False


class _Stub:
    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        return _Stub()

    def __call__(self, *a, **kw):
        return _Stub()


def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda n: _Stub  # type: ignore[attr-defined]
    mod.__path__ = []
    return mod


def _inject_rest_stubs() -> None:
    """Pre-seed ``sys.modules`` with stubs for ``cyberwave.rest.*``."""
    if _rest_module_is_real():
        return

    _REST_MODULES = [
        "cyberwave.rest",
        "cyberwave.rest.models",
        "cyberwave.rest.models.twin_joint_calibration_schema",
        "cyberwave.rest.models.universal_schema_patch_schema",
        "cyberwave.rest.models.twin_universal_schema_patch_schema",
    ]

    for name in _REST_MODULES:
        sys.modules[name] = _make_stub_module(name)


_inject_rest_stubs()
