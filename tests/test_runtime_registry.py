"""Tests for cyberwave.models.runtimes — registry and runtime resolution."""

import pytest

from cyberwave.models.runtimes import (
    _RUNTIME_REGISTRY,
    available_runtimes,
    get_runtime,
    register_runtime,
)
from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult


class _AvailableRuntime(ModelRuntime):
    name = "test_available"

    def is_available(self):
        return True

    def load(self, model_path, *, device=None, **kwargs):
        return None

    def predict(self, model_handle, input_data, *, confidence=0.5, classes=None, **kwargs):
        return PredictionResult()


class _UnavailableRuntime(ModelRuntime):
    name = "test_unavailable"

    def is_available(self):
        return False

    def load(self, model_path, *, device=None, **kwargs):
        return None

    def predict(self, model_handle, input_data, *, confidence=0.5, classes=None, **kwargs):
        return PredictionResult()


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """Remove test runtimes after each test."""
    yield
    _RUNTIME_REGISTRY.pop("test_available", None)
    _RUNTIME_REGISTRY.pop("test_unavailable", None)


class TestRuntimeRegistry:
    def test_register_and_get(self):
        register_runtime(_AvailableRuntime)
        rt = get_runtime("test_available")
        assert isinstance(rt, _AvailableRuntime)

    def test_unknown_runtime_raises(self):
        with pytest.raises(ValueError, match="Unknown model runtime"):
            get_runtime("nonexistent_runtime_xyz")

    def test_unavailable_runtime_raises(self):
        register_runtime(_UnavailableRuntime)
        with pytest.raises(ImportError, match="dependencies are not installed"):
            get_runtime("test_unavailable")

    def test_available_runtimes(self):
        register_runtime(_AvailableRuntime)
        register_runtime(_UnavailableRuntime)
        avail = available_runtimes()
        assert "test_available" in avail
        assert "test_unavailable" not in avail

    def test_ultralytics_is_registered(self):
        assert "ultralytics" in _RUNTIME_REGISTRY
