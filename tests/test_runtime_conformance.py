"""Shared conformance checks for ModelRuntime implementations.

Every runtime test module should subclass ``RuntimeConformanceMixin``
and set ``runtime_class`` to validate that the basic contract is met.
"""

import inspect

from cyberwave.models.runtimes.base import ModelRuntime


class RuntimeConformanceMixin:
    """Mixin that verifies a ``ModelRuntime`` subclass satisfies the contract.

    Subclasses must set ``runtime_class`` to the class under test.
    """

    runtime_class: type[ModelRuntime]

    def _instance(self) -> ModelRuntime:
        return self.runtime_class()

    def test_has_name(self):
        assert isinstance(self._instance().name, str)
        assert len(self._instance().name) > 0

    def test_is_available_returns_bool(self):
        result = self._instance().is_available()
        assert isinstance(result, bool)

    def test_load_signature(self):
        sig = inspect.signature(self.runtime_class.load)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "model_path" in params
        assert "device" in params

    def test_predict_signature(self):
        sig = inspect.signature(self.runtime_class.predict)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "model_handle" in params
        assert "input_data" in params
        assert "confidence" in params
        assert "classes" in params

    def test_is_subclass_of_model_runtime(self):
        assert issubclass(self.runtime_class, ModelRuntime)
