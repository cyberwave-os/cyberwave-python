"""Tests for cyberwave.models.runtimes.tensorrt_rt — TensorRT backend."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from cyberwave.models.runtimes.tensorrt_rt import TensorRTRuntime
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestTensorRTRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = TensorRTRuntime


class TestTensorRTRuntimeIsAvailable:
    def test_available_when_installed(self):
        with patch.dict("sys.modules", {"tensorrt": MagicMock()}):
            assert TensorRTRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict("sys.modules", {"tensorrt": None}):
            assert TensorRTRuntime().is_available() is False


class TestTensorRTRuntimeLoad:
    def test_load_deserialises_engine(self):
        mock_trt = MagicMock()
        mock_engine = MagicMock()
        mock_engine.num_bindings = 4
        mock_trt.Runtime.return_value.deserialize_cuda_engine.return_value = mock_engine

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            rt = TensorRTRuntime()
            with patch("builtins.open", mock_open(read_data=b"\x00")):
                handle = rt.load("/path/model.engine")
            assert handle is mock_engine

    def test_load_raises_on_null_engine(self):
        mock_trt = MagicMock()
        mock_trt.Runtime.return_value.deserialize_cuda_engine.return_value = None

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            rt = TensorRTRuntime()
            with patch("builtins.open", mock_open(read_data=b"\x00")):
                with pytest.raises(RuntimeError, match="Failed to deserialise"):
                    rt.load("/path/bad.engine")


class TestTensorRTRuntimePredict:
    def test_predict_raises_not_implemented(self):
        rt = TensorRTRuntime()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            rt.predict(MagicMock(), MagicMock())


class TestTensorRTSupportsPredictFlag:
    def test_supports_predict_is_false(self):
        assert TensorRTRuntime().supports_predict is False
