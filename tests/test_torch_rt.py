"""Tests for cyberwave.models.runtimes.torch_rt — PyTorch native backend."""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.models.runtimes.torch_rt import TorchRuntime
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestTorchRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = TorchRuntime


class TestTorchRuntimeIsAvailable:
    def test_available_when_installed(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            assert TorchRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert TorchRuntime().is_available() is False


class TestTorchRuntimeLoad:
    def test_load_tries_jit_first(self):
        mock_torch = MagicMock()
        mock_model = MagicMock()
        mock_torch.jit.load.return_value = mock_model

        with patch.dict("sys.modules", {"torch": mock_torch}):
            rt = TorchRuntime()
            handle = rt.load("/path/model.pt", device="cpu")
            mock_torch.jit.load.assert_called_once_with("/path/model.pt", map_location="cpu")
            assert handle is mock_model
            mock_model.eval.assert_called_once()

    def test_load_falls_back_to_torch_load(self):
        mock_torch = MagicMock()
        mock_torch.jit.load.side_effect = RuntimeError("not a TorchScript model")
        mock_model = MagicMock()
        mock_torch.load.return_value = mock_model

        with patch.dict("sys.modules", {"torch": mock_torch}):
            rt = TorchRuntime()
            handle = rt.load("/path/model.pth", device="cuda:0")
            mock_torch.load.assert_called_once_with(
                "/path/model.pth", map_location="cuda:0", weights_only=True,
            )
            assert handle is mock_model

    def test_load_respects_device(self):
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            rt = TorchRuntime()
            rt.load("/path/model.pt", device="cuda:1")
            mock_torch.jit.load.assert_called_once_with("/path/model.pt", map_location="cuda:1")


class TestTorchRuntimePredict:
    def test_predict_raises_not_implemented(self):
        rt = TorchRuntime()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            rt.predict(MagicMock(), MagicMock())


class TestTorchSupportsPredictFlag:
    def test_supports_predict_is_false(self):
        assert TorchRuntime().supports_predict is False
