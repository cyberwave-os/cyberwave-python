"""Tests for cyberwave.models.runtimes.torch_rt — PyTorch native backend."""

import numpy as np
from unittest.mock import MagicMock, patch

from cyberwave.models.runtimes.torch_rt import TorchRuntime
from cyberwave.models.types import ClassificationResult, CustomResult
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
    def test_predict_dispatches_classification_output(self):
        mock_torch = MagicMock()
        mock_model = MagicMock()

        tensor_out = MagicMock()
        tensor_out.cpu.return_value.float.return_value.numpy.return_value = np.array(
            [[0.1, 0.7, 0.2]], dtype=np.float32
        )
        # Assigning __call__ on a MagicMock *instance* has no effect — dunder
        # lookup goes through the type — so the model has to be wired via
        # return_value or the forward pass silently yields a bare MagicMock.
        mock_model.return_value = tensor_out
        mock_torch.from_numpy.return_value.unsqueeze.return_value.to.return_value = (
            MagicMock()
        )
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.Tensor = type(tensor_out)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            rt = TorchRuntime()
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            result = rt.predict(mock_model, img)
            assert isinstance(result, ClassificationResult)

    def test_predict_returns_custom_result_for_3d_output(self):
        """A 3-D tensor must NOT be guessed as YOLO detections.

        This runtime is the catch-all for arbitrary user checkpoints, where
        (B, C, N) is equally consistent with segmentation logits, per-token
        embeddings or batched depth. Decoding boxes from those produced
        confidently-wrong detections, so the raw tensor is handed back instead.
        """
        mock_torch = MagicMock()
        mock_model = MagicMock()

        tensor_out = MagicMock()
        tensor_out.cpu.return_value.float.return_value.numpy.return_value = np.zeros(
            (1, 84, 8400), dtype=np.float32
        )
        mock_model.return_value = tensor_out
        mock_torch.from_numpy.return_value.unsqueeze.return_value.to.return_value = (
            MagicMock()
        )
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.Tensor = type(tensor_out)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            rt = TorchRuntime()
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            result = rt.predict(mock_model, img)
            assert isinstance(result, CustomResult)
            assert result.data.shape == (1, 84, 8400)


class TestTorchSupportsPredictFlag:
    def test_supports_predict_is_true(self):
        assert TorchRuntime().supports_predict is True
