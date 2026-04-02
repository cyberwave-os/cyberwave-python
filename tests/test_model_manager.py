"""Tests for cyberwave.models.manager — ModelManager."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.manager import ModelManager
from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRuntime(ModelRuntime):
    """Minimal runtime for testing without real ML dependencies."""

    name = "fake"

    def is_available(self) -> bool:
        return True

    def load(self, model_path, *, device=None, **kwargs):
        return {"path": model_path, "device": device}

    def predict(self, model_handle, input_data, *, confidence=0.5, classes=None, **kwargs):
        return PredictionResult()


@pytest.fixture(autouse=True)
def _register_fake_runtime():
    """Temporarily register _FakeRuntime for the duration of each test."""
    from cyberwave.models.runtimes import _RUNTIME_REGISTRY, register_runtime

    register_runtime(_FakeRuntime)
    yield
    _RUNTIME_REGISTRY.pop("fake", None)


# ---------------------------------------------------------------------------
# Model directory from env
# ---------------------------------------------------------------------------


class TestModelDirFromEnv:
    def test_explicit_model_dir(self, tmp_path):
        mgr = ModelManager(model_dir=str(tmp_path))
        assert mgr._model_dir == tmp_path

    def test_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_MODEL_DIR", str(tmp_path))
        mgr = ModelManager()
        assert mgr._model_dir == tmp_path

    def test_fallback(self, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_MODEL_DIR", raising=False)
        mgr = ModelManager()
        assert str(mgr._model_dir).endswith(".cyberwave/models") or str(
            mgr._model_dir
        ) == str(Path("/app/models"))


# ---------------------------------------------------------------------------
# _detect_runtime heuristics
# ---------------------------------------------------------------------------


class TestDefaultDevice:
    def test_default_device_from_constructor(self, tmp_path):
        mgr = ModelManager(model_dir=str(tmp_path), default_device="cuda:1")
        assert mgr._default_device == "cuda:1"

    def test_default_device_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_MODEL_DEVICE", "cuda:2")
        mgr = ModelManager(model_dir=str(tmp_path))
        assert mgr._default_device == "cuda:2"

    def test_constructor_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_MODEL_DEVICE", "cuda:2")
        mgr = ModelManager(model_dir=str(tmp_path), default_device="cpu")
        assert mgr._default_device == "cpu"

    def test_device_used_in_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_MODEL_DEVICE", "cuda:3")
        (tmp_path / "yolov8n.pt").touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        m = mgr.load("yolov8n", runtime="fake")
        assert m.device == "cuda:3"


class TestDetectRuntime:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("yolov8n", "ultralytics"),
            ("yolov11s", "ultralytics"),
            ("YOLO-custom", "ultralytics"),
            ("yolov5m", "ultralytics"),
            ("haar-face", "opencv"),
            ("background-subtraction-mog2", "opencv"),
            ("cascade-classifier", "opencv"),
        ],
    )
    def test_known_ids(self, model_id, expected):
        assert ModelManager._detect_runtime(model_id) == expected

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Cannot auto-detect runtime"):
            ModelManager._detect_runtime("my-custom-model")


# ---------------------------------------------------------------------------
# _detect_runtime_from_extension
# ---------------------------------------------------------------------------


class TestDetectRuntimeFromExtension:
    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".pt", "ultralytics"),
            (".onnx", "onnxruntime"),
            (".tflite", "tflite"),
            (".xml", "opencv"),
            (".engine", "tensorrt"),
            (".PT", "ultralytics"),
            (".unknown", "ultralytics"),
        ],
    )
    def test_mapping(self, ext, expected):
        assert ModelManager._detect_runtime_from_extension(ext) == expected


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


class TestDetectDevice:
    def test_cpu_when_torch_unavailable(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert ModelManager._detect_device() == "cpu"

    def test_cpu_when_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert ModelManager._detect_device() == "cpu"

    def test_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert ModelManager._detect_device() == "cuda:0"


# ---------------------------------------------------------------------------
# _resolve_model_path
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    def test_exact_file(self, tmp_path):
        (tmp_path / "yolov8n.pt").touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        result = mgr._resolve_model_path("yolov8n", "ultralytics")
        assert result == tmp_path / "yolov8n.pt"

    def test_subdirectory(self, tmp_path):
        sub = tmp_path / "my_model"
        sub.mkdir()
        (sub / "weights.pt").touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        result = mgr._resolve_model_path("my_model", "ultralytics")
        assert result == sub / "weights.pt"

    def test_ultralytics_fallback(self, tmp_path):
        mgr = ModelManager(model_dir=str(tmp_path))
        result = mgr._resolve_model_path("yolov8n", "ultralytics")
        assert result == Path("yolov8n")

    def test_non_ultralytics_not_found(self, tmp_path):
        mgr = ModelManager(model_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="not found"):
            mgr._resolve_model_path("missing", "onnxruntime")


# ---------------------------------------------------------------------------
# load() caching
# ---------------------------------------------------------------------------


class TestLoadCaching:
    def test_second_load_returns_cached(self, tmp_path):
        (tmp_path / "yolov8n.pt").touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        m1 = mgr.load("yolov8n", runtime="fake", device="cpu")
        m2 = mgr.load("yolov8n", runtime="fake", device="cpu")
        assert m1 is m2

    def test_different_device_not_cached(self, tmp_path):
        (tmp_path / "yolov8n.pt").touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        m1 = mgr.load("yolov8n", runtime="fake", device="cpu")
        m2 = mgr.load("yolov8n", runtime="fake", device="cuda:0")
        assert m1 is not m2

    def test_loaded_model_properties(self, tmp_path):
        (tmp_path / "yolov8n.pt").touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        m = mgr.load("yolov8n", runtime="fake", device="cpu")
        assert isinstance(m, LoadedModel)
        assert m.name == "yolov8n"
        assert m.runtime == "fake"
        assert m.device == "cpu"

    def test_cache_key_uses_resolved_runtime(self, tmp_path):
        """Explicit runtime= and auto-detected runtime share the same
        cache entry when they resolve to the same backend."""
        from cyberwave.models.runtimes import _RUNTIME_REGISTRY, register_runtime

        class _UltralyticsLike(_FakeRuntime):
            name = "ultralytics"

            def is_available(self):
                return True

        old = _RUNTIME_REGISTRY.get("ultralytics")
        register_runtime(_UltralyticsLike)
        try:
            (tmp_path / "yolov8n.pt").touch()
            mgr = ModelManager(model_dir=str(tmp_path))
            m1 = mgr.load("yolov8n", device="cpu")
            m2 = mgr.load("yolov8n", runtime="ultralytics", device="cpu")
            assert m1 is m2
        finally:
            if old is not None:
                _RUNTIME_REGISTRY["ultralytics"] = old


# ---------------------------------------------------------------------------
# load_from_file()
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    def test_file_not_found(self):
        mgr = ModelManager()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            mgr.load_from_file("/nonexistent/model.pt")

    def test_success(self, tmp_path):
        model_file = tmp_path / "custom.pt"
        model_file.touch()
        mgr = ModelManager(model_dir=str(tmp_path))
        m = mgr.load_from_file(str(model_file), runtime="fake")
        assert m.name == "custom"
        assert m.runtime == "fake"
