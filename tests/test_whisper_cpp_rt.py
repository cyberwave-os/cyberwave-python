"""Tests for cyberwave.models.runtimes.whisper_cpp_rt."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cyberwave.models.runtimes.whisper_cpp_rt import WhisperCppRuntime


class TestWhisperCppRuntimeAvailability:
    def test_available_with_pywhispercpp(self):
        with patch.dict(
            "sys.modules",
            {
                "pywhispercpp": MagicMock(),
                "pywhispercpp.model": MagicMock(),
            },
        ):
            assert WhisperCppRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict(
            "sys.modules",
            {
                "pywhispercpp": None,
                "pywhispercpp.model": None,
            },
        ):
            assert WhisperCppRuntime().is_available() is False


class TestWhisperCppRuntime:
    def test_load_creates_pywhispercpp_model(self, tmp_path):
        model_file = tmp_path / "ggml-tiny.en-q5_1.bin"
        model_file.write_bytes(b"weights")

        model_cls = MagicMock()
        module = MagicMock(Model=model_cls)

        with patch.dict(
            "sys.modules",
            {
                "pywhispercpp": MagicMock(),
                "pywhispercpp.model": module,
            },
        ):
            handle = WhisperCppRuntime().load(str(model_file), n_threads=2)

        assert handle is model_cls.return_value
        model_cls.assert_called_once_with(str(model_file), n_threads=2)

    def test_predict_returns_transcription_payload(self, tmp_path):
        audio_file = tmp_path / "chunk.wav"
        audio_file.write_bytes(b"RIFF")
        handle = MagicMock()
        handle.transcribe.return_value = [
            SimpleNamespace(text=" hello", t0=0, t1=1250),
            SimpleNamespace(text=" world", start=1.25, end=2.5),
        ]

        result = WhisperCppRuntime().predict(handle, str(audio_file), language="en")

        handle.transcribe.assert_called_once_with(str(audio_file), language="en")
        assert result.detections == []
        assert result.raw == {
            "text": "hello world",
            "segments": [
                {"text": "hello", "start": 0.0, "end": 1.25},
                {"text": "world", "start": 1.25, "end": 2.5},
            ],
            "language": "en",
        }
        assert result.metadata["text"] == "hello world"
