"""Tests for cyberwave.models.runtimes.faster_whisper_rt."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.models.runtimes import _RUNTIME_REGISTRY
from cyberwave.models.runtimes.faster_whisper_rt import FasterWhisperRuntime
from cyberwave.models.types import TextResult


def test_faster_whisper_runtime_is_registered() -> None:
    assert "faster_whisper" in _RUNTIME_REGISTRY
    assert _RUNTIME_REGISTRY["faster_whisper"] is FasterWhisperRuntime


class TestFasterWhisperRuntimeAvailability:
    def test_available_with_faster_whisper(self):
        with patch.dict(
            "sys.modules",
            {"faster_whisper": MagicMock()},
        ):
            assert FasterWhisperRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict(
            "sys.modules",
            {"faster_whisper": None},
        ):
            assert FasterWhisperRuntime().is_available() is False


def _patched_faster_whisper(model_cls: MagicMock, downloaded_path: str):
    """Fake ``faster_whisper`` package with both ``WhisperModel`` and
    ``utils.download_model`` so the runtime's pre-resolve path is exercised."""
    fake_pkg = ModuleType("faster_whisper")
    fake_pkg.WhisperModel = model_cls
    fake_utils = ModuleType("faster_whisper.utils")
    fake_utils.download_model = MagicMock(return_value=downloaded_path)
    fake_pkg.utils = fake_utils
    return fake_pkg, fake_utils


class TestFasterWhisperRuntime:
    def test_load_pre_resolves_snapshot_then_calls_whisper_model(self, tmp_path):
        """When model_path is not a local directory, ``download_model`` is
        called to pre-resolve the HuggingFace snapshot path before
        ``WhisperModel`` is constructed (prevents CWD short-circuit)."""
        # model_path does NOT exist as a directory — triggers the HF download path
        model_path = str(tmp_path / "nonexistent" / "tiny.en")
        download_root = str(tmp_path / "cache")
        resolved_snapshot = str(
            tmp_path / "cache" / "models--Systran--faster-whisper-tiny.en"
        )

        model_cls = MagicMock()
        fake_pkg, fake_utils = _patched_faster_whisper(model_cls, resolved_snapshot)

        with patch.dict(
            "sys.modules",
            {"faster_whisper": fake_pkg, "faster_whisper.utils": fake_utils},
        ):
            handle = FasterWhisperRuntime().load(
                model_path,
                faster_whisper_model_id="tiny.en",
                compute_type="int8",
                device="cpu",
                download_root=download_root,
            )

        assert handle is model_cls.return_value
        fake_utils.download_model.assert_called_once_with(
            "tiny.en",
            output_dir=download_root,
            local_files_only=False,
        )
        model_cls.assert_called_once()
        positional = model_cls.call_args.args
        call_kwargs = model_cls.call_args.kwargs
        assert positional == (resolved_snapshot,)
        assert "download_root" not in call_kwargs
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["compute_type"] == "int8"
        assert call_kwargs["local_files_only"] is False

    def test_load_uses_local_dir_without_downloading(self, tmp_path):
        """When model_path points to a complete local snapshot the runtime
        uses it directly — ``download_model`` must never be called."""
        model_dir = tmp_path / "models" / "tiny.en"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"ct2")

        model_cls = MagicMock()
        fake_pkg, fake_utils = _patched_faster_whisper(model_cls, "unused")

        with patch.dict(
            "sys.modules",
            {"faster_whisper": fake_pkg, "faster_whisper.utils": fake_utils},
        ):
            handle = FasterWhisperRuntime().load(
                str(model_dir),
                faster_whisper_model_id="tiny.en",
                compute_type="int8",
                device="cpu",
            )

        assert handle is model_cls.return_value
        fake_utils.download_model.assert_not_called()
        assert model_cls.call_args.args == (str(model_dir),)

    @pytest.mark.parametrize(
        "model_id",
        ["tiny.en", "base.en", "small.en", "medium.en"],
    )
    def test_load_empty_local_dir_triggers_download(self, tmp_path, model_id):
        """An empty staging directory must not skip HuggingFace download."""
        model_dir = tmp_path / "models" / model_id
        model_dir.mkdir(parents=True)
        resolved_snapshot = str(
            tmp_path / "models" / f"models--Systran--faster-whisper-{model_id}"
        )

        model_cls = MagicMock()
        fake_pkg, fake_utils = _patched_faster_whisper(model_cls, resolved_snapshot)

        with patch.dict(
            "sys.modules",
            {"faster_whisper": fake_pkg, "faster_whisper.utils": fake_utils},
        ):
            FasterWhisperRuntime().load(
                str(model_dir),
                faster_whisper_model_id=model_id,
                device="cpu",
                compute_type="int8",
            )

        fake_utils.download_model.assert_called_once_with(
            model_id,
            output_dir=str(model_dir.parent),
            local_files_only=False,
        )
        assert model_cls.call_args.args == (resolved_snapshot,)

    def test_load_ignores_stale_cwd_collision(self, tmp_path, monkeypatch):
        """A stale ``./base.en/`` in CWD must not affect loading — when
        model_path is an absolute directory it is used directly, so CWD
        is never consulted."""
        monkeypatch.chdir(tmp_path)
        stale = tmp_path / "base.en"
        stale.mkdir()
        (stale / "garbage.part").write_bytes(b"interrupted download")

        cache_dir = tmp_path / "cache" / "base.en"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.bin").write_bytes(b"ct2")

        model_cls = MagicMock()
        fake_pkg, fake_utils = _patched_faster_whisper(model_cls, "unused")

        with patch.dict(
            "sys.modules",
            {"faster_whisper": fake_pkg, "faster_whisper.utils": fake_utils},
        ):
            FasterWhisperRuntime().load(
                str(cache_dir),
                faster_whisper_model_id="base.en",
                device="cpu",
                compute_type="int8",
            )

        # Must use the absolute cache_dir path, not the stale relative one
        assert model_cls.call_args.args == (str(cache_dir),)
        fake_utils.download_model.assert_not_called()
        assert stale.exists(), "Must never delete operator/partial data in CWD."

    def test_load_propagates_local_files_only_for_dir(self, tmp_path):
        """``local_files_only`` is forwarded to ``WhisperModel`` even when
        model_path is a pre-staged local directory."""
        cache_dir = tmp_path / "cache" / "tiny.en"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.bin").write_bytes(b"ct2")

        model_cls = MagicMock()
        fake_pkg, fake_utils = _patched_faster_whisper(model_cls, "unused")

        with patch.dict(
            "sys.modules",
            {"faster_whisper": fake_pkg, "faster_whisper.utils": fake_utils},
        ):
            FasterWhisperRuntime().load(
                str(cache_dir),
                faster_whisper_model_id="tiny.en",
                local_files_only=True,
            )

        fake_utils.download_model.assert_not_called()
        assert model_cls.call_args.kwargs["local_files_only"] is True

    def test_load_propagates_local_files_only_for_hf_download(self, tmp_path):
        """``local_files_only`` is forwarded to both ``download_model`` and
        ``WhisperModel`` when model_path is not a local directory."""
        model_path = str(tmp_path / "nonexistent" / "tiny.en")
        resolved = str(tmp_path / "snap")

        model_cls = MagicMock()
        fake_pkg, fake_utils = _patched_faster_whisper(model_cls, resolved)

        with patch.dict(
            "sys.modules",
            {"faster_whisper": fake_pkg, "faster_whisper.utils": fake_utils},
        ):
            FasterWhisperRuntime().load(
                model_path,
                faster_whisper_model_id="tiny.en",
                local_files_only=True,
            )

        fake_utils.download_model.assert_called_once_with(
            "tiny.en",
            output_dir=None,
            local_files_only=True,
        )
        assert model_cls.call_args.kwargs["local_files_only"] is True

    def test_predict_returns_transcription_payload(self, tmp_path):
        audio_file = tmp_path / "chunk.wav"
        audio_file.write_bytes(b"RIFF")

        segment = SimpleNamespace(
            id=0,
            text=" hello",
            start=0.0,
            end=1.25,
            avg_logprob=-0.1,
            compression_ratio=1.1,
            no_speech_prob=0.01,
        )
        info = SimpleNamespace(language="en", duration=1.25, duration_after_vad=1.25)
        handle = MagicMock()
        handle.transcribe.return_value = ([segment], info)

        result = FasterWhisperRuntime().predict(
            handle,
            str(audio_file),
            language="en",
            prompt="context hint",
        )

        handle.transcribe.assert_called_once()
        transcribe_kwargs = handle.transcribe.call_args.kwargs
        assert transcribe_kwargs["language"] == "en"
        assert transcribe_kwargs["initial_prompt"] == "context hint"
        assert isinstance(result, TextResult)
        assert result.text == "hello"
        assert result.raw["text"] == "hello"
        assert result.raw["segments"][0]["text"] == "hello"
        assert result.metadata["segments"][0]["text"] == "hello"
