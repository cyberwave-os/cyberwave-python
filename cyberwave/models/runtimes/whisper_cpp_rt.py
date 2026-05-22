"""whisper.cpp runtime backend for edge speech-to-text models."""

from __future__ import annotations

import os
import tempfile
import wave
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult


class WhisperCppRuntime(ModelRuntime):
    """Run local Whisper transcription through ``pywhispercpp``."""

    name = "whisper_cpp"

    def is_available(self) -> bool:
        try:
            from pywhispercpp.model import Model  # noqa: F401
        except Exception:
            return False
        return True

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from pywhispercpp.model import Model

        model_kwargs = dict(kwargs)
        if device and device != "cpu":
            model_kwargs.setdefault("gpu", True)
        return Model(model_path, **model_kwargs)

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        language = kwargs.get("language")
        translate = bool(kwargs.get("translate", False))
        sample_rate_hz = int(kwargs.get("sample_rate_hz") or 16000)
        channels = int(kwargs.get("channels") or 1)

        audio_path, should_cleanup = _audio_input_to_path(
            input_data,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        try:
            transcribe_kwargs: dict[str, Any] = {}
            if language:
                transcribe_kwargs["language"] = language
            if translate:
                transcribe_kwargs["translate"] = True
            segments = model_handle.transcribe(audio_path, **transcribe_kwargs)
        finally:
            if should_cleanup:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

        payload = _segments_to_payload(segments, language=language)
        return PredictionResult(
            raw=payload,
            metadata={
                "text": payload["text"],
                "segments": payload["segments"],
                "language": payload.get("language"),
            },
        )


def _audio_input_to_path(
    input_data: Any,
    *,
    sample_rate_hz: int,
    channels: int,
) -> tuple[str, bool]:
    if isinstance(input_data, str | os.PathLike):
        return str(input_data), False

    if isinstance(input_data, bytes | bytearray | memoryview):
        raw = bytes(input_data)
        suffix = ".wav" if raw.startswith(b"RIFF") else ".raw.wav"
        return _write_audio_bytes(
            raw, suffix=suffix, sample_rate_hz=sample_rate_hz, channels=channels
        ), True

    if isinstance(input_data, np.ndarray):
        pcm = _numpy_to_pcm16(input_data)
        return _write_wav_bytes(
            pcm,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        ), True

    raise TypeError(
        "WhisperCppRuntime expects a WAV path, WAV/PCM bytes, or a numpy audio array"
    )


def _write_audio_bytes(
    data: bytes,
    *,
    suffix: str,
    sample_rate_hz: int,
    channels: int,
) -> str:
    if suffix == ".wav":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with tmp:
            tmp.write(data)
        return tmp.name
    return _write_wav_bytes(data, sample_rate_hz=sample_rate_hz, channels=channels)


def _write_wav_bytes(data: bytes, *, sample_rate_hz: int, channels: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    with wave.open(tmp.name, "wb") as wav:
        wav.setnchannels(max(1, channels))
        wav.setsampwidth(2)
        wav.setframerate(max(1, sample_rate_hz))
        wav.writeframes(data)
    return tmp.name


def _numpy_to_pcm16(audio: np.ndarray) -> bytes:
    array = np.asarray(audio)
    if array.dtype.kind == "f":
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype(np.int16)
    elif array.dtype != np.int16:
        array = array.astype(np.int16)
    return np.ascontiguousarray(array).tobytes()


def _segments_to_payload(segments: Any, *, language: str | None) -> dict[str, Any]:
    if isinstance(segments, str):
        text = segments.strip()
        return {"text": text, "segments": [], "language": language}

    normalized_segments = [_segment_to_dict(segment) for segment in segments or []]
    text = " ".join(
        segment["text"] for segment in normalized_segments if segment["text"]
    ).strip()
    return {
        "text": text,
        "segments": normalized_segments,
        "language": language,
    }


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    text = str(getattr(segment, "text", "") or "").strip()
    start = _coerce_seconds(getattr(segment, "start", getattr(segment, "t0", None)))
    end = _coerce_seconds(getattr(segment, "end", getattr(segment, "t1", None)))
    return {"text": text, "start": start, "end": end}


def _coerce_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds > 100.0:
        return seconds / 1000.0
    return seconds
