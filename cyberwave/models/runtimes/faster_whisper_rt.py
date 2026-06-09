"""faster-whisper runtime backend for edge speech-to-text models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.runtimes.whisper_cpp_rt import _audio_input_to_path
from cyberwave.models.types import PredictionResult, TextResult


def _resolve_device_compute(
    device: str | None,
    compute_type: str | None,
) -> tuple[str, str]:
    resolved_device = (device or "cpu").strip().lower()
    if resolved_device == "auto":
        try:
            import ctranslate2

            resolved_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            resolved_device = "cpu"

    resolved_compute = (compute_type or "auto").strip().lower()
    if resolved_compute == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    return resolved_device, resolved_compute


def _segment_to_payload(segment: Any) -> dict[str, Any]:
    text = str(getattr(segment, "text", "") or "").strip()
    return {
        "id": getattr(segment, "id", None),
        "text": text,
        "start": float(getattr(segment, "start", 0.0) or 0.0),
        "end": float(getattr(segment, "end", 0.0) or 0.0),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
    }


class FasterWhisperRuntime(ModelRuntime):
    """Run local Whisper transcription through ``faster-whisper`` (CTranslate2).

    A single ``WhisperModel`` handle must not run ``transcribe()`` concurrently.
    :class:`~cyberwave.models.loaded_model.LoadedModel` serializes
    ``predict()`` and ``warm_up()`` with an inference lock.
    """

    name = "faster_whisper"

    def is_available(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
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
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model

        model_id = (
            kwargs.get("faster_whisper_model_id")
            or kwargs.get("model_id")
            or Path(model_path).name
            or model_path
        )
        resolved_device, resolved_compute = _resolve_device_compute(
            device,
            kwargs.get("compute_type"),
        )
        local_files_only = bool(kwargs.get("local_files_only", False))

        path = Path(model_path)
        if path.is_dir():
            # Model is pre-staged or already cached locally — skip HuggingFace
            # entirely so edge devices without internet don't hang on startup.
            resolved_model_path = str(path)
        else:
            download_root = kwargs.get("download_root") or os.getenv("WHISPER_CACHE_DIR") or os.getenv("HF_HOME")
            # Pre-resolve to an absolute snapshot path so ``WhisperModel`` never
            # runs its ``os.path.isdir(model_id)`` check against CWD — a stale
            # ``./base.en/`` from a prior interrupted run would otherwise be
            # mistaken for a local snapshot and crash ctranslate2.
            resolved_model_path = download_model(
                str(model_id),
                output_dir=download_root,
                local_files_only=local_files_only,
            )

        return WhisperModel(
            resolved_model_path,
            device=resolved_device,
            compute_type=resolved_compute,
            local_files_only=local_files_only,
        )

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        del confidence, classes  # STT path does not use detection knobs.

        language = kwargs.get("language")
        if language in (None, "", "auto"):
            language = None

        task = kwargs.get("task") or "transcribe"
        translate = task == "translate"
        sample_rate_hz = int(kwargs.get("sample_rate_hz") or 16000)
        channels = int(kwargs.get("channels") or 1)
        beam_size = int(kwargs.get("beam_size") or 5)
        vad_filter = bool(kwargs.get("vad_filter", False))
        initial_prompt = kwargs.get("initial_prompt") or kwargs.get("prompt")

        audio_path, should_cleanup = _audio_input_to_path(
            input_data,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        try:
            segments_iter, info = model_handle.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
                word_timestamps=bool(kwargs.get("word_timestamps", False)),
            )
            segments = [_segment_to_payload(segment) for segment in segments_iter]
        finally:
            if should_cleanup:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

        text = "".join(segment["text"] for segment in segments).strip()
        payload = {
            "text": text,
            "segments": segments,
            "language": getattr(info, "language", language),
            "duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
        }
        if translate:
            payload["task"] = "translate"
        return TextResult(
            text=text,
            raw=payload,
            metadata={
                "segments": segments,
                "language": payload.get("language"),
                "duration": payload.get("duration"),
                "duration_after_vad": payload.get("duration_after_vad"),
                "task": payload.get("task"),
            },
        )
