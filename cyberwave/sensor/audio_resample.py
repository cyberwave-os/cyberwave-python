"""Inbound/outbound PCM resampling for sensor streaming (speaker playback path)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _layout_for_channels(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    raise ValueError("Only mono and stereo audio are supported")


def _as_int16_2d(chunk: np.ndarray, channels: int) -> np.ndarray:
    audio = np.asarray(chunk, dtype=np.int16)
    if audio.ndim == 1:
        if channels <= 0 or audio.size % channels != 0:
            raise ValueError("Flat audio chunk size is not divisible by channel count")
        audio = audio.reshape(-1, channels)
    if audio.ndim != 2:
        raise ValueError("Audio chunk must be a 1D or 2D int16 array")
    if audio.shape[1] != channels:
        raise ValueError(
            f"Audio chunk has {audio.shape[1]} channels, expected {channels}"
        )
    return np.ascontiguousarray(audio, dtype=np.int16)


def _convert_channels(chunk: np.ndarray, output_channels: int) -> np.ndarray:
    input_channels = int(chunk.shape[1])
    if input_channels == output_channels:
        return chunk
    if input_channels == 1 and output_channels == 2:
        return np.repeat(chunk, 2, axis=1)
    if input_channels == 2 and output_channels == 1:
        mixed = np.rint(chunk.astype(np.float32).mean(axis=1))
        return np.clip(mixed, -32768, 32767).astype(np.int16).reshape(-1, 1)
    raise ValueError(
        f"Unsupported channel conversion: {input_channels} -> {output_channels}"
    )


def _linear_resample(
    chunk: np.ndarray,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> np.ndarray:
    if input_sample_rate == output_sample_rate:
        return chunk
    input_frames = int(chunk.shape[0])
    if input_frames == 0:
        return chunk
    output_frames = max(
        1,
        int(round(input_frames * output_sample_rate / input_sample_rate)),
    )
    if input_frames == 1:
        return np.repeat(chunk, output_frames, axis=0)

    old_positions = np.linspace(0, input_frames - 1, num=input_frames)
    new_positions = np.linspace(0, input_frames - 1, num=output_frames)
    channels = []
    for channel_idx in range(chunk.shape[1]):
        values = np.interp(new_positions, old_positions, chunk[:, channel_idx])
        channels.append(np.rint(values))
    resampled = np.stack(channels, axis=1)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def _frame_to_interleaved_int16(frame: Any, channels: int) -> np.ndarray:
    array = np.asarray(frame.to_ndarray(), dtype=np.int16)
    if array.ndim == 1:
        if channels == 1:
            return array.reshape(-1, 1)
        return array.reshape(-1, channels)
    if array.ndim == 2:
        if array.shape[0] == channels and array.shape[1] != channels:
            return np.ascontiguousarray(array.T, dtype=np.int16)
        if array.shape[1] == channels:
            return np.ascontiguousarray(array, dtype=np.int16)
        if array.shape[0] == 1:
            return array.reshape(-1, channels)
    raise ValueError(
        f"Unexpected PyAV audio ndarray shape {array.shape!r} for {channels} channel(s)"
    )


class AudioResampler:
    """Convert int16 PCM chunks between sample rates and channel layouts."""

    def __init__(
        self,
        *,
        input_sample_rate: int,
        input_channels: int,
        output_sample_rate: int,
        output_channels: int,
        enabled: bool = True,
    ) -> None:
        self.input_sample_rate = int(input_sample_rate)
        self.input_channels = int(input_channels)
        self.output_sample_rate = int(output_sample_rate)
        self.output_channels = int(output_channels)
        self.enabled = bool(enabled)
        self._needs_conversion = self.enabled and (
            self.input_sample_rate != self.output_sample_rate
            or self.input_channels != self.output_channels
        )
        self._warned_pyav_fallback = False
        self._resampler: Any | None = None

        _layout_for_channels(self.input_channels)
        _layout_for_channels(self.output_channels)

        if self._needs_conversion:
            self._init_pyav_resampler()

    @property
    def needs_conversion(self) -> bool:
        return self._needs_conversion

    def _init_pyav_resampler(self) -> None:
        try:
            from av.audio.resampler import AudioResampler as PyAVAudioResampler

            self._resampler = PyAVAudioResampler(
                format="s16",
                layout=_layout_for_channels(self.output_channels),
                rate=self.output_sample_rate,
            )
        except Exception as exc:  # pragma: no cover - optional native libs
            self._resampler = None
            logger.warning(
                "PyAV audio resampler unavailable; falling back to linear resampling: %s",
                exc,
            )

    def resample(self, chunk: np.ndarray) -> np.ndarray:
        audio = _as_int16_2d(chunk, self.input_channels)
        if not self._needs_conversion:
            return audio

        if self._resampler is not None:
            try:
                pyav_out = self._resample_with_pyav(audio)
                if pyav_out.size > 0:
                    return pyav_out
                if not self._warned_pyav_fallback:
                    logger.warning(
                        "PyAV resampler returned no output for a non-empty chunk; "
                        "using linear fallback"
                    )
                    self._warned_pyav_fallback = True
            except Exception as exc:
                if not self._warned_pyav_fallback:
                    logger.warning(
                        "PyAV audio resampling failed; using linear fallback: %s",
                        exc,
                    )
                    self._warned_pyav_fallback = True

        converted = _convert_channels(audio, self.output_channels)
        return _linear_resample(
            converted,
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.output_sample_rate,
        )

    def _resample_with_pyav(self, audio: np.ndarray) -> np.ndarray:
        from av import AudioFrame

        frame = AudioFrame.from_ndarray(
            audio.reshape(1, -1),
            format="s16",
            layout=_layout_for_channels(self.input_channels),
        )
        frame.sample_rate = self.input_sample_rate
        frames = self._resampler.resample(frame)
        if not frames:
            return np.empty((0, self.output_channels), dtype=np.int16)

        chunks = [
            _frame_to_interleaved_int16(out_frame, self.output_channels)
            for out_frame in frames
        ]
        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.int16)


class InboundAudioAdapter:
    """Lazily rebuilds :class:`AudioResampler` when inbound format changes."""

    def __init__(
        self,
        *,
        output_sample_rate: int,
        output_channels: int,
        enabled: bool = True,
    ) -> None:
        self._output_sample_rate = int(output_sample_rate)
        self._output_channels = int(output_channels)
        self._enabled = bool(enabled)
        self._resampler: AudioResampler | None = None
        self._input_sample_rate: int | None = None
        self._input_channels: int | None = None

    def convert(
        self,
        chunk: np.ndarray,
        *,
        input_sample_rate: int,
        input_channels: int,
    ) -> np.ndarray:
        in_rate = int(input_sample_rate)
        in_channels = int(input_channels)
        if (
            self._resampler is None
            or self._input_sample_rate != in_rate
            or self._input_channels != in_channels
        ):
            self._resampler = AudioResampler(
                input_sample_rate=in_rate,
                input_channels=in_channels,
                output_sample_rate=self._output_sample_rate,
                output_channels=self._output_channels,
                enabled=self._enabled,
            )
            self._input_sample_rate = in_rate
            self._input_channels = in_channels
        assert self._resampler is not None
        return self._resampler.resample(chunk)


def infer_webrtc_frame_channels(frame: Any) -> int:
    """Best-effort channel count from an aiortc/PyAV audio frame."""
    layout = getattr(frame, "layout", None)
    if layout is not None:
        channels = getattr(layout, "channels", None)
        if channels:
            return len(channels)
        name = str(getattr(layout, "name", "")).lower()
        if name == "mono":
            return 1
        if name == "stereo":
            return 2
    array = np.asarray(frame.to_ndarray())
    if array.ndim == 1:
        return 1
    if array.shape[0] in {1, 2} and array.shape[0] != array.shape[1]:
        return int(array.shape[0])
    return int(array.shape[1])


__all__ = [
    "AudioResampler",
    "InboundAudioAdapter",
    "infer_webrtc_frame_channels",
]
