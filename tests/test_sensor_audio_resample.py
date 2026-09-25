"""Tests for speaker inbound PCM resampling."""

from __future__ import annotations

import numpy as np

from cyberwave.sensor.audio_resample import AudioResampler, InboundAudioAdapter


def test_inbound_adapter_rebuilds_on_format_change() -> None:
    adapter = InboundAudioAdapter(
        output_sample_rate=48_000,
        output_channels=2,
    )
    chunk = np.zeros((640, 2), dtype=np.int16)
    out_a = adapter.convert(
        chunk,
        input_sample_rate=32_000,
        input_channels=2,
    )
    mono_chunk = np.zeros((882, 1), dtype=np.int16)
    out_b = adapter.convert(
        mono_chunk,
        input_sample_rate=44_100,
        input_channels=1,
    )
    assert out_a.shape[1] == 2
    assert 900 <= out_a.shape[0] <= 980
    assert out_b.shape[1] == 2
    assert 900 <= out_b.shape[0] <= 980


def test_audio_resampler_mono_to_stereo_output() -> None:
    resampler = AudioResampler(
        input_sample_rate=48_000,
        input_channels=1,
        output_sample_rate=48_000,
        output_channels=2,
        enabled=True,
    )
    resampler._resampler = None
    chunk = np.array([[100], [-100]], dtype=np.int16)
    out = resampler.resample(chunk)
    assert out.tolist() == [[100, 100], [-100, -100]]
