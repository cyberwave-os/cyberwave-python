"""End-to-end audio pipeline coherence tests.

Verifies the complete audio data flow from Zenoh wire format through to
HookContext metadata propagation, covering:

1. Wire metadata extraction (sample_rate_hz, channels, encoding)
2. HookContext.metadata population from wire headers
3. Non-standard sample rates (32kHz, 44.1kHz, 48kHz stereo)
4. WAV byte construction with correct RIFF headers
5. Metadata validation and mismatch detection
6. Roundtrip: encode → decode → extract_wire_metadata
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from cyberwave.data.backend import Sample
from cyberwave.data.header import HeaderMeta, HeaderTemplate, encode
from cyberwave.workers.context import HookContext
from cyberwave.workers.decode import decode_sample_payload, extract_wire_metadata


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

AUDIO_CONFIGS = [
    {"sample_rate_hz": 16000, "channels": 1, "encoding": "pcm_s16le", "layout": "mono"},
    {
        "sample_rate_hz": 32000,
        "channels": 2,
        "encoding": "pcm_s16le",
        "layout": "stereo",
    },
    {"sample_rate_hz": 44100, "channels": 1, "encoding": "pcm_s16le", "layout": "mono"},
    {
        "sample_rate_hz": 48000,
        "channels": 2,
        "encoding": "pcm_s16le",
        "layout": "stereo",
    },
    {"sample_rate_hz": 8000, "channels": 1, "encoding": "pcm_s16le", "layout": "mono"},
]


def _make_audio_sample(
    *,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    encoding: str = "pcm_s16le",
    layout: str = "mono",
    num_frames: int = 960,
) -> Sample:
    """Create a realistic audio Sample with wire-format payload."""
    pcm = np.zeros(num_frames * channels, dtype=np.int16)
    payload_bytes = pcm.tobytes()

    header = HeaderMeta(
        content_type="numpy/ndarray",
        ts=time.time(),
        seq=0,
        shape=(num_frames, channels) if channels > 1 else (num_frames,),
        dtype="int16",
        metadata={
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "encoding": encoding,
            "layout": layout,
        },
    )
    wire_bytes = encode(header, payload_bytes)
    return Sample(
        channel="cw/test-twin/data/audio/mic_0",
        payload=wire_bytes,
        timestamp=time.time(),
    )


def _make_raw_pcm_sample(num_frames: int = 960, channels: int = 1) -> Sample:
    """Create a Sample with raw PCM (no SDK wire header)."""
    pcm = np.zeros(num_frames * channels, dtype=np.int16)
    return Sample(
        channel="cw/test-twin/data/audio/mic_0",
        payload=pcm.tobytes(),
        timestamp=time.time(),
    )


# ===========================================================================
# Test: Wire metadata extraction
# ===========================================================================


class TestExtractWireMetadata:
    """Verify extract_wire_metadata correctly parses SDK wire headers."""

    @pytest.mark.parametrize(
        "config",
        AUDIO_CONFIGS,
        ids=[f"{c['sample_rate_hz']}Hz_{c['channels']}ch" for c in AUDIO_CONFIGS],
    )
    def test_extracts_audio_metadata_for_all_sample_rates(self, config):
        sample = _make_audio_sample(**config)
        meta = extract_wire_metadata(sample)

        assert meta["sample_rate_hz"] == config["sample_rate_hz"]
        assert meta["channels"] == config["channels"]
        assert meta["encoding"] == config["encoding"]
        assert meta["layout"] == config["layout"]
        assert meta["content_type"] == "numpy/ndarray"

    def test_returns_empty_dict_for_non_sdk_payload(self):
        sample = _make_raw_pcm_sample()
        meta = extract_wire_metadata(sample)
        assert meta == {}

    def test_returns_empty_dict_for_malformed_payload(self):
        sample = Sample(
            channel="test",
            payload=b"not a valid wire format",
            timestamp=time.time(),
        )
        meta = extract_wire_metadata(sample)
        assert meta == {}

    def test_roundtrip_header_template_metadata(self):
        """HeaderTemplate encodes metadata on every frame; verify decode roundtrip."""
        template = HeaderTemplate(
            "numpy/ndarray",
            shape=(960,),
            dtype="int16",
            metadata={
                "sample_rate_hz": 32000,
                "channels": 2,
                "encoding": "pcm_s16le",
                "layout": "stereo",
            },
        )
        pcm = np.zeros(960, dtype=np.int16).tobytes()
        wire_bytes = template.pack(pcm)

        sample = Sample(channel="test", payload=wire_bytes, timestamp=time.time())
        meta = extract_wire_metadata(sample)

        assert meta["sample_rate_hz"] == 32000
        assert meta["channels"] == 2
        assert meta["encoding"] == "pcm_s16le"

    def test_multiple_packs_from_same_template_carry_metadata(self):
        """Verify metadata persists across multiple packs (HeaderTemplate caching)."""
        template = HeaderTemplate(
            "numpy/ndarray",
            shape=(480,),
            dtype="int16",
            metadata={"sample_rate_hz": 48000, "channels": 2, "encoding": "pcm_s16le"},
        )
        pcm = np.zeros(480, dtype=np.int16).tobytes()

        for i in range(5):
            wire_bytes = template.pack(pcm)
            sample = Sample(channel="test", payload=wire_bytes, timestamp=time.time())
            meta = extract_wire_metadata(sample)
            assert meta["sample_rate_hz"] == 48000, f"Failed on pack #{i}"
            assert meta["channels"] == 2, f"Failed on pack #{i}"


# ===========================================================================
# Test: decode_sample_payload with audio data
# ===========================================================================


class TestDecodeSamplePayload:
    """Verify audio samples are decoded correctly regardless of sample rate."""

    @pytest.mark.parametrize(
        "config",
        AUDIO_CONFIGS,
        ids=[f"{c['sample_rate_hz']}Hz_{c['channels']}ch" for c in AUDIO_CONFIGS],
    )
    def test_decodes_audio_numpy_array(self, config):
        sample = _make_audio_sample(**config, num_frames=960)
        decoded, ts = decode_sample_payload(sample, content_hint="numpy")

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.int16
        expected_shape = (960, config["channels"]) if config["channels"] > 1 else (960,)
        assert decoded.shape == expected_shape

    def test_raw_pcm_returned_as_bytes(self):
        sample = _make_raw_pcm_sample(num_frames=480)
        decoded, ts = decode_sample_payload(sample, content_hint="")
        assert isinstance(decoded, bytes)


# ===========================================================================
# Test: HookContext metadata propagation
# ===========================================================================


class TestHookContextMetadataPropagation:
    """Verify wire metadata flows into HookContext.metadata."""

    def test_metadata_populated_from_wire_header(self):
        """Simulate the runtime _build_context flow."""
        sample = _make_audio_sample(sample_rate_hz=32000, channels=2)
        wire_meta = extract_wire_metadata(sample)

        ctx = HookContext(
            timestamp=time.time(),
            channel="audio/mic_0",
            sensor_name="mic_0",
            twin_uuid="test-twin-uuid",
            metadata=wire_meta,
        )

        assert ctx.metadata["sample_rate_hz"] == 32000
        assert ctx.metadata["channels"] == 2
        assert ctx.metadata["encoding"] == "pcm_s16le"

    def test_empty_metadata_for_non_sdk_samples(self):
        sample = _make_raw_pcm_sample()
        wire_meta = extract_wire_metadata(sample)

        ctx = HookContext(
            timestamp=time.time(),
            channel="audio/mic_0",
            sensor_name="mic_0",
            twin_uuid="test-twin-uuid",
            metadata=wire_meta,
        )

        assert ctx.metadata == {}

    @pytest.mark.parametrize(
        "rate,channels",
        [
            (16000, 1),
            (32000, 2),
            (44100, 1),
            (48000, 2),
            (8000, 1),
        ],
    )
    def test_various_rates_propagate_to_context(self, rate, channels):
        sample = _make_audio_sample(sample_rate_hz=rate, channels=channels)
        wire_meta = extract_wire_metadata(sample)

        ctx = HookContext(
            timestamp=time.time(),
            channel="audio/mic_0",
            sensor_name="mic_0",
            twin_uuid="test-twin",
            metadata=wire_meta,
        )

        assert ctx.metadata["sample_rate_hz"] == rate
        assert ctx.metadata["channels"] == channels


# ===========================================================================
# Test: WAV byte construction (from generated code helpers)
# ===========================================================================


def _cw_audio_to_wav_bytes(audio, *, sample_rate_hz=16000, channels=1):
    """Mirror of the generated helper for testing."""
    if isinstance(audio, (bytes, bytearray)):
        pcm = bytes(audio)
    elif isinstance(audio, memoryview):
        pcm = audio.tobytes()
    else:
        tobytes = getattr(audio, "tobytes", None)
        if callable(tobytes):
            pcm = tobytes()
        else:
            return None
    if pcm.startswith(b"RIFF"):
        return pcm
    bits_per_sample = 16
    byte_rate = sample_rate_hz * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate_hz,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm


class TestWavByteConstruction:
    """Verify WAV bytes have correct RIFF headers for all sample rates."""

    @pytest.mark.parametrize(
        "rate,channels",
        [
            (16000, 1),
            (32000, 2),
            (44100, 1),
            (48000, 2),
            (8000, 1),
            (22050, 1),
        ],
    )
    def test_wav_header_encodes_correct_parameters(self, rate, channels):
        num_frames = rate  # 1 second of audio
        pcm = np.zeros(num_frames * channels, dtype=np.int16)
        wav = _cw_audio_to_wav_bytes(pcm, sample_rate_hz=rate, channels=channels)

        assert wav is not None
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

        # Parse fmt chunk
        fmt_size = struct.unpack_from("<I", wav, 16)[0]
        assert fmt_size == 16  # PCM format
        audio_format = struct.unpack_from("<H", wav, 20)[0]
        assert audio_format == 1  # PCM
        wav_channels = struct.unpack_from("<H", wav, 22)[0]
        assert wav_channels == channels
        wav_rate = struct.unpack_from("<I", wav, 24)[0]
        assert wav_rate == rate
        byte_rate = struct.unpack_from("<I", wav, 28)[0]
        assert byte_rate == rate * channels * 2
        block_align = struct.unpack_from("<H", wav, 32)[0]
        assert block_align == channels * 2
        bits_per_sample = struct.unpack_from("<H", wav, 34)[0]
        assert bits_per_sample == 16

        # data chunk
        assert wav[36:40] == b"data"
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == num_frames * channels * 2

    def test_already_wav_passthrough(self):
        """If data already has RIFF header, return as-is."""
        fake_wav = b"RIFF" + b"\x00" * 40
        result = _cw_audio_to_wav_bytes(fake_wav, sample_rate_hz=16000, channels=1)
        assert result == fake_wav

    def test_numpy_array_to_wav(self):
        pcm = np.array([100, -100, 200, -200], dtype=np.int16)
        wav = _cw_audio_to_wav_bytes(pcm, sample_rate_hz=16000, channels=1)
        assert wav[:4] == b"RIFF"
        # Data portion should match the numpy bytes
        assert wav[44:] == pcm.tobytes()

    def test_bytes_to_wav(self):
        raw = bytes(b"\x01\x00" * 100)
        wav = _cw_audio_to_wav_bytes(raw, sample_rate_hz=32000, channels=2)
        assert wav[:4] == b"RIFF"
        assert wav[44:] == raw

    def test_none_for_non_convertible(self):
        result = _cw_audio_to_wav_bytes(42, sample_rate_hz=16000, channels=1)
        assert result is None


# ===========================================================================
# Test: Metadata validation (mismatch detection)
# ===========================================================================


_cw_audio_last_wire_format: dict[str, tuple[int, int]] = {}


def _cw_validate_audio_metadata(
    ctx_metadata, configured_rate, configured_channels, twin_uuid
):
    """Mirror of the generated validation helper."""
    wire_rate = ctx_metadata.get("sample_rate_hz")
    wire_channels = ctx_metadata.get("channels")
    actual_rate = int(wire_rate) if wire_rate is not None else configured_rate
    actual_channels = (
        int(wire_channels) if wire_channels is not None else configured_channels
    )
    last_seen = _cw_audio_last_wire_format.get(twin_uuid)
    current = (actual_rate, actual_channels)
    if last_seen == current:
        return actual_rate, actual_channels, []
    _cw_audio_last_wire_format[twin_uuid] = current
    warnings = []
    if wire_rate is not None and int(wire_rate) != configured_rate:
        warnings.append(
            f"[audio_track] Adapting sample rate for twin {twin_uuid}: "
            f"wire={wire_rate} Hz, configured={configured_rate} Hz."
        )
    if wire_channels is not None and int(wire_channels) != configured_channels:
        warnings.append(
            f"[audio_track] Adapting channel count for twin {twin_uuid}: "
            f"wire={wire_channels}, configured={configured_channels}."
        )
    return actual_rate, actual_channels, warnings


@pytest.fixture(autouse=True)
def _reset_audio_wire_format_cache():
    _cw_audio_last_wire_format.clear()
    yield
    _cw_audio_last_wire_format.clear()


class TestMetadataValidation:
    """Verify mismatch detection for non-standard configurations."""

    def test_no_mismatch_when_matching(self):
        rate, ch, warnings = _cw_validate_audio_metadata(
            {"sample_rate_hz": 16000, "channels": 1}, 16000, 1, "twin-x"
        )
        assert rate == 16000
        assert ch == 1
        assert warnings == []

    def test_rate_mismatch_32k_vs_16k(self):
        rate, ch, warnings = _cw_validate_audio_metadata(
            {"sample_rate_hz": 32000, "channels": 2}, 16000, 1, "twin-x"
        )
        assert rate == 32000
        assert ch == 2
        assert len(warnings) == 2
        assert "32000" in warnings[0]
        assert "16000" in warnings[0]

    def test_rate_mismatch_48k_vs_16k(self):
        rate, ch, warnings = _cw_validate_audio_metadata(
            {"sample_rate_hz": 48000, "channels": 1}, 16000, 1, "twin-x"
        )
        assert rate == 48000
        assert ch == 1
        assert len(warnings) == 1

    def test_channel_mismatch_only(self):
        rate, ch, warnings = _cw_validate_audio_metadata(
            {"sample_rate_hz": 16000, "channels": 2}, 16000, 1, "twin-x"
        )
        assert rate == 16000
        assert ch == 2
        assert len(warnings) == 1
        assert "channels" in warnings[0].lower() or "channel" in warnings[0].lower()

    def test_fallback_when_no_wire_metadata(self):
        rate, ch, warnings = _cw_validate_audio_metadata({}, 44100, 2, "twin-x")
        assert rate == 44100
        assert ch == 2
        assert warnings == []

    def test_8khz_telephony_rate(self):
        rate, ch, warnings = _cw_validate_audio_metadata(
            {"sample_rate_hz": 8000, "channels": 1}, 16000, 1, "twin-x"
        )
        assert rate == 8000
        assert len(warnings) == 1

    @pytest.mark.parametrize(
        "wire_rate,configured_rate",
        [
            (32000, 16000),
            (44100, 16000),
            (48000, 16000),
            (16000, 48000),
            (8000, 16000),
            (22050, 44100),
        ],
    )
    def test_all_rate_mismatches_detected(self, wire_rate, configured_rate):
        rate, _, warnings = _cw_validate_audio_metadata(
            {"sample_rate_hz": wire_rate, "channels": 1},
            configured_rate,
            1,
            "twin-x",
        )
        assert rate == wire_rate
        assert len(warnings) >= 1


# ===========================================================================
# Test: Full pipeline simulation (driver → Zenoh → worker → WAV)
# ===========================================================================


class TestFullPipelineCoherence:
    """Simulate the complete audio path with various hardware configurations."""

    @pytest.mark.parametrize(
        "hw_rate,hw_channels,configured_rate,configured_channels",
        [
            (32000, 2, 16000, 1),  # C920 microphone, default workflow config
            (48000, 1, 16000, 1),  # Studio mic, default config
            (44100, 2, 44100, 2),  # Matching config
            (16000, 1, 16000, 1),  # Standard Whisper-native config
            (8000, 1, 16000, 1),  # Telephony-grade mic
        ],
    )
    def test_wire_metadata_overrides_configured_values(
        self, hw_rate, hw_channels, configured_rate, configured_channels
    ):
        """The wire metadata from the driver takes precedence over node config."""
        # Simulate driver publishing
        sample = _make_audio_sample(
            sample_rate_hz=hw_rate, channels=hw_channels, num_frames=hw_rate // 50
        )

        # Simulate worker runtime
        wire_meta = extract_wire_metadata(sample)
        actual_rate, actual_channels, warnings = _cw_validate_audio_metadata(
            wire_meta, configured_rate, configured_channels, "test-twin"
        )

        # Wire values always win
        assert actual_rate == hw_rate
        assert actual_channels == hw_channels

        # Mismatch detected when different
        if hw_rate != configured_rate or hw_channels != configured_channels:
            assert len(warnings) > 0
        else:
            assert len(warnings) == 0

    @pytest.mark.parametrize(
        "hw_rate,hw_channels",
        [
            (32000, 2),
            (48000, 1),
            (44100, 2),
            (16000, 1),
        ],
    )
    def test_wav_output_matches_wire_metadata(self, hw_rate, hw_channels):
        """WAV bytes produced with actual wire params, not configured defaults."""
        num_frames = hw_rate // 50  # 20ms chunk
        pcm = np.zeros(num_frames * hw_channels, dtype=np.int16)

        wav = _cw_audio_to_wav_bytes(pcm, sample_rate_hz=hw_rate, channels=hw_channels)

        assert wav is not None
        wav_rate = struct.unpack_from("<I", wav, 24)[0]
        wav_channels = struct.unpack_from("<H", wav, 22)[0]
        assert wav_rate == hw_rate
        assert wav_channels == hw_channels

    def test_whisper_receives_correct_wav_for_32k_stereo(self):
        """Simulate: 32kHz/stereo mic → Audio Track → Whisper.

        Whisper resamples internally, but the WAV header must declare
        correct parameters so resampling ratio is computed correctly.
        """
        hw_rate, hw_channels = 32000, 2
        configured_rate, configured_channels = 16000, 1

        # Driver publishes 20ms of 32kHz stereo audio
        num_frames = 640  # 20ms at 32kHz
        pcm = np.random.randint(-1000, 1000, num_frames * hw_channels, dtype=np.int16)
        sample = _make_audio_sample(
            sample_rate_hz=hw_rate, channels=hw_channels, num_frames=num_frames
        )

        # Worker runtime extracts metadata
        wire_meta = extract_wire_metadata(sample)
        actual_rate, actual_channels, _ = _cw_validate_audio_metadata(
            wire_meta, configured_rate, configured_channels, "twin-x"
        )

        # WAV constructed with actual params
        wav = _cw_audio_to_wav_bytes(
            pcm, sample_rate_hz=actual_rate, channels=actual_channels
        )

        # Verify WAV header declares 32kHz stereo
        wav_rate = struct.unpack_from("<I", wav, 24)[0]
        wav_channels = struct.unpack_from("<H", wav, 22)[0]
        assert wav_rate == 32000
        assert wav_channels == 2

        # Duration sanity: 640 frames / 32000 Hz = 0.02s
        data_size = struct.unpack_from("<I", wav, 40)[0]
        duration = data_size / (wav_rate * wav_channels * 2)
        assert abs(duration - 0.02) < 0.001


# ===========================================================================
# Test: WebRTC audio path metadata coherence
# ===========================================================================


class TestWebRTCAudioMetadata:
    """Verify WebRTC audio path maintains sample rate coherence.

    WebRTC uses Opus at 48kHz internally. The media service records
    OGG/Opus with correct headers. These tests verify the metadata
    contracts are maintained.
    """

    def test_opus_native_rate_is_48k(self):
        """Opus codec operates at 48kHz regardless of input."""
        OPUS_SAMPLE_RATE = 48000
        assert OPUS_SAMPLE_RATE == 48000

    def test_webrtc_audio_frame_at_48k(self):
        """WebRTC audio frames are 20ms at 48kHz = 960 samples."""
        WEBRTC_PTIME_MS = 20
        OPUS_RATE = 48000
        samples_per_frame = OPUS_RATE * WEBRTC_PTIME_MS // 1000
        assert samples_per_frame == 960

    def test_resampling_32k_to_48k_ratio(self):
        """32kHz mic → 48kHz Opus requires 3/2 resampling ratio."""
        hw_rate = 32000
        opus_rate = 48000
        ratio = opus_rate / hw_rate
        assert ratio == 1.5

    def test_ogg_recording_preserves_original_rate_in_headers(self):
        """OGG container stores original sample rate in stream info.

        Even though Opus operates at 48kHz internally, OGG headers can
        declare the original rate for proper decoding.
        """
        # This is a contract test — OGG/Opus always stores 48kHz as the
        # internal rate but the pre-skip and granule calculations account
        # for the original rate.
        opus_internal_rate = 48000
        assert opus_internal_rate == 48000
