"""Tests for cyberwave.workers.decode.decode_sample — see CYB-3248.

``decode_sample`` replaced a ``decode_sample_payload`` + ``extract_wire_metadata``
pair in the worker dispatch loop.  Each of those independently called
:func:`cyberwave.data.header.decode`, which slices the payload out of the wire
buffer — so a raw BGR frame was copied twice per sample and one copy thrown
away.  These tests pin the single-decode behaviour and the equivalence with the
pair it replaced.
"""

import json

import numpy as np
import pytest

from cyberwave.data.backend import Sample
from cyberwave.data.header import (
    CONTENT_TYPE_BYTES,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_NUMPY,
    HeaderMeta,
    encode,
)
from cyberwave.workers import decode as decode_mod
from cyberwave.workers.decode import (
    decode_sample,
    decode_sample_payload,
    extract_wire_metadata,
)


def _sdk_sample(payload: bytes, **header_kwargs) -> Sample:
    header = HeaderMeta(ts=1234.5, seq=7, **header_kwargs)
    return Sample(channel="ch", payload=encode(header, payload), timestamp=99.0)


def _frame_sample(w: int = 32, h: int = 24) -> Sample:
    frame = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    return _sdk_sample(
        frame.tobytes(),
        content_type=CONTENT_TYPE_NUMPY,
        shape=frame.shape,
        dtype=str(frame.dtype),
        metadata={"encoding": "bgr8", "source": "camera"},
    )


def _audio_sample() -> Sample:
    pcm = np.linspace(-1.0, 1.0, 480, dtype=np.float32)
    return _sdk_sample(
        pcm.tobytes(),
        content_type=CONTENT_TYPE_NUMPY,
        shape=pcm.shape,
        dtype=str(pcm.dtype),
        metadata={"sample_rate_hz": 16000, "channels": 1, "layout": "mono"},
    )


def _shape_mismatch_sample() -> Sample:
    """Header declares 300 bytes of uint8; the payload carries 100."""
    return _sdk_sample(
        b"\x00" * 100,
        content_type=CONTENT_TYPE_NUMPY,
        shape=(10, 10, 3),
        dtype="uint8",
    )


def _cases() -> list[tuple[str, Sample, str]]:
    """(name, sample, content_hint) covering every wire format the loop sees."""
    numpy_no_shape = _sdk_sample(
        b"\x01\x02\x03\x04", content_type=CONTENT_TYPE_NUMPY
    )
    return [
        ("sdk numpy frame, hinted", _frame_sample(), "numpy"),
        ("sdk numpy frame, unhinted", _frame_sample(), ""),
        ("sdk numpy audio, hinted", _audio_sample(), "numpy"),
        ("sdk numpy audio, unhinted", _audio_sample(), ""),
        ("sdk numpy without shape", numpy_no_shape, "numpy"),
        (
            "sdk json",
            _sdk_sample(
                json.dumps({"a": 1, "b": [2, 3]}).encode(),
                content_type=CONTENT_TYPE_JSON,
            ),
            "",
        ),
        (
            "sdk bytes",
            _sdk_sample(b"\xde\xad\xbe\xef", content_type=CONTENT_TYPE_BYTES),
            "",
        ),
        (
            "bare json",
            Sample(channel="ch", payload=b'{"alert":"x"}', timestamp=42.0),
            "",
        ),
        ("garbage", Sample(channel="ch", payload=b"\x00\x01\x02", timestamp=5.0), ""),
        (
            "truncated header",
            Sample(channel="ch", payload=b"\xff\xff\x00\x00abc", timestamp=6.0),
            "",
        ),
        # Valid header, undecodable payload.  The cases above either decode
        # cleanly or fail at the framing; these fail *after* decode() has
        # succeeded, which is the only class where a narrowed try/except
        # would let the exception escape into dispatch_loop.
        *[
            (f"payload/shape disagreement, {label}", _shape_mismatch_sample(), hint)
            for label, hint in (("hinted", "numpy"), ("unhinted", ""))
        ],
        (
            "numpy header with unknown dtype",
            _sdk_sample(
                b"\x00" * 12,
                content_type=CONTENT_TYPE_NUMPY,
                shape=(2, 2, 3),
                dtype="not-a-dtype",
            ),
            "numpy",
        ),
        (
            "json header with non-JSON body",
            _sdk_sample(b"\xde\xad\xbe\xef", content_type=CONTENT_TYPE_JSON),
            "",
        ),
    ]


@pytest.mark.parametrize("name,sample,hint", _cases(), ids=[c[0] for c in _cases()])
def test_matches_the_two_call_pair_it_replaced(name, sample, hint):
    want_data, want_ts = decode_sample_payload(sample, content_hint=hint)
    want_meta = extract_wire_metadata(sample)

    got_data, got_ts, got_meta = decode_sample(sample, content_hint=hint)

    if isinstance(want_data, np.ndarray):
        assert np.array_equal(got_data, want_data)
        assert got_data.dtype == want_data.dtype
        assert got_data.shape == want_data.shape
    else:
        assert got_data == want_data
    assert got_ts == want_ts
    assert got_meta == want_meta


def test_decodes_the_wire_buffer_exactly_once(monkeypatch):
    """The whole point: one decode() per sample, not two.

    decode() copies the payload out of the wire buffer, so a second call
    duplicates a multi-megabyte frame for header fields alone.
    """
    calls = {"n": 0}
    real_decode = decode_mod.decode

    def counting_decode(raw):
        calls["n"] += 1
        return real_decode(raw)

    monkeypatch.setattr(decode_mod, "decode", counting_decode)

    decode_sample(_frame_sample(), content_hint="numpy")

    assert calls["n"] == 1


def test_payload_level_failure_falls_back_instead_of_raising():
    """dispatch_loop calls decode_sample outside the try guarding the hook.

    An escaping exception there kills the hook's dispatch thread for the life
    of the worker — no restart, no _publish_hook_error_alert, frames silently
    stop.  So a header that decodes fine but whose payload does not must
    degrade to raw bytes, exactly as decode_sample_payload does, and must
    still return the header metadata it already parsed.
    """
    sample = _shape_mismatch_sample()

    data, ts, meta = decode_sample(sample, content_hint="numpy")

    assert data == sample.payload
    assert ts == sample.timestamp
    assert meta["content_type"] == CONTENT_TYPE_NUMPY
    assert meta["dtype"] == "uint8"


def test_resize_failure_falls_back_instead_of_raising(monkeypatch):
    """_maybe_resize runs on every numpy channel, not just camera frames.

    With CYBERWAVE_WORKER_INPUT_RESOLUTION set, cv2.resize rejects some
    perfectly well-formed arrays (2-D bool/int8/uint32/int64), so this is
    reachable without any malformed data on the wire.
    """
    def boom(_data):
        raise RuntimeError("cv2.resize rejected this array")

    monkeypatch.setattr(decode_mod, "_maybe_resize", boom)

    data, ts, meta = decode_sample(_frame_sample(), content_hint="numpy")

    assert isinstance(data, bytes)
    assert meta["encoding"] == "bgr8"


def test_returns_header_metadata_for_a_frame():
    _, ts, meta = decode_sample(_frame_sample(), content_hint="numpy")

    assert ts == 1234.5
    assert meta["encoding"] == "bgr8"
    assert meta["source"] == "camera"
    assert meta["content_type"] == CONTENT_TYPE_NUMPY
    assert meta["dtype"] == "uint8"


def test_returns_empty_metadata_for_non_sdk_payloads():
    _, _, meta = decode_sample(
        Sample(channel="ch", payload=b'{"alert":"x"}', timestamp=42.0)
    )

    assert meta == {}


def test_decodes_raw_jpeg_from_native_drivers():
    cv2 = pytest.importorskip("cv2")

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:12, 4:12] = 255
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    sample = Sample(channel="ch", payload=buf.tobytes(), timestamp=8.0)

    data, ts, meta = decode_sample(sample, content_hint="numpy")

    assert isinstance(data, np.ndarray)
    assert data.shape == (16, 16, 3)
    assert ts == 8.0
    assert meta == {}
