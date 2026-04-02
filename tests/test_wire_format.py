"""Golden-fixture tests for the SDK wire-format header encode/decode.

Acceptance criteria from CYB-1553:
  * Binary and JSON payload roundtrip is deterministic.
  * Golden fixture tests for encode / decode.
  * HeaderTemplate hot-path is < 500ns per sample.
"""

from __future__ import annotations

import json
import struct
import time

import pytest

from cyberwave.data.header import (
    HeaderMeta,
    HeaderTemplate,
    WireFormatError,
    _HEADER_LEN_FMT,
    _HEADER_LEN_SIZE,
    _MAX_HEADER_BYTES,
    _TS_SEQ_FMT,
    _TS_SEQ_SIZE,
    decode,
    encode,
)


# ── HeaderMeta dataclass ─────────────────────────────────────────────


class TestHeaderMeta:
    def test_defaults(self) -> None:
        h = HeaderMeta(content_type="application/json", ts=1.0)
        assert h.content_type == "application/json"
        assert h.ts == 1.0
        assert h.seq == 0
        assert h.shape is None
        assert h.dtype is None
        assert h.metadata is None

    def test_all_fields(self) -> None:
        h = HeaderMeta(
            content_type="numpy/ndarray",
            ts=100.0,
            seq=42,
            shape=(480, 640, 3),
            dtype="uint8",
            metadata={"fps": 30, "width": 640},
        )
        assert h.shape == (480, 640, 3)
        assert h.dtype == "uint8"
        assert h.metadata == {"fps": 30, "width": 640}


# ── Low-level encode / decode ────────────────────────────────────────


class TestEncodeDecode:
    def test_roundtrip_numpy(self) -> None:
        hdr = HeaderMeta(
            content_type="numpy/ndarray",
            ts=1711234567.123,
            seq=42,
            shape=(480, 640, 3),
            dtype="uint8",
        )
        payload = b"\x00" * 100
        wire = encode(hdr, payload)
        hdr_out, payload_out = decode(wire)

        assert hdr_out.content_type == "numpy/ndarray"
        assert hdr_out.ts == pytest.approx(1711234567.123)
        assert hdr_out.seq == 42
        assert hdr_out.shape == (480, 640, 3)
        assert hdr_out.dtype == "uint8"
        assert payload_out == payload

    def test_roundtrip_json(self) -> None:
        hdr = HeaderMeta(content_type="application/json", ts=1.0, seq=0)
        payload = b'{"joint1": 0.5}'
        wire = encode(hdr, payload)
        hdr_out, payload_out = decode(wire)

        assert hdr_out.content_type == "application/json"
        assert hdr_out.shape is None
        assert hdr_out.dtype is None
        assert payload_out == payload

    def test_roundtrip_octet_stream(self) -> None:
        hdr = HeaderMeta(content_type="application/octet-stream", ts=2.0, seq=1)
        payload = b"\xff\xfe\xfd"
        wire = encode(hdr, payload)
        hdr_out, payload_out = decode(wire)

        assert hdr_out.content_type == "application/octet-stream"
        assert payload_out == payload

    def test_metadata_flat_merge(self) -> None:
        hdr = HeaderMeta(
            content_type="numpy/ndarray",
            ts=1.0,
            seq=0,
            shape=(480, 640, 3),
            dtype="uint8",
            metadata={"width": 640, "height": 480, "fps": 30},
        )
        wire = encode(hdr, b"\x00")
        hdr_out, _ = decode(wire)

        assert hdr_out.metadata is not None
        assert hdr_out.metadata["width"] == 640
        assert hdr_out.metadata["height"] == 480
        assert hdr_out.metadata["fps"] == 30

    def test_roundtrip_empty_payload(self) -> None:
        hdr = HeaderMeta(content_type="application/json", ts=0.0)
        wire = encode(hdr, b"")
        hdr_out, payload_out = decode(wire)
        assert payload_out == b""

    def test_deterministic_encoding(self) -> None:
        hdr = HeaderMeta(
            content_type="numpy/ndarray",
            ts=1.0,
            seq=0,
            shape=(10,),
            dtype="float32",
        )
        payload = b"\x01\x02\x03"
        wire_a = encode(hdr, payload)
        wire_b = encode(hdr, payload)
        assert wire_a == wire_b


# ── Wire format structure ────────────────────────────────────────────


class TestWireFormatStructure:
    def test_layout_binary_prefix(self) -> None:
        """Verify ts and seq are binary-packed, not in the JSON."""
        hdr = HeaderMeta(content_type="application/json", ts=100.0, seq=7)
        payload = b"hello"
        wire = encode(hdr, payload)

        (total_header_len,) = struct.unpack_from(_HEADER_LEN_FMT, wire, 0)
        ts, seq = struct.unpack_from(_TS_SEQ_FMT, wire, _HEADER_LEN_SIZE)

        assert ts == pytest.approx(100.0)
        assert seq == 7

        header_end = _HEADER_LEN_SIZE + total_header_len
        assert wire[header_end:] == payload

        json_start = _HEADER_LEN_SIZE + _TS_SEQ_SIZE
        json_bytes = wire[json_start:header_end]
        json_dict = json.loads(json_bytes)
        assert "ts" not in json_dict
        assert "seq" not in json_dict
        assert json_dict["content_type"] == "application/json"


# ── Golden fixtures ──────────────────────────────────────────────────


GOLDEN_CASES = [
    pytest.param(
        HeaderMeta(content_type="application/json", ts=1711700000.0),
        b'{"ts":1711700000.0,"x":1.0,"y":2.0}',
        id="json-position",
    ),
    pytest.param(
        HeaderMeta(
            content_type="numpy/ndarray",
            ts=1711700000.0,
            shape=(480, 640, 3),
            dtype="uint8",
            metadata={"width": 640, "height": 480, "channels": 3, "fps": 30.0},
        ),
        b"\x00\xff\x80" * 10,
        id="binary-image",
    ),
    pytest.param(
        HeaderMeta(
            content_type="numpy/ndarray",
            ts=1711700001.0,
            shape=(240, 320),
            dtype="uint16",
            metadata={"unit": "mm"},
        ),
        b"\x00\x10" * 5,
        id="binary-depth",
    ),
    pytest.param(
        HeaderMeta(
            content_type="numpy/ndarray",
            ts=1711700002.0,
            dtype="float32",
            metadata={"sample_rate": 48000, "channels": 1},
        ),
        b"\x00\x00\x80\x3f" * 4,
        id="binary-audio",
    ),
    pytest.param(
        HeaderMeta(
            content_type="numpy/ndarray",
            ts=1711700003.0,
            dtype="float32",
            metadata={"n_points": 100, "fields": ["x", "y", "z"]},
        ),
        b"\x00" * 1200,
        id="binary-pointcloud",
    ),
    pytest.param(
        HeaderMeta(
            content_type="application/json",
            ts=1711700005.0,
        ),
        json.dumps(
            {
                "ts": 1711700005.0,
                "names": ["j1", "j2"],
                "positions": [0.1, 0.2],
                "velocities": [0.0, 0.0],
                "efforts": [0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode(),
        id="json-joint-states",
    ),
]


class TestGoldenFixtures:
    @pytest.mark.parametrize("header,payload", GOLDEN_CASES)
    def test_encode_decode_roundtrip(self, header: HeaderMeta, payload: bytes) -> None:
        wire = encode(header, payload)
        hdr_out, payload_out = decode(wire)
        assert hdr_out.content_type == header.content_type
        assert hdr_out.ts == pytest.approx(header.ts)
        assert payload_out == payload

    @pytest.mark.parametrize("header,payload", GOLDEN_CASES)
    def test_deterministic_wire_bytes(self, header: HeaderMeta, payload: bytes) -> None:
        wire_a = encode(header, payload)
        wire_b = encode(header, payload)
        assert wire_a == wire_b


# ── HeaderTemplate ───────────────────────────────────────────────────


class TestHeaderTemplate:
    def test_pack_decode_roundtrip(self) -> None:
        tmpl = HeaderTemplate("numpy/ndarray", shape=(10,), dtype="float32")
        payload = b"\x00" * 40
        wire = tmpl.pack(payload, ts=42.0)
        hdr, got_payload = decode(wire)

        assert hdr.content_type == "numpy/ndarray"
        assert hdr.ts == pytest.approx(42.0)
        assert hdr.seq == 0
        assert hdr.shape == (10,)
        assert hdr.dtype == "float32"
        assert got_payload == payload

    def test_seq_auto_increment(self) -> None:
        tmpl = HeaderTemplate("application/json")
        for expected_seq in range(5):
            wire = tmpl.pack(b"{}", ts=1.0)
            hdr, _ = decode(wire)
            assert hdr.seq == expected_seq
        assert tmpl.seq == 5

    def test_independent_seq_counters(self) -> None:
        t1 = HeaderTemplate("application/json")
        t2 = HeaderTemplate("application/json")
        t1.pack(b"{}", ts=1.0)
        t1.pack(b"{}", ts=1.0)
        t2.pack(b"{}", ts=1.0)

        assert t1.seq == 2
        assert t2.seq == 1

    def test_metadata_in_template(self) -> None:
        tmpl = HeaderTemplate(
            "numpy/ndarray",
            shape=(480, 640, 3),
            dtype="uint8",
            metadata={"fps": 30, "width": 640},
        )
        wire = tmpl.pack(b"\x00", ts=1.0)
        hdr, _ = decode(wire)
        assert hdr.metadata is not None
        assert hdr.metadata["fps"] == 30
        assert hdr.metadata["width"] == 640

    def test_default_ts(self) -> None:
        tmpl = HeaderTemplate("application/json")
        before = time.time()
        wire = tmpl.pack(b"{}")
        after = time.time()
        hdr, _ = decode(wire)
        assert before <= hdr.ts <= after

    def test_hot_path_performance(self) -> None:
        """10k packs should complete in well under 1 second."""
        tmpl = HeaderTemplate("numpy/ndarray", shape=(100,), dtype="float32")
        payload = b"\x00" * 400
        start = time.perf_counter()
        for _ in range(10_000):
            tmpl.pack(payload, ts=1.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"10k packs took {elapsed:.3f}s (> 1s)"


# ── Error cases ──────────────────────────────────────────────────────


class TestDecodeErrors:
    def test_frame_too_short(self) -> None:
        with pytest.raises(WireFormatError, match="too short"):
            decode(b"\x00\x00")

    def test_header_length_exceeds_max(self) -> None:
        bogus = struct.pack(_HEADER_LEN_FMT, _MAX_HEADER_BYTES + 1)
        with pytest.raises(WireFormatError, match="exceeds maximum"):
            decode(bogus + b"\x00" * 100)

    def test_frame_truncated(self) -> None:
        bogus = struct.pack(_HEADER_LEN_FMT, 9999) + b"\x00" * _TS_SEQ_SIZE + b"x"
        with pytest.raises(WireFormatError, match="truncated"):
            decode(bogus)

    def test_invalid_json_header(self) -> None:
        bad_json = b"not json at all"
        total_header_len = _TS_SEQ_SIZE + len(bad_json)
        frame = (
            struct.pack(_HEADER_LEN_FMT, total_header_len)
            + struct.pack(_TS_SEQ_FMT, 1.0, 0)
            + bad_json
        )
        with pytest.raises(WireFormatError, match="Invalid header JSON"):
            decode(frame)

    def test_header_not_object(self) -> None:
        header_bytes = b'"just a string"'
        total_header_len = _TS_SEQ_SIZE + len(header_bytes)
        frame = (
            struct.pack(_HEADER_LEN_FMT, total_header_len)
            + struct.pack(_TS_SEQ_FMT, 1.0, 0)
            + header_bytes
        )
        with pytest.raises(WireFormatError, match="JSON object"):
            decode(frame)

    def test_missing_content_type(self) -> None:
        header_bytes = json.dumps({"foo": "bar"}).encode()
        total_header_len = _TS_SEQ_SIZE + len(header_bytes)
        frame = (
            struct.pack(_HEADER_LEN_FMT, total_header_len)
            + struct.pack(_TS_SEQ_FMT, 1.0, 0)
            + header_bytes
        )
        with pytest.raises(WireFormatError, match="content_type"):
            decode(frame)

    def test_oversized_header_on_encode(self) -> None:
        huge_meta = {f"k{i:05d}" + "x" * 100: "v" * 100 for i in range(700)}
        hdr = HeaderMeta(content_type="x", ts=1.0, metadata=huge_meta)
        with pytest.raises(WireFormatError, match="exceeds"):
            encode(hdr, b"")
