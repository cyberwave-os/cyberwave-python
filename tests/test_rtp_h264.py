"""H.264 over RTP/UDP ingest: RTP parsing, RFC 6184 depayloading, the UDP receiver."""

import base64
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path

import pytest

from cyberwave.sensor.h264_bitstream import (
    NAL_AUD,
    NAL_IDR,
    NAL_PPS,
    NAL_SEI,
    NAL_SPS,
    ParameterSetCache,
    first_mb_in_slice,
    join_annexb,
    nal_type,
    parse_sps_dimensions,
    split_annexb,
)
from cyberwave.sensor.rtp_h264 import (
    NAL_FU_A,
    AccessUnit,
    H264Depayloader,
    RtpH264Packetizer,
    RtpPacket,
    RtpParseError,
    UdpRtpReceiver,
    format_sprop_parameter_sets,
    parse_bind,
    parse_sprop_parameter_sets,
    payload_starts_access_unit,
    sequence_delta,
)

# Real SPS/PPS from libx264.
SPS_640x360 = base64.b64decode("Z0LAHtoCgL/lhAAAAwAEAAADAPI8WLqA")
PPS = base64.b64decode("aM4PyA==")
SPS_HIGH_1080 = base64.b64decode("Z2QAKKzZQHgCJ+WEAAADAAQAAAMA8Dxgxlg=")
SPS_HIGH_INTERLACED_576 = base64.b64decode("Z2QAH6zZQLQk0IAAAAMAgAAAHg+LFss=")
SPS_BASELINE_720 = base64.b64decode("Z0LAH9oBQBboQAAAAwBAAAAPA8YMqA==")


def slice_nal(header: int, body: bytes, first_mb: int = 0) -> bytes:
    """A coded slice whose header starts with ue(first_mb_in_slice)."""
    code = first_mb + 1
    bits = "0" * (code.bit_length() - 1) + format(code, "b")
    bits += "1" * (-len(bits) % 8)
    head = int(bits, 2).to_bytes(len(bits) // 8, "big")
    return bytes([header]) + head + body


def idr(body: bytes, first_mb: int = 0) -> bytes:
    return slice_nal(0x65, body, first_mb)


def p_slice(body: bytes, first_mb: int = 0) -> bytes:
    return slice_nal(0x41, body, first_mb)


IDR = idr(bytes(range(1, 200)))
P_SLICE = p_slice(b"\x9a" * 50)
AUD = bytes([0x09, 0xF0])

SSRC = 0x1234ABCD


def rtp(
    payload: bytes,
    seq: int,
    ts: int,
    *,
    marker: bool = False,
    pt: int = 96,
    ssrc: int = SSRC,
) -> RtpPacket:
    return RtpPacket(
        payload_type=pt,
        sequence=seq & 0xFFFF,
        timestamp=ts,
        ssrc=ssrc,
        payload=payload,
        marker=marker,
    )


def stap_a(*nals: bytes) -> bytes:
    nri = max(n[0] & 0x60 for n in nals)
    return bytes([nri | 24]) + b"".join(struct.pack("!H", len(n)) + n for n in nals)


def fu_a(nal: bytes, chunk: int) -> list[bytes]:
    indicator = (nal[0] & 0xE0) | NAL_FU_A
    body = nal[1:]
    parts = [body[i : i + chunk] for i in range(0, len(body), chunk)]
    out = []
    for i, part in enumerate(parts):
        header = nal[0] & 0x1F
        if i == 0:
            header |= 0x80
        if i == len(parts) - 1:
            header |= 0x40
        out.append(bytes([indicator, header]) + part)
    return out


def push_all(depay: H264Depayloader, packets: list[RtpPacket]) -> list[AccessUnit]:
    out = []
    for packet in packets:
        out.extend(depay.push(packet))
    return out


# -- RTP header ------------------------------------------------------------------


def test_rtp_parse_basic_header():
    raw = bytes([0x80, 0x80 | 96]) + struct.pack("!HII", 513, 90000, SSRC) + b"\x65abc"
    packet = RtpPacket.parse(raw)
    assert packet.payload_type == 96
    assert packet.marker is True
    assert packet.sequence == 513
    assert packet.timestamp == 90000
    assert packet.ssrc == SSRC
    assert packet.payload == b"\x65abc"
    assert packet.csrcs == ()


def test_rtp_parse_strips_csrc_extension_and_padding():
    header = bytes([0x80 | 0x20 | 0x10 | 2, 96]) + struct.pack("!HII", 7, 1, SSRC)
    csrcs = struct.pack("!II", 11, 22)
    extension = struct.pack("!HH", 0xBEDE, 2) + b"\x01\x02\x03\x04\x05\x06\x07\x08"
    payload = b"\x41payload"
    padding = b"\x00\x00\x03"
    packet = RtpPacket.parse(header + csrcs + extension + payload + padding)
    assert packet.csrcs == (11, 22)
    assert packet.extension_profile == 0xBEDE
    assert packet.extension_data == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert packet.payload == payload
    assert packet.padding
    assert packet.extension


@pytest.mark.parametrize(
    "raw",
    [
        b"\x80\x60\x00",  # too short
        bytes([0x40, 96]) + bytes(10),  # version 1
        bytes([0x83, 96]) + bytes(10),  # 3 CSRCs announced, none present
        bytes([0x90, 96]) + bytes(10) + struct.pack("!HH", 0, 5),  # ext too long
        bytes([0xA0, 96]) + bytes(10) + b"\x41\x00",  # padding byte 0
        bytes([0xA0, 96]) + bytes(10) + b"\x41\x09",  # padding past payload
    ],
)
def test_rtp_parse_rejects_malformed(raw):
    with pytest.raises(RtpParseError):
        RtpPacket.parse(raw)


def test_rtp_serialize_round_trip():
    packet = RtpPacket(
        payload_type=97,
        sequence=65535,
        timestamp=0xFFFFFFFF,
        ssrc=SSRC,
        payload=b"\x65xyz",
        marker=True,
        extension=True,
        extension_profile=0x1000,
        extension_data=b"\xaa" * 4,
        csrcs=(5,),
    )
    assert RtpPacket.parse(packet.serialize()) == packet


def test_sequence_delta_wraps():
    assert sequence_delta(65535, 0) == 1
    assert sequence_delta(65530, 3) == 9
    assert sequence_delta(3, 65530) == -9
    assert sequence_delta(10, 10) == 0


# -- Annex-B / parameter sets ---------------------------------------------------------


def test_split_annexb_handles_three_and_four_byte_start_codes():
    data = (
        b"\x00\x00\x00\x01"
        + SPS_640x360
        + b"\x00\x00\x01"
        + PPS
        + b"\x00\x00\x00\x01"
        + IDR
    )
    assert split_annexb(data) == [SPS_640x360, PPS, IDR]
    assert split_annexb(join_annexb([P_SLICE])) == [P_SLICE]
    assert split_annexb(b"") == []


@pytest.mark.parametrize(
    ("sps", "size"),
    [
        (SPS_640x360, (640, 360)),
        (SPS_BASELINE_720, (1280, 720)),
        (SPS_HIGH_1080, (1920, 1080)),
        (SPS_HIGH_INTERLACED_576, (720, 576)),
    ],
)
def test_parse_sps_dimensions(sps, size):
    assert parse_sps_dimensions(sps) == size


def test_parse_sps_rejects_non_sps():
    with pytest.raises(ValueError, match="SPS"):
        parse_sps_dimensions(PPS)


def test_sprop_round_trip():
    value = format_sprop_parameter_sets([SPS_640x360, PPS])
    assert parse_sprop_parameter_sets(value) == [SPS_640x360, PPS]
    # Missing base64 padding is common in hand-written SDP.
    assert parse_sprop_parameter_sets("Z0LAHtoCgL/lhAAAAwAEAAADAPI8WLqA,aM4PyA") == [
        SPS_640x360,
        PPS,
    ]


# -- depayloader ----------------------------------------------------------------------


def test_single_nal_units_make_one_keyframe_access_unit():
    depay = H264Depayloader()
    units = push_all(
        depay,
        [
            rtp(SPS_640x360, 1, 3000),
            rtp(PPS, 2, 3000),
            rtp(IDR, 3, 3000, marker=True),
        ],
    )
    assert len(units) == 1
    assert units[0].data == join_annexb([SPS_640x360, PPS, IDR])
    assert units[0].is_keyframe
    assert units[0].timestamp == 3000
    assert (depay.width, depay.height) == (640, 360)
    assert depay.stats.parameter_sets_prepended == 0


def test_stap_a_and_fu_a_reassembly():
    big_idr = idr(bytes(i % 251 for i in range(5000)))
    packets = [rtp(stap_a(SPS_640x360, PPS), 10, 0)]
    fragments = fu_a(big_idr, 1000)
    for i, fragment in enumerate(fragments):
        packets.append(rtp(fragment, 11 + i, 0, marker=i == len(fragments) - 1))
    packets.append(rtp(P_SLICE, 11 + len(fragments), 3000, marker=True))
    units = push_all(H264Depayloader(), packets)
    assert [u.data for u in units] == [
        join_annexb([SPS_640x360, PPS, big_idr]),
        join_annexb([P_SLICE]),
    ]
    assert [u.is_keyframe for u in units] == [True, False]


def test_access_unit_emitted_on_timestamp_change_without_marker():
    depay = H264Depayloader()
    units = push_all(
        depay,
        [
            rtp(stap_a(SPS_640x360, PPS), 1, 0),
            rtp(IDR, 2, 0),  # no marker
            rtp(P_SLICE, 3, 3000),
        ],
    )
    assert len(units) == 1
    assert units[0].is_keyframe
    assert depay.flush()[0].data == join_annexb([P_SLICE])


def test_sequence_wrap_is_not_loss():
    packets = [
        rtp(stap_a(SPS_640x360, PPS), 65534, 0),
        rtp(IDR, 65535, 0, marker=True),
        rtp(P_SLICE, 0, 3000, marker=True),
        rtp(P_SLICE, 1, 6000, marker=True),
    ]
    depay = H264Depayloader()
    units = push_all(depay, packets)
    assert len(units) == 3
    assert depay.stats.losses == 0


def test_loss_drops_until_next_idr():
    depay = H264Depayloader()
    packets = [
        rtp(stap_a(SPS_640x360, PPS), 1, 0),
        rtp(IDR, 2, 0, marker=True),
        rtp(P_SLICE, 3, 3000, marker=True),
        # seq 4 (a P-frame) lost
        rtp(P_SLICE, 5, 9000, marker=True),
        rtp(P_SLICE, 6, 12000, marker=True),
        rtp(IDR, 7, 15000, marker=True),
        rtp(P_SLICE, 8, 18000, marker=True),
    ]
    units = push_all(depay, packets)
    assert [u.timestamp for u in units] == [0, 3000, 15000, 18000]
    assert depay.stats.losses == 1
    assert depay.stats.lost_packets == 1
    assert depay.stats.dropped_access_units == 2
    # The resync IDR carried no parameter sets in-band: the cached ones are added.
    assert units[2].data == join_annexb([SPS_640x360, PPS, IDR])


def test_lost_fu_a_fragment_drops_the_picture():
    big_p = p_slice(b"\x55" * 3000)
    fragments = fu_a(big_p, 1000)
    depay = H264Depayloader()
    packets = [
        rtp(stap_a(SPS_640x360, PPS), 1, 0),
        rtp(IDR, 2, 0, marker=True),
        rtp(fragments[0], 3, 3000),
        # fragments[1] lost (seq 4)
        rtp(fragments[2], 5, 3000, marker=True),
        rtp(P_SLICE, 6, 6000, marker=True),
    ]
    units = push_all(depay, packets)
    assert [u.timestamp for u in units] == [0]
    assert depay.waiting_for_keyframe


def test_fu_a_end_without_start_waits_for_keyframe():
    big_idr = idr(b"\x11" * 3000)
    fragments = fu_a(big_idr, 1000)
    depay = H264Depayloader()
    # Joined mid-NAL: first packet seen is a middle fragment.
    units = push_all(
        depay,
        [rtp(fragments[1], 100, 0), rtp(fragments[2], 101, 0, marker=True)],
    )
    assert units == []
    assert depay.waiting_for_keyframe


def test_waits_for_first_keyframe():
    depay = H264Depayloader()
    units = push_all(
        depay,
        [
            rtp(P_SLICE, 1, 0, marker=True),
            rtp(P_SLICE, 2, 3000, marker=True),
            rtp(stap_a(SPS_640x360, PPS), 3, 6000),
            rtp(IDR, 4, 6000, marker=True),
        ],
    )
    assert [u.timestamp for u in units] == [6000]


def test_cached_parameter_sets_prepended_after_aud():
    depay = H264Depayloader()
    push_all(
        depay,
        [
            rtp(stap_a(SPS_640x360, PPS), 1, 0),
            rtp(IDR, 2, 0, marker=True),
        ],
    )
    (unit,) = push_all(depay, [rtp(stap_a(AUD, IDR), 3, 3000, marker=True)])
    assert unit.data == join_annexb([AUD, SPS_640x360, PPS, IDR])
    assert unit.nal_types == (NAL_AUD, NAL_SPS, NAL_PPS, NAL_IDR)
    assert depay.stats.parameter_sets_prepended == 1


def test_cached_pps_follows_the_sps_the_keyframe_carries():
    """A fresh decoder parses a PPS against its SPS, so the PPS goes after it."""
    depay = H264Depayloader()
    push_all(depay, [rtp(stap_a(SPS_640x360, PPS), 1, 0), rtp(IDR, 2, 0, marker=True)])
    (unit,) = push_all(depay, [rtp(stap_a(SPS_640x360, IDR), 3, 3000, marker=True)])
    assert unit.nal_types == (NAL_SPS, NAL_PPS, NAL_IDR)
    assert depay.stats.parameter_sets_prepended == 1


# -- ParameterSetCache: one insertion point for the depayloader and the encoder ---------

SEI = bytes([0x06, 0x05, 0x01, 0x00, 0x80])


def _cache() -> ParameterSetCache:
    cache = ParameterSetCache()
    for nal in (SPS_640x360, PPS):
        cache.remember(nal)
    return cache


@pytest.mark.parametrize(
    ("access_unit", "completed"),
    [
        # After a leading AUD (which must stay first), before SEI and slices.
        ([AUD, SEI, IDR], [AUD, SPS_640x360, PPS, SEI, IDR]),
        ([SEI, IDR], [SPS_640x360, PPS, SEI, IDR]),
        ([IDR], [SPS_640x360, PPS, IDR]),
        # Only what is missing; a cached PPS goes after the AU's own SPS ...
        ([AUD, SPS_640x360, IDR], [AUD, SPS_640x360, PPS, IDR]),
        # ... and a cached SPS before the AU's own PPS.
        ([PPS, IDR], [SPS_640x360, PPS, IDR]),
        ([SPS_640x360, PPS, IDR], [SPS_640x360, PPS, IDR]),
    ],
)
def test_parameter_set_insertion_point(access_unit, completed):
    assert nal_type(SEI) == NAL_SEI
    assert _cache().complete(access_unit) == completed


def test_parameter_set_cache_without_cached_sets_is_a_no_op():
    assert ParameterSetCache().complete([AUD, IDR]) == [AUD, IDR]


def test_parameter_set_cache_reports_a_new_sps_only():
    cache = ParameterSetCache()
    assert cache.remember(SPS_640x360) is True
    assert cache.remember(SPS_640x360) is False
    assert cache.remember(PPS) is False
    assert cache.remember(IDR) is False
    assert (cache.sps, cache.pps) == (SPS_640x360, PPS)
    assert cache.remember(SPS_BASELINE_720) is True


def test_sprop_parameter_sets_seed_the_cache():
    depay = H264Depayloader(
        sprop_parameter_sets=format_sprop_parameter_sets([SPS_640x360, PPS])
    )
    assert (depay.width, depay.height) == (640, 360)
    (unit,) = push_all(depay, [rtp(IDR, 1, 0, marker=True)])
    assert unit.data == join_annexb([SPS_640x360, PPS, IDR])
    assert depay.stats.keyframes_without_parameter_sets == 0


def test_keyframe_without_any_parameter_sets_is_counted_but_forwarded():
    depay = H264Depayloader()
    (unit,) = push_all(depay, [rtp(IDR, 1, 0, marker=True)])
    assert unit.data == join_annexb([IDR])
    assert depay.stats.keyframes_without_parameter_sets == 1


def test_unsupported_packetization_is_counted_and_forces_resync():
    depay = H264Depayloader()
    push_all(depay, [rtp(stap_a(SPS_640x360, PPS), 1, 0), rtp(IDR, 2, 0, marker=True)])
    stap_b = bytes([0x79]) + b"\x00\x01" + struct.pack("!H", 2) + b"\x41\x00"
    units = push_all(
        depay,
        [rtp(stap_b, 3, 3000, marker=True), rtp(P_SLICE, 4, 6000, marker=True)],
    )
    assert units == []
    assert depay.stats.unsupported_packets == 1


def test_duplicate_and_late_packets_are_ignored():
    depay = H264Depayloader()
    units = push_all(
        depay,
        [
            rtp(stap_a(SPS_640x360, PPS), 10, 0),
            rtp(IDR, 11, 0, marker=True),
            rtp(IDR, 11, 0, marker=True),
            rtp(P_SLICE, 9, 3000, marker=True),
            rtp(P_SLICE, 12, 3000, marker=True),
        ],
    )
    assert len(units) == 2
    assert depay.stats.duplicates == 1
    assert depay.stats.reordered == 1
    assert depay.stats.losses == 0


def test_ssrc_change_resynchronises():
    depay = H264Depayloader()
    push_all(depay, [rtp(stap_a(SPS_640x360, PPS), 1, 0), rtp(IDR, 2, 0, marker=True)])
    units = push_all(
        depay,
        [
            rtp(P_SLICE, 500, 3000, marker=True, ssrc=99),
            rtp(IDR, 501, 6000, marker=True, ssrc=99),
        ],
    )
    assert [u.timestamp for u in units] == [6000]
    assert depay.stats.stream_resets == 1
    assert depay.stats.losses == 0


def test_first_mb_in_slice_and_access_unit_start_detection():
    assert first_mb_in_slice(IDR) == 0
    assert first_mb_in_slice(idr(b"\x00", first_mb=40)) == 40
    assert first_mb_in_slice(SPS_640x360) is None
    assert payload_starts_access_unit(IDR)
    assert payload_starts_access_unit(AUD)
    assert payload_starts_access_unit(stap_a(SPS_640x360, PPS))
    assert not payload_starts_access_unit(idr(b"\x00" * 10, first_mb=40))
    fragments = fu_a(idr(b"\x33" * 3000), 1000)
    assert payload_starts_access_unit(fragments[0])
    assert not payload_starts_access_unit(fragments[1])
    tail = fu_a(idr(b"\x33" * 3000, first_mb=80), 1000)
    assert not payload_starts_access_unit(tail[0])


def _gop_start(depay: H264Depayloader) -> list[AccessUnit]:
    return push_all(
        depay,
        [
            rtp(stap_a(SPS_640x360, PPS), 1, 0),
            rtp(IDR, 2, 0, marker=True),
            rtp(P_SLICE, 3, 3000, marker=True),
        ],
    )


def test_lost_first_slice_of_multi_slice_idr_is_not_a_keyframe():
    # Regression: losing slice 0 used to emit (True, (7, 8, 5)) and clear
    # waiting_for_keyframe, forwarding a half-grey IDR as a resync point.
    depay = H264Depayloader()
    assert len(_gop_start(depay)) == 2
    units = push_all(
        depay,
        [
            # seq 4: slice 0 of the IDR at ts 6000, lost
            rtp(idr(b"\x01" * 100, first_mb=40), 5, 6000),
            rtp(idr(b"\x02" * 100, first_mb=80), 6, 6000, marker=True),
            rtp(P_SLICE, 7, 9000, marker=True),
        ],
    )
    assert units == []
    assert depay.waiting_for_keyframe
    assert depay.stats.poisoned_packets == 2  # both surviving slices of ts 6000
    (unit,) = push_all(depay, [rtp(IDR, 8, 12000, marker=True)])
    assert unit.is_keyframe
    assert unit.timestamp == 12000


def test_lost_middle_slice_poisons_the_rest_of_the_picture():
    depay = H264Depayloader()
    _gop_start(depay)
    units = push_all(
        depay,
        [
            rtp(stap_a(SPS_640x360, PPS), 4, 6000),
            rtp(IDR, 5, 6000),
            # seq 6 (slice at mb 40) lost
            rtp(idr(b"\x02" * 100, first_mb=80), 7, 6000, marker=True),
            rtp(P_SLICE, 8, 9000, marker=True),
        ],
    )
    assert units == []
    assert depay.waiting_for_keyframe
    assert depay.stats.poisoned_packets == 1  # the slice after the gap


def test_loss_at_tail_of_previous_picture_keeps_a_clean_next_idr():
    depay = H264Depayloader()
    _gop_start(depay)
    units = push_all(
        depay,
        [
            rtp(P_SLICE, 4, 6000),
            # seq 5 (last slice + marker of ts 6000) lost
            rtp(stap_a(SPS_640x360, PPS), 6, 9000),
            rtp(IDR, 7, 9000, marker=True),
        ],
    )
    assert [(u.timestamp, u.is_keyframe) for u in units] == [(9000, True)]
    assert not depay.waiting_for_keyframe


def test_joining_mid_idr_without_gap_is_not_a_keyframe():
    depay = H264Depayloader(
        sprop_parameter_sets=format_sprop_parameter_sets([SPS_640x360, PPS])
    )
    units = push_all(
        depay,
        [
            rtp(idr(b"\x02" * 100, first_mb=80), 500, 0, marker=True),
            rtp(P_SLICE, 501, 3000, marker=True),
        ],
    )
    assert units == []
    assert depay.stats.incomplete_access_units == 1
    assert depay.waiting_for_keyframe


def test_sender_restart_with_same_ssrc_and_backwards_sequence_resyncs():
    # Regression: a restarted sender reusing its SSRC with a lower sequence was
    # discarded as "reordered" for up to 32768 packets (frozen video).
    depay = H264Depayloader()
    push_all(
        depay,
        [
            rtp(stap_a(SPS_640x360, PPS), 40000, 0),
            rtp(IDR, 40001, 0, marker=True),
        ],
    )
    units = push_all(
        depay,
        [
            rtp(P_SLICE, 30000, 777, marker=True),  # mid-GOP after restart
            rtp(stap_a(SPS_640x360, PPS), 30001, 3777),
            rtp(IDR, 30002, 3777, marker=True),
            rtp(P_SLICE, 30003, 6777, marker=True),
        ],
    )
    assert [(u.timestamp, u.is_keyframe) for u in units] == [
        (3777, True),
        (6777, False),
    ]
    assert depay.stats.stream_resets == 1
    assert depay.stats.reordered == 0
    # A small backwards step is still just a late packet.
    assert push_all(depay, [rtp(P_SLICE, 30000, 777, marker=True)]) == []
    assert depay.stats.reordered == 1


# -- packetizer ---------------------------------------------------------------------------


def test_packetizer_round_trip_is_byte_exact_including_large_nals():
    big_idr = idr(bytes((i * 7) % 256 for i in range(10_000)))
    big_p = p_slice(bytes((i * 13) % 256 for i in range(2_500)))
    access_units = [
        join_annexb([SPS_640x360, PPS, big_idr]),
        join_annexb([P_SLICE]),
        join_annexb([big_p]),
        join_annexb([AUD, P_SLICE]),
    ]
    packetizer = RtpH264Packetizer(ssrc=SSRC, initial_sequence=65530, mtu=1200)
    depay = H264Depayloader()
    out = []
    for index, au in enumerate(access_units):
        packets = packetizer.packetize(au, 0xFFFFFF00 + index * 3000)
        parsed = [RtpPacket.parse(p) for p in packets]
        assert all(len(p.payload) <= 1200 for p in parsed)
        assert [p.marker for p in parsed] == [False] * (len(parsed) - 1) + [True]
        assert {p.timestamp for p in parsed} == {
            (0xFFFFFF00 + index * 3000) & 0xFFFFFFFF
        }
        for packet in parsed:
            out.extend(depay.push(packet))
    assert [u.data for u in out] == access_units
    assert depay.stats.losses == 0
    assert packetizer.sequence < 65530  # wrapped


def test_packetizer_uses_single_nal_up_to_mtu_and_fu_a_above():
    packetizer = RtpH264Packetizer(ssrc=1, initial_sequence=0, mtu=100)
    exact = bytes([0x41]) + b"\x01" * 99
    over = bytes([0x41]) + b"\x01" * 100
    (single,) = packetizer.packetize(join_annexb([exact]), 0)
    assert RtpPacket.parse(single).payload == exact
    fragments = [
        RtpPacket.parse(p).payload for p in packetizer.packetize(join_annexb([over]), 1)
    ]
    assert len(fragments) == 2
    assert all(nal_type(f) == NAL_FU_A for f in fragments)
    assert fragments[0][1] & 0x80
    assert fragments[1][1] & 0x40


# -- receiver -------------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _datagrams(access_units: list[bytes], start_ts: int = 0) -> list[bytes]:
    packetizer = RtpH264Packetizer(ssrc=SSRC, initial_sequence=1)
    out = []
    for index, au in enumerate(access_units):
        out.extend(packetizer.packetize(au, start_ts + index * 3000))
    return out


def test_receiver_hands_each_access_unit_over_once():
    clock = FakeClock()
    rx = UdpRtpReceiver("127.0.0.1", 0, clock=clock)
    assert rx.get_packet() is None
    for datagram in _datagrams(
        [join_annexb([SPS_640x360, PPS, IDR]), join_annexb([P_SLICE])]
    ):
        rx.handle_datagram(datagram, "10.0.0.5")
    assert rx.keyframe_seen
    assert rx.get_packet() == (join_annexb([SPS_640x360, PPS, IDR]), True)
    assert rx.get_packet() == (join_annexb([P_SLICE]), False)
    assert rx.get_packet() is None
    clock.now += 2.5
    stats = rx.stats()
    assert stats.access_units == 2
    assert stats.keyframes == 1
    assert stats.packets == 4  # SPS, PPS, IDR, P: one NAL per packet
    assert stats.last_packet_age_s == pytest.approx(2.5)
    assert (stats.width, stats.height) == (640, 360)
    assert stats.ssrc == SSRC
    assert stats.source == "10.0.0.5"


def test_receiver_source_allowlist_and_payload_type_filter():
    rx = UdpRtpReceiver("127.0.0.1", 0, allowed_source="192.168.144.10")
    datagram = _datagrams([join_annexb([SPS_640x360, PPS, IDR])])[0]
    assert rx.handle_datagram(datagram, "192.168.144.99") == []
    rtcp_like = bytes([0x80, 72]) + bytes(10)
    rx.handle_datagram(rtcp_like, "192.168.144.10")
    rx.handle_datagram(b"garbage", "192.168.144.10")
    stats = rx.stats()
    assert stats.rejected_packets == 3
    assert stats.packets == 0


def test_receiver_queue_overflow_keeps_a_decodable_chain():
    rx = UdpRtpReceiver("127.0.0.1", 0, queue_size=4)
    gop = [join_annexb([SPS_640x360, PPS, IDR])] + [join_annexb([P_SLICE])] * 2
    # Two GOPs of 3 into a queue of 4: the overflow restarts at the 2nd IDR.
    for datagram in _datagrams(gop + gop + [join_annexb([P_SLICE])]):
        rx.handle_datagram(datagram, "127.0.0.1")
    drained = []
    while (item := rx.get_packet()) is not None:
        drained.append(item[1])
    assert drained == [True, False, False, False]
    assert rx.stats().queue_overflows == 1


def test_receiver_overflow_without_queued_keyframe_skips_to_next_idr():
    rx = UdpRtpReceiver("127.0.0.1", 0, queue_size=3)
    aus = [join_annexb([SPS_640x360, PPS, IDR])] + [join_annexb([P_SLICE])] * 5
    aus += [join_annexb([IDR]), join_annexb([P_SLICE])]
    for datagram in _datagrams(aus):
        rx.handle_datagram(datagram, "127.0.0.1")
    drained = []
    while (item := rx.get_packet()) is not None:
        drained.append(item[1])
    assert drained == [True, False]


def test_receiver_over_real_udp_socket():
    aus = [join_annexb([SPS_640x360, PPS, idr(b"\x22" * 4000)])]
    aus += [join_annexb([P_SLICE])] * 5
    with UdpRtpReceiver("127.0.0.1", 0) as rx:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for datagram in _datagrams(aus):
                sender.sendto(datagram, rx.address)
        finally:
            sender.close()
        assert rx.wait_for_keyframe(2.0)
        deadline = time.monotonic() + 2.0
        received = []
        while len(received) < len(aus) and time.monotonic() < deadline:
            item = rx.get_packet()
            if item is None:
                time.sleep(0.01)
            else:
                received.append(item[0])
    assert received == aus


# -- helpers ---------------------------------------------------------------------------------


def test_parse_bind():
    assert parse_bind("0.0.0.0:5600") == ("0.0.0.0", 5600)  # noqa: S104
    assert parse_bind(":5601") == ("0.0.0.0", 5601)  # noqa: S104
    assert parse_bind("10.0.0.2") == ("10.0.0.2", 5600)
    assert parse_bind("[::]:5602") == ("::", 5602)


# -- integration: real encoders --------------------------------------------------------------

FFMPEG = shutil.which("ffmpeg") or (
    "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else None
)


def _decode(access_units: list[tuple[bytes, bool]]) -> list[tuple[int, int]]:
    av = pytest.importorskip("av")
    decoder = av.CodecContext.create("h264", "r")
    sizes = []
    for data, _ in access_units:
        sizes.extend((f.width, f.height) for f in decoder.decode(av.Packet(data)))
    return sizes


def _drain(rx: UdpRtpReceiver) -> list[tuple[bytes, bool]]:
    out = []
    while (item := rx.get_packet()) is not None:
        out.append(item)
    return out


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_ffmpeg_rtp_stream_decodes():
    pytest.importorskip("av")
    with UdpRtpReceiver("127.0.0.1", 0, queue_size=200) as rx:
        port = rx.address[1]
        subprocess.run(  # noqa: S603 - fixed argv, test-only
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=640x360:rate=30",
                "-t",
                "3",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                "15",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rtp",
                "-payload_type",
                "96",
                f"rtp://127.0.0.1:{port}",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        time.sleep(0.3)
        stats = rx.stats()
        units = _drain(rx)
    assert stats.keyframes >= 5
    assert stats.access_units >= 80
    assert stats.losses == 0
    assert (stats.width, stats.height) == (640, 360)
    assert units[0][1] is True
    sizes = _decode(units)
    assert len(sizes) >= 80
    assert set(sizes) == {(640, 360)}


# -- RtpH264CameraStreamer -------------------------------------------------------------------


def test_rtp_streamer_waits_for_keyframe_and_reports_sps_resolution(monkeypatch):
    import asyncio
    from unittest import mock

    pytest.importorskip("av")
    pytest.importorskip("aiortc")
    from cyberwave.sensor import camera_h264
    from cyberwave.sensor.camera_h264 import RtpH264CameraStreamer

    seen = {}

    async def fake_run(
        self, stop_event=None, command_callback=None, subscribe_to_commands=True
    ):
        track = self.initialize_track()
        seen["attributes"] = track.get_stream_attributes()
        seen["packet"] = self.get_packet()
        seen["subscribe"] = subscribe_to_commands

    monkeypatch.setattr(
        camera_h264.H264PacketCameraStreamer, "run_with_auto_reconnect", fake_run
    )
    streamer = RtpH264CameraStreamer(
        mock.MagicMock(topic_prefix="", client_id="edge"),
        bind="127.0.0.1:0",
        twin_uuid="5f2c7d0e-8a51-4d0f-9e0b-2a9c1f4d6b11",
        camera_name="main",
        fps=30,
    )
    assert streamer.framing == "annexb"

    async def main():
        task = asyncio.create_task(
            streamer.run_with_auto_reconnect(subscribe_to_commands=False)
        )
        for _ in range(100):
            if streamer.receiver._sock is not None:
                break
            await asyncio.sleep(0.01)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for datagram in _datagrams([join_annexb([SPS_640x360, PPS, IDR])]):
                sender.sendto(datagram, streamer.receiver.address)
        finally:
            sender.close()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())
    assert seen["attributes"]["width"] == 640
    assert seen["attributes"]["height"] == 360
    assert seen["packet"] == (join_annexb([SPS_640x360, PPS, IDR]), True)
    assert seen["subscribe"] is False
    # The streamer owned the receiver, so it stopped it.
    assert streamer.receiver._sock is None


def test_rtp_streamer_stops_before_keyframe_without_streaming(monkeypatch):
    import asyncio
    from unittest import mock

    pytest.importorskip("av")
    pytest.importorskip("aiortc")
    from cyberwave.sensor import camera_h264
    from cyberwave.sensor.camera_h264 import RtpH264CameraStreamer

    base_run = mock.AsyncMock()
    monkeypatch.setattr(
        camera_h264.H264PacketCameraStreamer, "run_with_auto_reconnect", base_run
    )
    receiver = UdpRtpReceiver("127.0.0.1", 0)
    streamer = RtpH264CameraStreamer(
        mock.MagicMock(topic_prefix="", client_id="edge"),
        receiver=receiver,
        twin_uuid="5f2c7d0e-8a51-4d0f-9e0b-2a9c1f4d6b11",
    )
    stop = asyncio.Event()
    stop.set()
    asyncio.run(streamer.run_with_auto_reconnect(stop_event=stop))
    base_run.assert_not_called()
    # A caller-supplied receiver is neither started nor stopped by the streamer.
    assert receiver._thread is None


def test_deeply_nested_headers_are_bounded_not_a_recursion_error():
    """PR #4622 review: ~1,400 nested FU-A headers cost a RecursionError per packet."""
    import time

    from cyberwave.sensor.rtp_h264 import payload_starts_access_unit

    hostile = b"\x7c" + b"\x9c" * 1400
    started = time.perf_counter()
    for _ in range(200):
        assert payload_starts_access_unit(hostile) is False
    assert time.perf_counter() - started < 0.5
