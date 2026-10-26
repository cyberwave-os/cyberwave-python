"""H.264 Annex-B bitstream helpers shared by the SDK's encode and ingest paths.

NAL unit framing (Annex-B start codes), the few bit-level reads the SDK needs
(SPS resolution, ``first_mb_in_slice``), and :class:`ParameterSetCache`, which
keeps keyframes self-contained for decoders that join mid-stream. Used by the
RTP ingest (:mod:`cyberwave.sensor.rtp_h264`), the hardware encoder
(:mod:`cyberwave.sensor.hw_encoder`) and the passthrough track
(:mod:`cyberwave.sensor.camera_h264`).

Stdlib-only, so it imports (and is tested) without PyAV, aiortc or GStreamer.
"""

from __future__ import annotations

from collections.abc import Iterable

ANNEXB_START_CODE = b"\x00\x00\x00\x01"

NAL_SLICE = 1
NAL_IDR = 5
NAL_SEI = 6
NAL_SPS = 7
NAL_PPS = 8
NAL_AUD = 9

VCL_SLICES = frozenset({NAL_SLICE, NAL_IDR})


def nal_type(nal: bytes) -> int:
    return nal[0] & 0x1F


def split_annexb(data: bytes) -> list[bytes]:
    """Split an Annex-B byte stream into NAL units (3- or 4-byte start codes)."""
    nals: list[bytes] = []
    n = len(data)
    i = data.find(b"\x00\x00\x01")
    if i < 0:
        return [data] if data else []
    start = i + 3
    while True:
        nxt = data.find(b"\x00\x00\x01", start)
        end = n if nxt < 0 else nxt
        # trailing_zero_8bits and the leading zero of a 4-byte start code.
        nal_end = end
        while nal_end > start and data[nal_end - 1] == 0:
            nal_end -= 1
        if nal_end > start:
            nals.append(data[start:nal_end])
        if nxt < 0:
            return nals
        start = nxt + 3


def join_annexb(nals: Iterable[bytes]) -> bytes:
    return b"".join(ANNEXB_START_CODE + nal for nal in nals)


class ParameterSetCache:
    """The latest SPS/PPS seen, re-inserted into keyframes that lack them.

    Some sources send SPS/PPS once at stream start (hardware encoders such as
    v4l2m2m/nvmpi, CYB-2835) or only out of band (SDP
    ``sprop-parameter-sets``), so a decoder that joins mid-stream gets an IDR it
    cannot decode. :meth:`complete` inserts the cached sets the access unit is
    missing.

    Insertion point: directly after a leading access unit delimiter, before
    everything else (SEI included). A decoder needs SPS/PPS before the first
    slice; H.264 7.4.1.2.3 requires an AUD, when present, to be the first NAL of
    its access unit, and SEI such as ``buffering_period`` refers to the active
    SPS, so parameter sets right after the AUD is the order encoders emit
    themselves. A cached PPS goes after the SPS it is parsed against.
    """

    def __init__(self) -> None:
        self.sps: bytes | None = None
        self.pps: bytes | None = None

    def remember(self, nal: bytes) -> bool:
        """Cache ``nal`` if it is an SPS or PPS; True for an SPS that is new."""
        kind = nal_type(nal)
        if kind == NAL_SPS and nal != self.sps:
            self.sps = nal
            return True
        if kind == NAL_PPS:
            self.pps = nal
        return False

    def complete(self, nals: list[bytes]) -> list[bytes]:
        """``nals`` (one access unit) with the cached SPS/PPS it lacks inserted."""
        types = {nal_type(nal) for nal in nals}
        out = list(nals)
        head = 1 if out and nal_type(out[0]) == NAL_AUD else 0
        if self.sps is not None and NAL_SPS not in types:
            out.insert(head, self.sps)
        if self.pps is not None and NAL_PPS not in types:
            sps_at = [i for i, nal in enumerate(out) if nal_type(nal) == NAL_SPS]
            out.insert(sps_at[-1] + 1 if sps_at else head, self.pps)
        return out


# -- bit-level parsing -----------------------------------------------------------


def _rbsp(nal_payload: bytes) -> bytes:
    """Remove emulation-prevention bytes (00 00 03 -> 00 00)."""
    out = bytearray()
    zeros = 0
    for byte in nal_payload:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def bit(self) -> int:
        byte = self._pos >> 3
        if byte >= len(self._data):
            raise ValueError("SPS truncated")
        value = (self._data[byte] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return value

    def bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value

    def ue(self) -> int:
        zeros = 0
        while self.bit() == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("exp-Golomb code too long")
        return (1 << zeros) - 1 + self.bits(zeros)

    def se(self) -> int:
        code = self.ue()
        return (code + 1) // 2 if code & 1 else -(code // 2)


def first_mb_in_slice(nal: bytes) -> int | None:
    """``first_mb_in_slice`` of a coded slice (NAL 1/5), None if unreadable.

    Zero means the slice starts the picture, so an access unit whose first
    slice is not at macroblock 0 lost its head and cannot be a clean keyframe.
    """
    if len(nal) < 2 or nal_type(nal) not in VCL_SLICES:
        return None
    try:
        return _BitReader(_rbsp(nal[1:9])).ue()
    except ValueError:
        return None


_HIGH_PROFILES = frozenset(
    {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}
)


def _skip_scaling_list(reader: _BitReader, size: int) -> None:
    last, next_scale = 8, 8
    for _ in range(size):
        if next_scale != 0:
            next_scale = (last + reader.se() + 256) % 256
        last = next_scale if next_scale != 0 else last


def parse_sps_dimensions(sps: bytes) -> tuple[int, int]:
    """Return (width, height) in pixels from an SPS NAL unit (header included).

    Only informational for the WebRTC offer, but it means the twin reports the
    stream's real resolution instead of a configured guess.
    """
    if not sps or nal_type(sps) != NAL_SPS:
        raise ValueError("not an SPS NAL unit")
    r = _BitReader(_rbsp(sps[1:]))
    profile_idc = r.bits(8)
    r.bits(16)  # constraint flags, reserved bits, level_idc
    r.ue()  # seq_parameter_set_id
    chroma_format_idc = 1
    separate_colour_plane = 0
    if profile_idc in _HIGH_PROFILES:
        chroma_format_idc = r.ue()
        if chroma_format_idc == 3:
            separate_colour_plane = r.bit()
        r.ue()  # bit_depth_luma_minus8
        r.ue()  # bit_depth_chroma_minus8
        r.bit()  # qpprime_y_zero_transform_bypass_flag
        if r.bit():  # seq_scaling_matrix_present_flag
            for i in range(8 if chroma_format_idc != 3 else 12):
                if r.bit():
                    _skip_scaling_list(r, 16 if i < 6 else 64)
    r.ue()  # log2_max_frame_num_minus4
    poc_type = r.ue()
    if poc_type == 0:
        r.ue()  # log2_max_pic_order_cnt_lsb_minus4
    elif poc_type == 1:
        r.bit()  # delta_pic_order_always_zero_flag
        r.se()  # offset_for_non_ref_pic
        r.se()  # offset_for_top_to_bottom_field
        for _ in range(r.ue()):
            r.se()
    r.ue()  # max_num_ref_frames
    r.bit()  # gaps_in_frame_num_value_allowed_flag
    width_mbs = r.ue() + 1
    height_map_units = r.ue() + 1
    frame_mbs_only = r.bit()
    if not frame_mbs_only:
        r.bit()  # mb_adaptive_frame_field_flag
    r.bit()  # direct_8x8_inference_flag
    crop_left = crop_right = crop_top = crop_bottom = 0
    if r.bit():  # frame_cropping_flag
        crop_left, crop_right, crop_top, crop_bottom = r.ue(), r.ue(), r.ue(), r.ue()

    chroma_array_type = 0 if separate_colour_plane else chroma_format_idc
    if chroma_array_type == 0:
        crop_unit_x, crop_unit_y = 1, 2 - frame_mbs_only
    else:
        sub_width = 1 if chroma_format_idc == 3 else 2
        sub_height = 2 if chroma_format_idc == 1 else 1
        crop_unit_x, crop_unit_y = sub_width, sub_height * (2 - frame_mbs_only)
    width = width_mbs * 16 - crop_unit_x * (crop_left + crop_right)
    height = (2 - frame_mbs_only) * height_map_units * 16 - crop_unit_y * (
        crop_top + crop_bottom
    )
    if width <= 0 or height <= 0:
        raise ValueError("SPS cropping yields an empty picture")
    return width, height


__all__ = [
    "ANNEXB_START_CODE",
    "NAL_AUD",
    "NAL_IDR",
    "NAL_PPS",
    "NAL_SEI",
    "NAL_SLICE",
    "NAL_SPS",
    "VCL_SLICES",
    "ParameterSetCache",
    "first_mb_in_slice",
    "join_annexb",
    "nal_type",
    "parse_sps_dimensions",
    "split_annexb",
]
