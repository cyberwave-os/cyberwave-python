"""Contract tests for key-expression parsing and validation.

Acceptance criteria from CYB-1553:
  * Key-expression parsing/validation is test-covered.
  * Contract tests for valid/invalid keys and payloads.
  * Python and native publishers can follow the same contract.
"""

from __future__ import annotations

import uuid as uuid_mod

import pytest

from cyberwave.data.exceptions import ChannelError
from cyberwave.data.keys import (
    COMMAND_CHANNELS,
    LATEST_VALUE_CHANNELS,
    STREAM_CHANNELS,
    WELL_KNOWN_CHANNELS,
    KeyExpression,
    build_key,
    build_keys,
    build_wildcard,
    channel_from_key,
    is_valid_key,
    parse_key,
)

SAMPLE_UUID = "550e8400-e29b-41d4-a716-446655440000"
SAMPLE_UUID_UPPER = "550E8400-E29B-41D4-A716-446655440000"


# ── Channel sets ─────────────────────────────────────────────────────


class TestChannelSets:
    def test_stream_channels_are_subset(self) -> None:
        assert STREAM_CHANNELS <= WELL_KNOWN_CHANNELS

    def test_latest_value_channels_are_subset(self) -> None:
        assert LATEST_VALUE_CHANNELS <= WELL_KNOWN_CHANNELS

    def test_no_overlap(self) -> None:
        assert not (STREAM_CHANNELS & LATEST_VALUE_CHANNELS)

    def test_command_channels_are_subset(self) -> None:
        assert COMMAND_CHANNELS <= WELL_KNOWN_CHANNELS

    def test_union_equals_all(self) -> None:
        assert STREAM_CHANNELS | LATEST_VALUE_CHANNELS | COMMAND_CHANNELS == WELL_KNOWN_CHANNELS

    def test_expected_stream_channels(self) -> None:
        expected = {"frames", "depth", "audio", "pointcloud", "imu", "force_torque"}
        assert STREAM_CHANNELS == expected

    def test_expected_latest_value_channels(self) -> None:
        expected = {
            "joint_states",
            "position",
            "attitude",
            "gps",
            "end_effector_pose",
            "gripper_state",
            "map",
            "battery",
            "temperature",
            "telemetry",
        }
        assert LATEST_VALUE_CHANNELS == expected


# ── build_key ────────────────────────────────────────────────────────


class TestBuildKey:
    def test_basic(self) -> None:
        key = build_key(SAMPLE_UUID, "frames")
        assert key == f"cw/{SAMPLE_UUID}/data/frames"

    def test_with_sensor_name(self) -> None:
        key = build_key(SAMPLE_UUID, "frames", "default")
        assert key == f"cw/{SAMPLE_UUID}/data/frames/default"

    def test_custom_prefix(self) -> None:
        key = build_key(SAMPLE_UUID, "depth", prefix="myapp")
        assert key == f"myapp/{SAMPLE_UUID}/data/depth"

    def test_uppercase_uuid_accepted(self) -> None:
        key = build_key(SAMPLE_UUID_UPPER, "position")
        assert SAMPLE_UUID_UPPER in key

    @pytest.mark.parametrize("channel", sorted(WELL_KNOWN_CHANNELS))
    def test_all_well_known_channels(self, channel: str) -> None:
        key = build_key(SAMPLE_UUID, channel)
        assert channel in key

    def test_custom_channel(self) -> None:
        key = build_key(SAMPLE_UUID, "custom_sensor")
        assert "custom_sensor" in key

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ChannelError, match="UUID"):
            build_key("not-a-uuid", "frames")

    def test_invalid_channel_raises(self) -> None:
        with pytest.raises(ChannelError, match="channel segment"):
            build_key(SAMPLE_UUID, "Invalid-Channel!")

    def test_invalid_sensor_name_raises(self) -> None:
        with pytest.raises(ChannelError, match="sensor name"):
            build_key(SAMPLE_UUID, "frames", "BAD NAME")

    def test_channel_starting_with_digit_raises(self) -> None:
        with pytest.raises(ChannelError, match="channel segment"):
            build_key(SAMPLE_UUID, "3d_scan")

    def test_channel_with_uppercase_raises(self) -> None:
        with pytest.raises(ChannelError, match="channel segment"):
            build_key(SAMPLE_UUID, "Frames")

    def test_empty_channel_raises(self) -> None:
        with pytest.raises(ChannelError, match="channel segment"):
            build_key(SAMPLE_UUID, "")


# ── parse_key ────────────────────────────────────────────────────────


class TestParseKey:
    def test_basic(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/frames")
        assert ke.prefix == "cw"
        assert ke.twin_uuid == SAMPLE_UUID
        assert ke.channel == "frames"
        assert ke.sensor_name is None

    def test_with_sensor_name(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/depth/default")
        assert ke.channel == "depth"
        assert ke.sensor_name == "default"

    def test_custom_prefix(self) -> None:
        ke = parse_key(f"myapp/{SAMPLE_UUID}/data/audio")
        assert ke.prefix == "myapp"

    def test_roundtrip_build_parse(self) -> None:
        key = build_key(SAMPLE_UUID, "joint_states", "left_arm")
        ke = parse_key(key)
        assert ke.twin_uuid == SAMPLE_UUID
        assert ke.channel == "joint_states"
        assert ke.sensor_name == "left_arm"

    def test_str_roundtrip(self) -> None:
        ke = KeyExpression(
            prefix="cw", twin_uuid=SAMPLE_UUID, channel="gps", sensor_name="main"
        )
        assert parse_key(str(ke)) == ke

    def test_too_short_raises(self) -> None:
        with pytest.raises(ChannelError, match="too short"):
            parse_key("cw/only/two")

    def test_missing_data_segment_raises(self) -> None:
        with pytest.raises(ChannelError, match="data"):
            parse_key(f"cw/{SAMPLE_UUID}/notdata/frames")

    def test_too_many_segments_raises(self) -> None:
        with pytest.raises(ChannelError, match="too many"):
            parse_key(f"cw/{SAMPLE_UUID}/data/frames/default/extra")

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ChannelError, match="UUID"):
            parse_key("cw/not-a-uuid/data/frames")

    def test_invalid_channel_raises(self) -> None:
        with pytest.raises(ChannelError, match="channel segment"):
            parse_key(f"cw/{SAMPLE_UUID}/data/BAD!")


# ── KeyExpression properties ─────────────────────────────────────────


class TestKeyExpressionProperties:
    def test_is_well_known_true(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/frames")
        assert ke.is_well_known is True

    def test_is_well_known_false(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/custom_channel")
        assert ke.is_well_known is False

    def test_is_stream(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/frames")
        assert ke.is_stream is True
        assert ke.is_latest_value is False

    def test_is_latest_value(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/joint_states")
        assert ke.is_stream is False
        assert ke.is_latest_value is True

    def test_custom_channel_neither(self) -> None:
        ke = parse_key(f"cw/{SAMPLE_UUID}/data/custom")
        assert ke.is_stream is False
        assert ke.is_latest_value is False


# ── build_wildcard ───────────────────────────────────────────────────


class TestBuildWildcard:
    def test_full_wildcard(self) -> None:
        w = build_wildcard()
        assert w == "cw/*/data/**"

    def test_twin_wildcard(self) -> None:
        w = build_wildcard(twin_uuid=SAMPLE_UUID)
        assert w == f"cw/{SAMPLE_UUID}/data/**"

    def test_twin_channel_wildcard(self) -> None:
        w = build_wildcard(twin_uuid=SAMPLE_UUID, channel="frames")
        assert w == f"cw/{SAMPLE_UUID}/data/frames/**"

    def test_custom_prefix(self) -> None:
        w = build_wildcard(prefix="dev")
        assert w == "dev/*/data/**"

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ChannelError, match="UUID"):
            build_wildcard(twin_uuid="bad")

    def test_invalid_channel_raises(self) -> None:
        with pytest.raises(ChannelError, match="channel"):
            build_wildcard(channel="BAD!")


# ── channel_from_key ─────────────────────────────────────────────────


class TestChannelFromKey:
    def test_extracts_channel(self) -> None:
        key = f"cw/{SAMPLE_UUID}/data/temperature"
        assert channel_from_key(key) == "temperature"

    def test_with_sensor(self) -> None:
        key = f"cw/{SAMPLE_UUID}/data/frames/wrist"
        assert channel_from_key(key) == "frames"


# ── build_keys (batch) ───────────────────────────────────────────────


class TestBuildKeys:
    def test_batch_build(self) -> None:
        channels = ["frames", "joint_states", "battery"]
        result = build_keys(SAMPLE_UUID, channels)
        assert len(result) == 3
        for ch in channels:
            assert ch in result
            assert parse_key(result[ch]).channel == ch


# ── is_valid_key ─────────────────────────────────────────────────────


class TestIsValidKey:
    def test_valid(self) -> None:
        assert is_valid_key(f"cw/{SAMPLE_UUID}/data/frames/default")

    def test_invalid(self) -> None:
        assert not is_valid_key("garbage")

    def test_invalid_uuid(self) -> None:
        assert not is_valid_key("cw/bad-uuid/data/frames")


# ── Integration: random UUIDs ────────────────────────────────────────


class TestRandomUUIDs:
    """Ensure key building/parsing works with freshly generated UUIDs."""

    @pytest.mark.parametrize("_", range(10))
    def test_random_uuid_roundtrip(self, _: int) -> None:
        u = str(uuid_mod.uuid4())
        key = build_key(u, "telemetry", "main")
        ke = parse_key(key)
        assert ke.twin_uuid == u
        assert ke.channel == "telemetry"
        assert ke.sensor_name == "main"
