"""Tests for HookContext dataclass fields and defaults."""

from cyberwave.workers.context import HookContext


def test_context_required_fields() -> None:
    ctx = HookContext(timestamp=1234567890.123, channel="frames/front")
    assert ctx.timestamp == 1234567890.123
    assert ctx.channel == "frames/front"


def test_context_defaults() -> None:
    ctx = HookContext(timestamp=0.0, channel="imu")
    assert ctx.sensor_name == "default"
    assert ctx.twin_uuid is None
    assert ctx.metadata == {}


def test_context_all_fields() -> None:
    meta = {"width": 640, "height": 480}
    ctx = HookContext(
        timestamp=1.0,
        channel="depth/front",
        sensor_name="front",
        twin_uuid="abc-123",
        metadata=meta,
    )
    assert ctx.sensor_name == "front"
    assert ctx.twin_uuid == "abc-123"
    assert ctx.metadata == meta


def test_context_metadata_isolation() -> None:
    """Each instance gets its own metadata dict."""
    a = HookContext(timestamp=0.0, channel="a")
    b = HookContext(timestamp=0.0, channel="b")
    a.metadata["key"] = "value"
    assert "key" not in b.metadata


def test_context_equality() -> None:
    a = HookContext(timestamp=1.0, channel="imu", twin_uuid="x")
    b = HookContext(timestamp=1.0, channel="imu", twin_uuid="x")
    assert a == b


def test_context_repr_contains_channel() -> None:
    ctx = HookContext(timestamp=0.0, channel="gps")
    assert "gps" in repr(ctx)


def test_context_twin_uuid_none_vs_string() -> None:
    """twin_uuid=None is distinct from twin_uuid=''."""
    ctx_none = HookContext(timestamp=0.0, channel="imu")
    ctx_empty = HookContext(timestamp=0.0, channel="imu", twin_uuid="")
    assert ctx_none.twin_uuid is None
    assert ctx_empty.twin_uuid == ""
    assert ctx_none != ctx_empty
