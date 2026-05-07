"""Tests for HookRegistry decorator registration.

Covers all 15 single-channel typed decorators, the generic ``on_data``
decorator, the ``on_synchronized`` multi-channel decorator, registry
clearing, duplicate detection, thread safety, and frozen dataclass
invariants.
"""

import threading

import pytest

from cyberwave.workers.hooks import WILDCARD_SENSOR, HookRegistration, HookRegistry


# ── Fixtures ──────────────────────────────────────────────────────

TWIN_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture()
def registry() -> HookRegistry:
    return HookRegistry()


# ── Individual channel decorators ─────────────────────────────────


def test_on_frame_registers_hook(registry: HookRegistry) -> None:
    """Without an explicit ``sensor=``, ``on_frame`` registers a wildcard hook
    that matches whatever sensor name the driver publishes under."""

    @registry.on_frame(TWIN_UUID)
    def handler(sample, ctx):
        pass

    assert len(registry.hooks) == 1
    h = registry.hooks[0]
    assert h.channel == "frames"
    assert h.twin_uuid == TWIN_UUID
    assert h.hook_type == "frame"
    assert h.sensor_name == WILDCARD_SENSOR
    assert h.is_wildcard_sensor
    assert h.callback is handler


def test_on_frame_explicit_wildcard(registry: HookRegistry) -> None:
    """``sensor="*"`` is equivalent to omitting the kwarg."""

    @registry.on_frame(TWIN_UUID, sensor="*")
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "frames"
    assert h.sensor_name == WILDCARD_SENSOR
    assert h.is_wildcard_sensor


def test_on_frame_custom_sensor(registry: HookRegistry) -> None:
    @registry.on_frame(TWIN_UUID, sensor="front")
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "frames/front"
    assert h.sensor_name == "front"
    assert not h.is_wildcard_sensor


def test_on_frame_with_fps(registry: HookRegistry) -> None:
    @registry.on_frame(TWIN_UUID, fps=15)
    def handler(sample, ctx):
        pass

    assert registry.hooks[0].options == {"fps": 15}


def test_on_frame_fps_zero_preserved(registry: HookRegistry) -> None:
    """fps=0 should be stored (not silently dropped as falsy)."""

    @registry.on_frame(TWIN_UUID, fps=0)
    def handler(sample, ctx):
        pass

    assert registry.hooks[0].options == {"fps": 0}


def test_on_frame_fps_none_omitted(registry: HookRegistry) -> None:
    @registry.on_frame(TWIN_UUID, fps=None)
    def handler(sample, ctx):
        pass

    assert registry.hooks[0].options == {}


def test_on_depth_registers_hook(registry: HookRegistry) -> None:
    @registry.on_depth(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "depth"
    assert h.sensor_name == WILDCARD_SENSOR
    assert h.hook_type == "depth"


def test_on_depth_custom_sensor(registry: HookRegistry) -> None:
    @registry.on_depth(TWIN_UUID, sensor="rear")
    def handler(sample, ctx):
        pass

    assert registry.hooks[0].channel == "depth/rear"
    assert registry.hooks[0].sensor_name == "rear"


def test_on_audio_registers_hook(registry: HookRegistry) -> None:
    @registry.on_audio(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "audio"
    assert h.sensor_name == WILDCARD_SENSOR
    assert h.hook_type == "audio"


def test_on_pointcloud_registers_hook(registry: HookRegistry) -> None:
    @registry.on_pointcloud(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "pointcloud"
    assert h.sensor_name == WILDCARD_SENSOR
    assert h.hook_type == "pointcloud"


def test_on_imu_registers_hook(registry: HookRegistry) -> None:
    @registry.on_imu(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "imu"
    assert h.hook_type == "imu"


def test_on_force_torque_registers_hook(registry: HookRegistry) -> None:
    @registry.on_force_torque(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "force_torque"
    assert h.hook_type == "force_torque"


def test_on_joint_states_registers_hook(registry: HookRegistry) -> None:
    @registry.on_joint_states(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "joint_states"
    assert h.hook_type == "joint_states"


def test_on_attitude_registers_hook(registry: HookRegistry) -> None:
    @registry.on_attitude(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "attitude"
    assert h.hook_type == "attitude"


def test_on_gps_registers_hook(registry: HookRegistry) -> None:
    @registry.on_gps(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "gps"
    assert h.hook_type == "gps"


def test_on_end_effector_pose_registers_hook(registry: HookRegistry) -> None:
    @registry.on_end_effector_pose(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "end_effector_pose"
    assert h.hook_type == "end_effector_pose"


def test_on_gripper_state_registers_hook(registry: HookRegistry) -> None:
    @registry.on_gripper_state(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "gripper_state"
    assert h.hook_type == "gripper_state"


def test_on_map_registers_hook(registry: HookRegistry) -> None:
    @registry.on_map(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "map"
    assert h.hook_type == "map"


def test_on_battery_registers_hook(registry: HookRegistry) -> None:
    @registry.on_battery(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "battery"
    assert h.hook_type == "battery"


def test_on_temperature_registers_hook(registry: HookRegistry) -> None:
    @registry.on_temperature(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "temperature"
    assert h.hook_type == "temperature"


def test_on_lidar_registers_hook(registry: HookRegistry) -> None:
    @registry.on_lidar(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "lidar"
    assert h.sensor_name == WILDCARD_SENSOR
    assert h.hook_type == "lidar"


def test_on_lidar_custom_sensor(registry: HookRegistry) -> None:
    @registry.on_lidar(TWIN_UUID, sensor="top")
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "lidar/top"
    assert h.sensor_name == "top"


# ── Generic on_data decorator ─────────────────────────────────────


def test_on_data_custom_channel(registry: HookRegistry) -> None:
    @registry.on_data(TWIN_UUID, "lidar/top")
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == "lidar/top"
    assert h.hook_type == "data"
    assert h.twin_uuid == TWIN_UUID


# ── on_synchronized decorator ─────────────────────────────────────


def test_on_synchronized_registers_group(registry: HookRegistry) -> None:
    @registry.on_synchronized(TWIN_UUID, ["frames/front", "depth/front"])
    def handler(samples, ctx):
        pass

    assert len(registry.synchronized_groups) == 1
    g = registry.synchronized_groups[0]
    assert g.channels == ("frames/front", "depth/front")
    assert g.twin_uuid == TWIN_UUID
    assert g.tolerance_ms == 50.0
    assert g.callback is handler


def test_on_synchronized_custom_tolerance(registry: HookRegistry) -> None:
    @registry.on_synchronized(
        TWIN_UUID, ["frames/front", "depth/front"], tolerance_ms=100.0
    )
    def handler(samples, ctx):
        pass

    assert registry.synchronized_groups[0].tolerance_ms == 100.0


def test_on_synchronized_channels_are_tuple(registry: HookRegistry) -> None:
    """Channels stored as tuple (immutable) even when passed as list."""

    @registry.on_synchronized(TWIN_UUID, ["a", "b"])
    def handler(samples, ctx):
        pass

    g = registry.synchronized_groups[0]
    assert isinstance(g.channels, tuple)


# ── All 16 decorators produce valid registrations ─────────────────


_STATELESS_DECORATORS = [
    ("on_imu", "imu", "imu"),
    ("on_force_torque", "force_torque", "force_torque"),
    ("on_joint_states", "joint_states", "joint_states"),
    ("on_attitude", "attitude", "attitude"),
    ("on_gps", "gps", "gps"),
    ("on_end_effector_pose", "end_effector_pose", "end_effector_pose"),
    ("on_gripper_state", "gripper_state", "gripper_state"),
    ("on_map", "map", "map"),
    ("on_battery", "battery", "battery"),
    ("on_temperature", "temperature", "temperature"),
]

_SENSOR_DECORATORS = [
    ("on_frame", "frames", "frame"),
    ("on_depth", "depth", "depth"),
    ("on_audio", "audio", "audio"),
    ("on_pointcloud", "pointcloud", "pointcloud"),
    ("on_lidar", "lidar", "lidar"),
]


@pytest.mark.parametrize("method,expected_channel,expected_type", _STATELESS_DECORATORS)
def test_stateless_decorator_registers(
    registry: HookRegistry, method: str, expected_channel: str, expected_type: str
) -> None:
    decorator_fn = getattr(registry, method)

    @decorator_fn(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == expected_channel
    assert h.hook_type == expected_type


@pytest.mark.parametrize("method,expected_channel,expected_type", _SENSOR_DECORATORS)
def test_sensor_decorator_registers(
    registry: HookRegistry, method: str, expected_channel: str, expected_type: str
) -> None:
    decorator_fn = getattr(registry, method)

    @decorator_fn(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    assert h.channel == expected_channel
    assert h.hook_type == expected_type


# ── Decorator returns original function ───────────────────────────


def test_decorator_returns_original_function(registry: HookRegistry) -> None:
    def my_handler(sample, ctx):
        return 42

    decorated = registry.on_frame(TWIN_UUID)(my_handler)
    assert decorated is my_handler
    assert decorated(None, None) == 42


def test_all_decorators_return_original_function(registry: HookRegistry) -> None:
    """Every decorator variant must return the original function unmodified."""

    def stub(sample, ctx):
        pass

    for method_name in [
        "on_frame",
        "on_depth",
        "on_audio",
        "on_pointcloud",
        "on_imu",
        "on_force_torque",
        "on_joint_states",
        "on_attitude",
        "on_gps",
        "on_end_effector_pose",
        "on_gripper_state",
        "on_map",
        "on_battery",
        "on_temperature",
        "on_lidar",
    ]:
        decorator_fn = getattr(registry, method_name)
        result = decorator_fn(TWIN_UUID)(stub)
        assert result is stub, f"{method_name} did not return the original function"

    result = registry.on_data(TWIN_UUID, "custom")(stub)
    assert result is stub

    result = registry.on_synchronized(TWIN_UUID, ["a", "b"])(stub)
    assert result is stub


# ── Multiple hooks on the same channel ────────────────────────────


def test_multiple_hooks_same_channel_different_callbacks(
    registry: HookRegistry,
) -> None:
    @registry.on_frame(TWIN_UUID, sensor="front")
    def handler_a(sample, ctx):
        pass

    @registry.on_frame(TWIN_UUID, sensor="front")
    def handler_b(sample, ctx):
        pass

    assert len(registry.hooks) == 2
    assert registry.hooks[0].callback is handler_a
    assert registry.hooks[1].callback is handler_b
    assert registry.hooks[0].channel == registry.hooks[1].channel == "frames/front"


# ── Duplicate registration warning ────────────────────────────────


def test_duplicate_registration_warns(registry: HookRegistry) -> None:
    def handler(sample, ctx):
        pass

    registry.on_frame(TWIN_UUID)(handler)

    with pytest.warns(UserWarning, match="Duplicate hook registration"):
        registry.on_frame(TWIN_UUID)(handler)

    assert len(registry.hooks) == 2


# ── Frozen dataclass invariants ───────────────────────────────────


def test_hook_registration_is_frozen(registry: HookRegistry) -> None:
    @registry.on_imu(TWIN_UUID)
    def handler(sample, ctx):
        pass

    h = registry.hooks[0]
    with pytest.raises(AttributeError):
        h.channel = "tampered"  # type: ignore[misc]


def test_synchronized_group_is_frozen(registry: HookRegistry) -> None:
    @registry.on_synchronized(TWIN_UUID, ["a", "b"])
    def handler(samples, ctx):
        pass

    g = registry.synchronized_groups[0]
    with pytest.raises(AttributeError):
        g.tolerance_ms = 999.0  # type: ignore[misc]


# ── __repr__ ──────────────────────────────────────────────────────


def test_hook_registration_repr(registry: HookRegistry) -> None:
    @registry.on_frame(TWIN_UUID, fps=30)
    def handler(sample, ctx):
        pass

    r = repr(registry.hooks[0])
    assert "'frames'" in r
    assert TWIN_UUID in r
    assert "frame" in r
    assert "fps" in r


def test_hook_registration_repr_no_options(registry: HookRegistry) -> None:
    @registry.on_imu(TWIN_UUID)
    def handler(sample, ctx):
        pass

    r = repr(registry.hooks[0])
    assert "options" not in r


# ── Registry clear ────────────────────────────────────────────────


def test_registry_clear(registry: HookRegistry) -> None:
    @registry.on_frame(TWIN_UUID)
    def handler(sample, ctx):
        pass

    @registry.on_synchronized(TWIN_UUID, ["frames/default"])
    def sync_handler(samples, ctx):
        pass

    assert len(registry.hooks) == 1
    assert len(registry.synchronized_groups) == 1

    registry.clear()

    assert registry.hooks == []
    assert registry.synchronized_groups == []


# ── hooks property returns a copy ─────────────────────────────────


def test_hooks_property_returns_copy(registry: HookRegistry) -> None:
    @registry.on_frame(TWIN_UUID)
    def handler(sample, ctx):
        pass

    hooks_copy = registry.hooks
    hooks_copy.clear()
    assert len(registry.hooks) == 1


# ── Thread safety ─────────────────────────────────────────────────


def test_concurrent_registration(registry: HookRegistry) -> None:
    """Multiple threads registering hooks concurrently must not lose entries."""
    n_threads = 8
    hooks_per_thread = 50
    barrier = threading.Barrier(n_threads)

    def register_many(thread_id: int) -> None:
        barrier.wait()
        for i in range(hooks_per_thread):

            def handler(sample, ctx, _tid=thread_id, _i=i):
                pass

            registry.on_data(TWIN_UUID, f"ch/{thread_id}/{i}")(handler)

    threads = [
        threading.Thread(target=register_many, args=(t,)) for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(registry.hooks) == n_threads * hooks_per_thread
