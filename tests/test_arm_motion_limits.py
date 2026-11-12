"""The physical-to-controller velocity resolver: quantization, bounds and refusals.

Every number that decides how fast an arm moves comes out of
``resolve_motion_limits``. These cases fix the two properties that matter on
hardware: the derived speed scale never asks for more than the operator
requested, and every failure is a closed-set refusal rather than a fallback.

The "commissioned" case mirrors a real arm (URDF ``[5,5,5,5,5,3]`` rad/s against
a 3 rad/s controller with a 1 % vendor resolution) with synthetic data, so the
SDK carries no robot's tables.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cyberwave.driver.kinematics.arm.limits import (
    PERCENT_QUANTIZATION,
    PLANNING_VELOCITY_SAFETY,
    CartesianLimitBases,
    ControllerVelocityProfile,
    MotionLimitError,
    MotionLimitRefusal,
    ResolvedMotionLimits,
    SpeedScaleQuantization,
    intersect_joint_limits,
    planning_limits_from,
    resolve_motion_limits,
    urdf_joint_limits,
    urdf_joint_velocity_limits,
)

JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FIRMWARE = "FW-1.0"
VENDOR_PERCENT = SpeedScaleQuantization(
    steps=100, min_steps=1, label="vendor_speed_percent"
)


def _profile(
    firmware: str = "SYNTHETIC", **velocities: object
) -> ControllerVelocityProfile:
    return ControllerVelocityProfile(
        firmware_version=firmware, max_velocity_rad_s=velocities, evidence="test probe"
    )


PROFILES = {FIRMWARE: _profile(FIRMWARE, **{joint: 3.0 for joint in JOINTS})}


def _write_urdf(tmp_path: Path, velocities: dict[str, str], name: str = "arm") -> str:
    joints = "".join(
        f'<joint name="{joint}" type="revolute">'
        f'<limit lower="-3" upper="3" effort="100"{value}/></joint>'
        for joint, value in velocities.items()
    )
    path = tmp_path / f"{name}.urdf"
    path.write_text(f"<robot name='synthetic'>{joints}</robot>")
    return str(path)


@pytest.fixture
def urdf(tmp_path: Path) -> str:
    """Five fast joints and a slower wrist, like a typical 6-DoF description."""
    velocities = {joint: ' velocity="5"' for joint in JOINTS}
    velocities["joint6"] = ' velocity="3"'
    return _write_urdf(tmp_path, velocities)


def _resolve(requested: object, urdf: str, **kwargs) -> ResolvedMotionLimits:
    kwargs.setdefault("joints", JOINTS)
    kwargs.setdefault("firmware_version", FIRMWARE)
    kwargs.setdefault("profiles", PROFILES)
    kwargs.setdefault("quantization", VENDOR_PERCENT)
    return resolve_motion_limits(requested, urdf_path=urdf, **kwargs)


def _refusal(requested: object, urdf: str, **kwargs) -> MotionLimitRefusal:
    with pytest.raises(MotionLimitError) as excinfo:
        _resolve(requested, urdf, **kwargs)
    return excinfo.value.reason


# ── a commissioned configuration ────────────────────────────────────────────


def test_a_ceiling_resolves_to_its_floored_percentage(urdf: str) -> None:
    limits = _resolve(0.3, urdf)

    assert limits.speed_scale_steps == 10
    assert limits.speed_scale == pytest.approx(0.1)
    assert all(limits.effective_velocity_rad_s[j] == pytest.approx(0.3) for j in JOINTS)
    assert all(limits.planner_velocity_rad_s[j] == pytest.approx(0.27) for j in JOINTS)
    assert limits.firmware_version == FIRMWARE
    assert limits.evidence == "test probe"
    assert limits.joints == JOINTS


def test_the_urdf_is_read_per_joint_not_broadcast(urdf: str) -> None:
    assert urdf_joint_velocity_limits(urdf, JOINTS) == {
        "joint1": 5.0,
        "joint2": 5.0,
        "joint3": 5.0,
        "joint4": 5.0,
        "joint5": 5.0,
        "joint6": 3.0,
    }


def test_only_the_requested_joints_are_read(urdf: str) -> None:
    """A gripper or passive joint in the description is out of scope."""
    assert urdf_joint_velocity_limits(urdf, ("joint1", "joint6")) == {
        "joint1": 5.0,
        "joint6": 3.0,
    }


def test_the_startup_record_names_the_request_the_evidence_and_the_derivations(
    urdf: str,
) -> None:
    record = _resolve(0.3, urdf).startup_record()

    for fragment in (
        "max_joint_velocity_rad_s=0.3",
        f"firmware={FIRMWARE}",
        "vendor_speed_percent=10",
        "controller_max=",
        "urdf_max=",
        "effective=",
        "planner=",
        "evidence=test probe",
    ):
        assert fragment in record


def test_results_are_read_only(urdf: str) -> None:
    limits = _resolve(0.3, urdf)
    with pytest.raises(TypeError):
        limits.planner_velocity_rad_s["joint1"] = 9.0  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        limits.speed_scale = 1.0  # type: ignore[misc]


# ── quantization ────────────────────────────────────────────────────────────


def test_a_value_just_above_a_percentage_boundary_quantizes_downward(urdf: str) -> None:
    # 0.359 rad/s sits between 11 % (0.33) and 12 % (0.36) of 3 rad/s.
    limits = _resolve(0.359, urdf)

    assert limits.speed_scale_steps == 11
    assert all(limits.effective_velocity_rad_s[j] <= 0.359 + 1e-12 for j in JOINTS)


@pytest.mark.parametrize("requested", [0.03, 0.31, 0.7, 1.4, 2.999, 3.0])
def test_no_effective_cap_ever_exceeds_the_request_or_the_urdf(
    urdf: str, requested: float
) -> None:
    limits = _resolve(requested, urdf)
    described = limits.urdf_velocity_rad_s

    for joint in JOINTS:
        effective = limits.effective_velocity_rad_s[joint]
        assert 0.0 < effective <= requested + 1e-12
        assert effective <= described[joint] + 1e-12
        assert limits.planner_velocity_rad_s[joint] == pytest.approx(
            min(requested, described[joint], effective) * PLANNING_VELOCITY_SAFETY
        )


def test_the_smallest_representable_ceiling_is_one_step_and_below_it_is_refused(
    urdf: str,
) -> None:
    assert _resolve(0.03, urdf).speed_scale_steps == 1
    assert _refusal(0.029, urdf) is MotionLimitRefusal.BELOW_MINIMUM_RESOLUTION


def test_a_controller_minimum_above_one_step_is_honoured(urdf: str) -> None:
    """Some controllers refuse very low percentages; never round up to them."""
    coarse = SpeedScaleQuantization(steps=100, min_steps=5)
    assert _resolve(0.15, urdf, quantization=coarse).speed_scale_steps == 5
    assert (
        _refusal(0.14, urdf, quantization=coarse)
        is MotionLimitRefusal.BELOW_MINIMUM_RESOLUTION
    )


def test_without_quantization_the_scale_is_continuous(urdf: str) -> None:
    limits = _resolve(0.359, urdf, quantization=None)

    assert limits.speed_scale_steps is None
    assert limits.speed_scale == pytest.approx(0.359 / 3.0)
    assert all(
        limits.effective_velocity_rad_s[j] == pytest.approx(0.359) for j in JOINTS
    )
    assert "speed_scale=0.1197" in limits.startup_record()
    # Continuous has no resolution floor: a tiny request is representable.
    assert _resolve(0.001, urdf, quantization=None).speed_scale > 0.0


def test_the_percent_preset_is_one_percent_resolution() -> None:
    assert (PERCENT_QUANTIZATION.steps, PERCENT_QUANTIZATION.min_steps) == (100, 1)


@pytest.mark.parametrize("steps,min_steps", [(0, 1), (100, 0), (100, 101)])
def test_a_nonsensical_quantization_is_rejected(steps: int, min_steps: int) -> None:
    with pytest.raises(ValueError):
        SpeedScaleQuantization(steps=steps, min_steps=min_steps)


def test_a_request_above_the_highest_effective_ceiling_is_refused_not_ignored(
    urdf: str,
) -> None:
    # 4 rad/s would silently resolve to 100 % = 3 rad/s, which is not what was asked.
    assert _refusal(4.0, urdf) is MotionLimitRefusal.ABOVE_HARDWARE_CEILING
    # The boundary itself is applicable: joint6's URDF limit of 3.0 == H.
    assert _resolve(3.0, urdf).speed_scale_steps == 100


# ── invalid inputs ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "requested", [0.0, -0.3, math.nan, math.inf, True, False, None, "fast", [0.3]]
)
def test_non_positive_non_finite_and_boolean_requests_are_refused(
    urdf: str, requested: object
) -> None:
    assert _refusal(requested, urdf) is MotionLimitRefusal.INVALID_REQUEST


def test_missing_and_unknown_firmware_are_two_different_refusals(urdf: str) -> None:
    assert _refusal(0.3, urdf, firmware_version=None) is MotionLimitRefusal.NO_FIRMWARE
    assert _refusal(0.3, urdf, firmware_version="") is MotionLimitRefusal.NO_FIRMWARE
    assert (
        _refusal(0.3, urdf, firmware_version="FW-9.9")
        is MotionLimitRefusal.UNKNOWN_FIRMWARE
    )


def test_the_firmware_hint_tells_the_operator_how_to_read_it(urdf: str) -> None:
    with pytest.raises(MotionLimitError) as excinfo:
        _resolve(0.3, urdf, firmware_version=None, firmware_hint="run the arm probe")
    assert "run the arm probe" in excinfo.value.detail


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, math.nan])
def test_an_incomplete_controller_profile_is_refused(urdf: str, bad: object) -> None:
    velocities: dict[str, object] = {joint: 3.0 for joint in JOINTS}
    velocities["joint4"] = bad
    profiles = {"SYNTHETIC": _profile(**velocities)}

    assert (
        _refusal(0.3, urdf, firmware_version="SYNTHETIC", profiles=profiles)
        is MotionLimitRefusal.INVALID_CONTROLLER_PROFILE
    )


def test_a_profile_missing_a_joint_entirely_is_refused(urdf: str) -> None:
    profiles = {"SYNTHETIC": _profile(**{j: 3.0 for j in JOINTS if j != "joint6"})}

    assert (
        _refusal(0.3, urdf, firmware_version="SYNTHETIC", profiles=profiles)
        is MotionLimitRefusal.INVALID_CONTROLLER_PROFILE
    )


def test_an_unreadable_urdf_is_refused(tmp_path: Path, urdf: str) -> None:
    missing = str(tmp_path / "nope.urdf")
    assert _refusal(0.3, missing) is MotionLimitRefusal.URDF_UNREADABLE

    malformed = tmp_path / "bad.urdf"
    malformed.write_text("<robot><joint name=")
    assert _refusal(0.3, str(malformed)) is MotionLimitRefusal.URDF_UNREADABLE


@pytest.mark.parametrize(
    "attribute", ["", ' velocity="0"', ' velocity="-2"', ' velocity="nan"']
)
def test_a_missing_or_malformed_urdf_velocity_limit_is_refused(
    tmp_path: Path, attribute: str
) -> None:
    velocities = {joint: ' velocity="5"' for joint in JOINTS}
    velocities["joint3"] = attribute
    urdf = _write_urdf(tmp_path, velocities)

    assert _refusal(0.3, urdf) is MotionLimitRefusal.URDF_MISSING_VELOCITY_LIMIT


def test_a_urdf_missing_an_arm_joint_entirely_is_refused(tmp_path: Path) -> None:
    urdf = _write_urdf(tmp_path, {j: ' velocity="5"' for j in JOINTS if j != "joint5"})

    assert _refusal(0.3, urdf) is MotionLimitRefusal.URDF_MISSING_VELOCITY_LIMIT


def test_an_empty_joint_list_is_a_programming_error(urdf: str) -> None:
    with pytest.raises(ValueError):
        _resolve(0.3, urdf, joints=())


# ── per-joint, not scalar ───────────────────────────────────────────────────


def test_a_heterogeneous_controller_profile_is_bounded_by_its_strictest_joint(
    tmp_path: Path,
) -> None:
    urdf = _write_urdf(tmp_path, {j: ' velocity="10"' for j in JOINTS})
    uniform = {"UNIFORM": _profile("UNIFORM", **{j: 6.0 for j in JOINTS})}
    limits = _resolve(0.5, urdf, firmware_version="UNIFORM", profiles=uniform)

    assert limits.speed_scale_steps == 8  # floor(100 * 0.5 / 6)
    assert all(
        limits.effective_velocity_rad_s[j] == pytest.approx(0.48) for j in JOINTS
    )

    velocities = {j: 6.0 for j in JOINTS}
    velocities["joint2"] = 1.0
    mixed = {"MIXED": _profile("MIXED", **velocities)}
    limits = _resolve(0.5, urdf, firmware_version="MIXED", profiles=mixed)

    # joint2 needs 50 %, joint1 only 8 %; the global scale must respect the
    # strictest ratio, so the fast joints are capped well under the request.
    assert limits.speed_scale_steps == 8
    assert limits.effective_velocity_rad_s["joint2"] == pytest.approx(0.08)
    assert limits.effective_velocity_rad_s["joint1"] == pytest.approx(0.48)


def test_a_urdf_velocity_below_the_hardware_maximum_constrains_the_scale(
    tmp_path: Path,
) -> None:
    # Manual commands and planning both ride on the derived scale, so a tight
    # URDF joint must pull the WHOLE command down, not just the planner.
    velocities = {joint: ' velocity="5"' for joint in JOINTS}
    velocities["joint6"] = ' velocity="0.15"'
    urdf = _write_urdf(tmp_path, velocities)
    limits = _resolve(0.3, urdf)

    assert limits.speed_scale_steps == 5  # floor(100 * 0.15 / 3)
    assert all(
        limits.effective_velocity_rad_s[j] == pytest.approx(0.15) for j in JOINTS
    )
    assert limits.planner_velocity_rad_s["joint1"] == pytest.approx(0.135)


def test_the_planning_safety_factor_is_the_only_planner_headroom(urdf: str) -> None:
    limits = _resolve(0.3, urdf, planning_safety=1.0)

    assert all(limits.planner_velocity_rad_s[j] == pytest.approx(0.3) for j in JOINTS)


# ── planner configuration bodies ────────────────────────────────────────────


def test_planning_limits_carry_per_joint_velocities_and_a_physical_acceleration(
    urdf: str,
) -> None:
    limits = _resolve(0.3, urdf)
    bodies = planning_limits_from(limits, joint_acceleration_rad_s2=2.0)

    assert set(bodies.joint_limits["joint_limits"]) == set(JOINTS)
    joint1 = bodies.joint_limits["joint_limits"]["joint1"]
    assert joint1 == {
        "has_velocity_limits": True,
        "max_velocity": pytest.approx(0.27),
        "has_acceleration_limits": True,
        # Physical, NOT scaled by the 10 % speed fraction.
        "max_acceleration": 2.0,
        # Pilz's own deceleration bound, mirrored and negative.
        "has_deceleration_limits": True,
        "max_deceleration": -2.0,
    }
    assert bodies.cartesian_limits == {}


def test_cartesian_limits_scale_with_the_resolved_fraction(urdf: str) -> None:
    limits = _resolve(0.3, urdf)
    bases = CartesianLimitBases(
        max_trans_vel_m_s=0.25, max_trans_acc_m_s2=0.5, max_rot_vel_rad_s=1.0
    )
    cartesian = planning_limits_from(
        limits, joint_acceleration_rad_s2=2.0, cartesian_bases=bases
    ).cartesian_limits["cartesian_limits"]

    assert cartesian == {
        "max_trans_vel": pytest.approx(0.025),
        "max_trans_acc": pytest.approx(0.05),
        "max_trans_dec": pytest.approx(0.05),
        "max_rot_vel": pytest.approx(0.1),
    }


# ── position limits ─────────────────────────────────────────────────────────


def test_urdf_joint_limits_reads_lower_and_upper(tmp_path: Path) -> None:
    path = tmp_path / "limits.urdf"
    path.write_text(
        "<robot name='r'>"
        '<joint name="a" type="revolute"><limit lower="-1" upper="2" velocity="1"/></joint>'
        '<joint name="b" type="continuous"><limit velocity="1"/></joint>'
        '<joint name="c" type="fixed"/>'
        "</robot>"
    )
    assert urdf_joint_limits(str(path)) == {"a": (-1.0, 2.0)}


def test_intersection_never_widens_the_description() -> None:
    described = {"a": (-1.0, 1.0), "b": (-2.0, 2.0)}
    overrides = {"a": (-5.0, 0.5), "b": (-1.5, 9.0)}

    assert intersect_joint_limits(described, overrides, ("a", "b")) == {
        "a": (-1.0, 0.5),
        "b": (-1.5, 2.0),
    }


def test_no_override_means_the_description() -> None:
    described = {"a": (-1.0, 1.0)}
    assert intersect_joint_limits(described, None, ("a",)) == {"a": (-1.0, 1.0)}
    assert intersect_joint_limits(described, {}, ("a",)) == {"a": (-1.0, 1.0)}


def test_an_unbounded_joint_is_refused_not_defaulted() -> None:
    with pytest.raises(ValueError, match="unbounded"):
        intersect_joint_limits({"a": (-1.0, 1.0)}, None, ("a", "b"))


def test_an_override_leaving_an_empty_range_is_refused() -> None:
    with pytest.raises(ValueError, match="empty range"):
        intersect_joint_limits({"a": (-1.0, 1.0)}, {"a": (2.0, 3.0)}, ("a",))
