"""CollisionAwareKinematics: limit overrides, SRDF pairing, world gates, IK policy.

Runs against ``fixtures/collision_arm.urdf`` -- six revolute joints with
primitive collision shapes, so the collision model builds anywhere pinocchio
does and no assertion can pass vacuously because meshes failed to resolve -- and
its reviewed ``collision_arm.srdf`` (adjacent links only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip when pinocchio can't be imported for ANY reason (native libs may be
# absent in CI); pytest>=8.2 narrowed importorskip() to ModuleNotFoundError.
try:
    import pinocchio  # noqa: F401
except ImportError:
    pytest.skip("pinocchio unavailable", allow_module_level=True)

try:
    import coal  # noqa: F401
except ImportError:  # pragma: no cover - depends on the pinocchio build
    try:
        import hppfcl  # noqa: F401
    except ImportError:
        pytest.skip("coal/hppfcl unavailable", allow_module_level=True)

from cyberwave.driver.kinematics.arm.base import BaseKinematicsManipulator
from cyberwave.driver.kinematics.arm.collision import (
    COLLISION_PAIR_SOURCE_HEURISTIC,
    COLLISION_PAIR_SOURCE_NONE,
    COLLISION_PAIR_SOURCE_SRDF,
    COLLISION_PAIR_SOURCE_SRDF_MISMATCH,
    FLOOR_THICKNESS_M,
    CollisionAwareKinematics,
    CollisionModelUnavailable,
    floor_slab_primitive,
    mesh_package_dirs,
    split_by_parent_joint,
)
from cyberwave.driver.kinematics.arm.config import ArmKinematicsConfig
from cyberwave.driver.motion.scene import Pose, Primitive

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_URDF = _FIXTURES / "collision_arm.urdf"
_SRDF = _FIXTURES / "collision_arm.srdf"
_ARM = tuple(f"joint{i}" for i in range(1, 7))

_HOME = {j: 0.0 for j in _ARM}
#: Elbow folded back and wrist bent into the upper arm: a real self-collision
#: that the reviewed (adjacent-only) matrix must still catch.
_FOLDED = {**_HOME, "joint3": 2.6, "joint5": 1.8}
#: The forearm driven down through the mounting plane, touching nothing else.
_THROUGH_FLOOR = {**_HOME, "joint2": 2.0, "joint3": 1.0}
#: A well-conditioned working posture, clear of everything.
_READY = {**_HOME, "joint2": 0.6, "joint3": 0.9, "joint5": 0.6}


def _cfg(urdf: Path | str = _URDF, **kwargs) -> ArmKinematicsConfig:
    # 200 iterations converges exactly as often as 1000 on this fixture and
    # keeps the multi-seed corpus fast.
    kwargs.setdefault("ik_max_iter", 200)
    return ArmKinematicsConfig(
        urdf_path=str(urdf),
        ee_frame="tool_link",
        arm_joints=_ARM,
        locked_joints=("finger_joint",),
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([0.0, 0.0, 1.0])},
        **kwargs,
    )


def _kin(**kwargs) -> CollisionAwareKinematics:
    kwargs.setdefault("srdf_path", str(_SRDF))
    return CollisionAwareKinematics(_cfg(), **kwargs)


def _reachable_cases(kin, n: int, seed: int = 13):
    rng = np.random.default_rng(seed)
    limits = kin.joint_limits()
    lo = np.array([limits[j][0] for j in _ARM])
    hi = np.array([limits[j][1] for j in _ARM])

    def d(v):
        return {j: float(x) for j, x in zip(_ARM, v)}

    return [(d(rng.uniform(lo, hi)), kin.fk(d(rng.uniform(lo, hi)))) for _ in range(n)]


def _unreachable_target(kin):
    """Far outside the workspace, so every seed is tried and all of them fail."""
    target = kin.fk(_HOME).copy()
    target[0, 3] += 5.0
    return target


# ── joint limit overrides ───────────────────────────────────────────────────


def test_no_override_keeps_the_model_limits() -> None:
    limits = _kin(reject_self_collision=False).joint_limits()
    assert limits["joint2"] == pytest.approx((-2.0, 2.0))
    assert limits["joint6"] == pytest.approx((-2.6, 2.6))


def test_an_override_only_narrows_and_never_widens() -> None:
    kin = _kin(
        reject_self_collision=False,
        joint_limit_overrides={"joint2": (-5.0, 1.0), "joint5": (-1.0, 9.0)},
    )
    limits = kin.joint_limits()
    assert limits["joint2"] == pytest.approx((-2.0, 1.0))
    assert limits["joint5"] == pytest.approx((-1.0, 1.8))
    assert limits["joint1"] == pytest.approx((-2.618, 2.618))


def test_an_override_for_an_unknown_joint_is_ignored() -> None:
    kin = _kin(reject_self_collision=False, joint_limit_overrides={"nope": (0, 1)})
    assert kin.joint_limits()["joint1"] == pytest.approx((-2.618, 2.618))


def test_an_override_leaving_an_empty_range_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="empty range"):
        _kin(reject_self_collision=False, joint_limit_overrides={"joint2": (3.0, 4.0)})


def test_ik_solutions_respect_the_active_override() -> None:
    kin = _kin(
        reject_self_collision=False,
        joint_limit_overrides={"joint2": (-1.0, 1.0), "joint3": (-1.5, 1.5)},
    )
    limits = kin.joint_limits()
    solved = 0
    for start, target in _reachable_cases(kin, 25, seed=3):
        solution = kin.ik(start, target)
        if solution is None:
            continue
        solved += 1
        for name in _ARM:
            assert limits[name][0] - 1e-6 <= solution[name] <= limits[name][1] + 1e-6
    assert solved > 0, "no target solved; the test proves nothing"


def test_a_thin_subclass_can_map_named_profiles_onto_overrides() -> None:
    """The shape a driver uses: a profile table, and nothing else."""

    profiles = {"urdf": None, "tight": ((-1.0, 1.0),) * 6}

    class NamedProfileKinematics(CollisionAwareKinematics):
        def __init__(self, config, *, joint_limit_profile: str = "urdf", **kwargs):
            if joint_limit_profile not in profiles:
                raise ValueError(f"unknown joint_limit_profile {joint_limit_profile!r}")
            bounds = profiles[joint_limit_profile]
            super().__init__(
                config,
                joint_limit_overrides=(
                    None if bounds is None else dict(zip(config.arm_joints, bounds))
                ),
                **kwargs,
            )
            self.joint_limit_profile = joint_limit_profile

    tight = NamedProfileKinematics(_cfg(), joint_limit_profile="tight")
    assert tight.joint_limits()["joint2"] == pytest.approx((-1.0, 1.0))
    with pytest.raises(ValueError, match="unknown joint_limit_profile"):
        NamedProfileKinematics(_cfg(), joint_limit_profile="whatever")


# ── the reviewed SRDF matrix ────────────────────────────────────────────────


def test_the_srdf_produces_a_valid_reviewed_collision_model() -> None:
    kin = _kin()
    assert kin.collision_pair_source == COLLISION_PAIR_SOURCE_SRDF
    assert kin.collision_model_valid is True
    assert kin.self_collision_available is True


def test_the_srdf_model_clears_the_home_configuration() -> None:
    """addAllCollisionPairs() alone reports adjacent links touching at home."""
    raw = _kin(srdf_path="", fallback_pairs=lambda geom: [])
    assert raw.self_collision_available is False
    assert _kin().in_self_collision(_HOME) is False


@pytest.mark.parametrize("sample", [_HOME, _READY])
def test_the_srdf_model_clears_working_postures(sample) -> None:
    assert _kin().in_self_collision(sample) is False


def test_the_srdf_model_still_detects_a_real_self_collision() -> None:
    """Clearing home must not have been achieved by disabling everything."""
    kin = _kin()
    assert kin.in_self_collision(_FOLDED) is True
    assert kin.collision_reason(_FOLDED) == "self-collision"


def test_random_sampling_finds_self_collisions() -> None:
    kin = _kin()
    limits = kin.joint_limits()
    rng = np.random.default_rng(7)
    lo = np.array([limits[j][0] for j in _ARM])
    hi = np.array([limits[j][1] for j in _ARM])
    hits = sum(
        1
        for _ in range(300)
        if kin.in_self_collision(
            {j: float(v) for j, v in zip(_ARM, rng.uniform(lo, hi))}
        )
    )
    assert hits > 0, "no sampled configuration collided; the gate detects nothing"


def test_an_srdf_naming_absent_links_is_not_a_valid_model(tmp_path: Path) -> None:
    """removeCollisionPairs silently ignores unknown links; this check must not."""
    payload_srdf = tmp_path / "payload.srdf"
    payload_srdf.write_text(
        _SRDF.read_text().replace(
            "</robot>",
            '  <disable_collisions link1="link6" link2="wrist_camera" '
            'reason="Adjacent"/>\n</robot>',
        )
    )
    kin = _kin(srdf_path=str(payload_srdf))
    assert kin.collision_pair_source == COLLISION_PAIR_SOURCE_SRDF_MISMATCH
    assert kin.collision_model_valid is False
    # Still a live (if untrusted) gate: it reports, it does not go silent.
    assert kin.self_collision_available is True
    with pytest.raises(CollisionModelUnavailable):
        _kin(srdf_path=str(payload_srdf), require_collision_model=True)


def test_a_required_collision_model_fails_closed_without_an_srdf() -> None:
    """ "Could not check" and "checked and clear" must not look the same."""
    with pytest.raises(CollisionModelUnavailable):
        _kin(srdf_path="", require_collision_model=True)


def test_a_required_collision_model_fails_closed_when_geometry_is_missing(
    tmp_path: Path,
) -> None:
    """A URDF whose meshes cannot be resolved builds no collision geometry."""
    meshless = tmp_path / "meshless.urdf"
    meshless.write_text(
        _URDF.read_text().replace(
            '<box size="0.05 0.05 0.25"/>',
            '<mesh filename="package://nowhere/meshes/link2.stl"/>',
        )
    )
    degraded = CollisionAwareKinematics(_cfg(meshless), srdf_path=str(_SRDF))
    assert degraded.self_collision_available is False
    assert degraded.in_self_collision(_FOLDED) is False  # unknown, not cleared
    with pytest.raises(CollisionModelUnavailable):
        CollisionAwareKinematics(
            _cfg(meshless), srdf_path=str(_SRDF), require_collision_model=True
        )


def test_the_srdf_is_accepted_as_required_when_it_matches() -> None:
    assert _kin(require_collision_model=True).collision_model_valid is True


def test_rejection_can_be_disabled() -> None:
    kin = _kin(reject_self_collision=False)
    assert kin.collision_pair_source == COLLISION_PAIR_SOURCE_NONE
    assert kin.self_collision_available is False
    assert kin.in_collision(_FOLDED) is False


# ── the development fallback ────────────────────────────────────────────────


def test_without_srdf_or_fallback_there_is_no_self_collision_gate() -> None:
    kin = _kin(srdf_path="")
    assert kin.collision_pair_source == COLLISION_PAIR_SOURCE_NONE
    assert kin.self_collision_available is False


def test_a_fallback_pairing_is_live_but_never_reported_valid() -> None:
    """A development aid that claims validity is how an unreviewed model ships."""
    kin = _kin(
        srdf_path="",
        fallback_pairs=split_by_parent_joint(distal_joint_min=4, proximal_joint_max=2),
    )
    assert kin.collision_pair_source == COLLISION_PAIR_SOURCE_HEURISTIC
    assert kin.self_collision_available is True
    assert kin.collision_model_valid is False
    assert kin.in_self_collision(_HOME) is False


def test_split_by_parent_joint_pairs_distal_against_proximal() -> None:
    kin = _kin(
        srdf_path="",
        fallback_pairs=split_by_parent_joint(distal_joint_min=4, proximal_joint_max=2),
    )
    geom = kin._geom_model
    for pair in geom.collisionPairs:
        first = geom.geometryObjects[pair.first].parentJoint
        second = geom.geometryObjects[pair.second].parentJoint
        assert {first >= 4, second >= 4} == {True, False}
        assert min(first, second) <= 2


# ── the floor slab ──────────────────────────────────────────────────────────


def test_the_floor_gate_is_off_unless_requested() -> None:
    kin = _kin()
    assert kin.floor_collision_available is False
    assert kin.in_floor_collision(_THROUGH_FLOOR) is False


def test_the_floor_gate_does_not_depend_on_the_self_collision_gate() -> None:
    """`reject_self_collision=False` must not silently disable the floor."""
    kin = _kin(reject_self_collision=False, reject_floor_collision=True)
    assert kin.floor_collision_available is True
    assert kin.self_collision_available is False, (
        "loaded geometry with no pairs is not a live self-collision gate"
    )
    assert kin.in_floor_collision(_THROUGH_FLOOR) is True
    assert kin.in_self_collision(_THROUGH_FLOOR) is False
    assert "floor plane" in (kin.collision_reason(_THROUGH_FLOOR) or "")


def test_root_geometry_is_excluded_and_every_movable_geometry_is_checked() -> None:
    """The base is bolted to the surface; the gripper is the part that hits it."""
    kin = _kin(reject_floor_collision=True)
    geom = kin._geom_model
    floor = kin._floor_geom_id
    partners = set()
    for index in kin._floor_pair_indices:
        pair = geom.collisionPairs[index]
        other = pair.first if pair.second == floor else pair.second
        partners.add(geom.geometryObjects[other].name)
    movable = {
        g.name
        for g in geom.geometryObjects
        if g.parentJoint != 0 and g.name != kin.FLOOR_GEOM_NAME
    }
    assert partners == movable
    assert "base_link_0" not in partners
    assert {"tool_link_0", "finger_link_0"} <= partners


@pytest.mark.parametrize("requested", [0.0, -0.15, 0.05])
def test_the_slab_top_face_sits_exactly_on_floor_z(requested: float) -> None:
    """A slab centred on floor_z would raise the surface by half its thickness."""
    kin = _kin(reject_floor_collision=True, floor_z=requested)
    placement = kin._geom_model.geometryObjects[kin._floor_geom_id].placement
    assert float(placement.translation[2]) == pytest.approx(
        requested - FLOOR_THICKNESS_M / 2.0
    )
    assert kin.floor_z == pytest.approx(requested)


def test_the_slab_size_is_configurable() -> None:
    kin = _kin(
        reject_floor_collision=True, floor_half_extent_m=0.5, floor_thickness_m=0.04
    )
    obj = kin._geom_model.geometryObjects[kin._floor_geom_id]
    assert np.allclose(2.0 * obj.geometry.halfSide, [1.0, 1.0, 0.04])
    assert float(obj.placement.translation[2]) == pytest.approx(-0.02)


def test_the_floor_does_not_perturb_the_reviewed_srdf_matrix() -> None:
    """World pairs are added AFTER the SRDF pairing, so the matrix is untouched."""
    without = _kin()
    with_floor = _kin(reject_floor_collision=True)
    assert with_floor.collision_pair_source == COLLISION_PAIR_SOURCE_SRDF
    reviewed = len(without._geom_model.collisionPairs)
    total = len(with_floor._geom_model.collisionPairs)
    assert total - len(with_floor._floor_pair_indices) == reviewed


def test_self_collision_does_not_answer_for_the_floor() -> None:
    kin = _kin(reject_floor_collision=True)
    assert kin.in_collision(_THROUGH_FLOOR) is True
    assert kin.in_self_collision(_THROUGH_FLOOR) is False


def test_the_reason_names_the_plane_and_its_height() -> None:
    kin = _kin(reject_floor_collision=True, floor_z=0.05)
    assert "0.0500" in (kin.collision_reason(_THROUGH_FLOOR) or "")


def test_both_scenes_are_named_when_both_are_hit() -> None:
    kin = _kin(reject_floor_collision=True, floor_z=0.3)
    reason = kin.collision_reason(_FOLDED) or ""
    assert "self-collision" in reason and "floor plane" in reason


def test_home_and_ready_are_clear_of_the_floor() -> None:
    kin = _kin(reject_floor_collision=True)
    for posture in (_HOME, _READY):
        assert kin.in_collision(posture) is False


def test_the_floor_primitive_reproduces_the_slab_the_gate_checks() -> None:
    """Exposed as a scene object, the startup floor must be the same geometry."""
    gate = _kin(reject_floor_collision=True, floor_z=-0.015)
    floor = gate.floor_primitive()
    assert floor is not None
    assert floor == floor_slab_primitive(-0.015)
    assert floor.id == "floor" and floor.kind == "box"
    assert floor.dimensions == pytest.approx((4.0, 4.0, 0.02))
    assert floor.pose.position[2] == pytest.approx(-0.025)

    as_scene = _kin(obstacles=(floor,))
    for posture in (_HOME, _READY, _THROUGH_FLOOR):
        assert as_scene.in_collision(posture) == gate.in_collision(posture)
    assert "obstacle 'floor'" in (as_scene.collision_reason(_THROUGH_FLOOR) or "")


def test_no_floor_primitive_when_the_gate_is_off() -> None:
    assert _kin().floor_primitive() is None


# ── runtime obstacles ───────────────────────────────────────────────────────


def _primitive(identifier, kind, dimensions, position, orientation=(0, 0, 0, 1)):
    return Primitive(
        identifier,
        kind,
        tuple(float(v) for v in dimensions),
        Pose(
            tuple(float(v) for v in position),
            tuple(float(v) for v in orientation),
        ),
    )


def test_a_runtime_scene_replaces_the_startup_floor() -> None:
    far = _primitive("far", "box", (0.1, 0.1, 0.02), (2.0, 2.0, -0.01))
    kin = _kin(
        reject_self_collision=False, reject_floor_collision=True, obstacles=(far,)
    )
    assert kin.floor_collision_available is False
    assert kin.obstacle_collision_available is True
    assert kin.in_collision(_THROUGH_FLOOR) is False


def test_an_obstacle_is_named_in_the_reason_and_is_not_self_collision() -> None:
    cell = _primitive("cell", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0))
    kin = _kin(reject_self_collision=False, obstacles=(cell,))
    assert "obstacle 'cell'" in (kin.collision_reason(_HOME) or "")
    assert kin.in_self_collision(_HOME) is False
    assert kin.in_collision(_HOME) is True


@pytest.mark.parametrize(
    ("kind", "dimensions"), [("sphere", (0.05,)), ("cylinder", (0.2, 0.05))]
)
def test_other_primitive_shapes_are_accepted(kind, dimensions) -> None:
    # Placed on the arm's own axis at shoulder height, so it is genuinely hit.
    obstacle = _primitive("fixture", kind, dimensions, (0.0, 0.0, 0.3))
    kin = _kin(reject_self_collision=False, obstacles=(obstacle,))
    assert kin.obstacle_collision_available
    assert "obstacle 'fixture'" in (kin.collision_reason(_HOME) or "")


def test_obstacle_orientation_is_honoured() -> None:
    """Same bar, same centre: it reaches the arm only when yawed toward it."""
    beside = (0.2, 0.0, 0.3)
    bar = (0.02, 0.5, 0.02)  # long along y: parallel to the arm, 0.2 m away
    identity = (0.0, 0.0, 0.0, 1.0)
    # 90 degrees about z (x, y, z, w): the long extent now points along x and
    # spans the arm's vertical column at x = 0.
    yawed = (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4)))
    clear = _primitive("bar", "box", bar, beside, identity)
    rotated = _primitive("bar", "box", bar, beside, yawed)
    assert (
        _kin(reject_self_collision=False, obstacles=(clear,)).in_collision(_HOME)
        is False
    )
    assert (
        _kin(reject_self_collision=False, obstacles=(rotated,)).in_collision(_HOME)
        is True
    )


def test_an_explicitly_empty_scene_removes_every_world_object() -> None:
    kin = _kin(reject_floor_collision=True, obstacles=())
    assert kin.floor_collision_available is False
    assert kin.obstacle_collision_available is False
    assert kin.collision_model_valid is True


def test_obstacles_on_unloadable_geometry_report_the_gate_unavailable(
    tmp_path: Path,
) -> None:
    """Geometry that fails to load degrades every gate, and says so.

    Callers that need "checked" rather than "not refused" read
    ``obstacle_collision_available`` (or set ``require_collision_model``).
    """
    meshless = tmp_path / "meshless.urdf"
    meshless.write_text(
        _URDF.read_text().replace(
            '<box size="0.05 0.05 0.25"/>',
            '<mesh filename="package://nowhere/meshes/link2.stl"/>',
        )
    )
    cell = _primitive("cell", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0))
    kin = CollisionAwareKinematics(_cfg(meshless), obstacles=(cell,))
    assert kin.obstacle_collision_available is False
    assert kin.in_collision(_HOME) is False  # unknown, not cleared
    assert kin.collision_model_valid is False


def test_a_failed_self_pairing_takes_the_world_scene_down_with_it(
    tmp_path: Path,
) -> None:
    """A half-built model is not a gate: obstacles then refuse to build at all."""
    unreadable = tmp_path / "broken.srdf"
    unreadable.write_text("<robot><disable_collisions link1=")
    cell = _primitive("cell", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0))
    with pytest.raises(CollisionModelUnavailable):
        _kin(srdf_path=str(unreadable), obstacles=(cell,))


# ── IK policy: multi-seed restart ───────────────────────────────────────────


def test_multi_seed_beats_single_seed_substantially() -> None:
    kin = _kin(reject_self_collision=False)
    cases = _reachable_cases(kin, 40)

    single = sum(
        1 for s, t in cases if BaseKinematicsManipulator.ik(kin, s, t) is not None
    )
    multi = sum(1 for s, t in cases if kin.ik(s, t) is not None)

    assert single / len(cases) < 0.6, f"baseline unexpectedly good ({single})"
    assert multi / len(cases) > 0.85, f"multi-seed only reached {multi}/{len(cases)}"


def test_first_seed_is_the_caller_start_so_nudges_stay_local() -> None:
    """Put the zero seed before ``start`` and this fails immediately."""
    kin = _kin(reject_self_collision=False)
    rng = np.random.default_rng(4)
    limits = kin.joint_limits()
    lo = np.array([limits[j][0] for j in _ARM])
    hi = np.array([limits[j][1] for j in _ARM])
    compared = 0
    for _ in range(30):
        q = np.clip(rng.uniform(lo, hi) * 0.6, lo, hi)
        start = {j: float(v) for j, v in zip(_ARM, q)}
        target = kin.apply_translation(kin.fk(start), {"forward": 0.01}, frame="tool")
        baseline = BaseKinematicsManipulator.ik(kin, start, target)
        solution = kin.ik(start, target)
        if baseline is None or solution is None:
            continue
        compared += 1
        assert (
            max(abs(solution[j] - start[j]) for j in _ARM)
            <= max(abs(baseline[j] - start[j]) for j in _ARM) + 1e-9
        )
    assert compared > 15, f"only {compared} comparable nudges; the test proves little"


def test_seeding_is_deterministic() -> None:
    kin = _kin(reject_self_collision=False)
    cases = _reachable_cases(kin, 15)
    assert [kin.ik(s, t) for s, t in cases] == [kin.ik(s, t) for s, t in cases]


def test_one_seed_attempt_reproduces_the_base_solver() -> None:
    kin = _kin(reject_self_collision=False)
    kin.IK_SEED_ATTEMPTS = 1
    cases = _reachable_cases(kin, 20)
    got = [kin.ik(s, t) is not None for s, t in cases]
    base = [BaseKinematicsManipulator.ik(kin, s, t) is not None for s, t in cases]
    assert got == base


def test_seed_sequence_is_start_zero_then_samples_within_limits() -> None:
    kin = _kin(reject_self_collision=False)
    start = {j: 0.1 * (i + 1) for i, j in enumerate(_ARM)}
    seeds = kin._ik_seeds(start)
    assert len(seeds) == kin.IK_SEED_ATTEMPTS
    assert seeds[0] == start and seeds[1] == _HOME
    limits = kin.joint_limits()
    for seed in seeds[2:]:
        for j in _ARM:
            assert limits[j][0] <= seed[j] <= limits[j][1]


def test_sampled_seeds_get_the_shorter_iteration_budget(monkeypatch) -> None:
    kin = _kin(reject_self_collision=False)
    budgets = []
    original = BaseKinematicsManipulator.ik

    def recording(self, start, target, **kwargs):
        budgets.append(kwargs.get("max_iter"))
        return original(self, start, target, **kwargs)

    monkeypatch.setattr(BaseKinematicsManipulator, "ik", recording)
    unreachable = _unreachable_target(kin)

    assert kin.ik(_HOME, unreachable) is None
    assert budgets == [None, None] + [kin.IK_RANDOM_SEED_MAX_ITER] * (
        kin.IK_SEED_ATTEMPTS - 2
    )

    budgets.clear()
    kin.ik(_HOME, unreachable, max_iter=50)
    assert set(budgets) == {50}


def _count_base_solves(monkeypatch) -> list[dict]:
    seen: list[dict] = []
    real = BaseKinematicsManipulator.ik

    def _record(self, seed, target, **kwargs):
        seen.append(dict(seed))
        return real(self, seed, target, **kwargs)

    monkeypatch.setattr(BaseKinematicsManipulator, "ik", _record)
    return seen


def test_restart_false_tries_exactly_one_seed(monkeypatch) -> None:
    kin = _kin(reject_self_collision=False)
    start = {j: 0.1 * (i + 1) for i, j in enumerate(_ARM)}
    seen = _count_base_solves(monkeypatch)

    assert kin.ik(start, _unreachable_target(kin), restart=False) is None
    assert seen == [start]


def test_restart_true_remains_the_default(monkeypatch) -> None:
    kin = _kin(reject_self_collision=False)
    start = {j: 0.1 * (i + 1) for i, j in enumerate(_ARM)}
    seen = _count_base_solves(monkeypatch)

    assert kin.ik(start, _unreachable_target(kin)) is None
    assert len(seen) == kin.IK_SEED_ATTEMPTS > 1
    assert seen[0] == start


def test_restart_false_still_solves_a_nearby_target() -> None:
    kin = _kin(reject_self_collision=False)
    target = kin.apply_translation(kin.fk(_READY), {"forward": 0.01}, frame="tool")
    solution = kin.ik(_READY, target, restart=False)
    assert solution is not None
    assert max(abs(solution[j] - _READY[j]) for j in _ARM) < 0.5


# ── unwrapping ──────────────────────────────────────────────────────────────


def test_unwrapping_snaps_a_full_turn_back_when_it_is_legal() -> None:
    kin = _kin(reject_self_collision=False)
    reference = {**_HOME, "joint1": -2.5}
    wrapped = {**_HOME, "joint1": -2.5 + 2.0 * np.pi}
    assert kin._unwrap_toward(wrapped, reference)["joint1"] == pytest.approx(-2.5)

    # A shift that would escape the limits is refused, not clamped.
    stuck = {**_HOME, "joint1": 2.6}
    assert kin._unwrap_toward(stuck, {**_HOME, "joint1": -2.6})["joint1"] == 2.6


def test_unwrapping_ignores_joints_without_a_reference() -> None:
    kin = _kin(reject_self_collision=False)
    solution = {**_HOME, "joint1": 1.0}
    assert kin._unwrap_toward(solution, {}) == solution


def test_unwrapped_solutions_stay_in_limits_and_on_target() -> None:
    kin = _kin(reject_self_collision=False)
    limits = kin.joint_limits()
    for start, target in _reachable_cases(kin, 20, seed=21):
        solution = kin.ik(start, target)
        if solution is None:
            continue
        for j in _ARM:
            assert limits[j][0] - 1e-6 <= solution[j] <= limits[j][1] + 1e-6
        assert np.allclose(kin.fk(solution)[:3, 3], target[:3, 3], atol=1e-3)


# ── the collision gate reaching IK ──────────────────────────────────────────


def test_colliding_solutions_are_not_returned() -> None:
    kin = _kin(reject_floor_collision=True)
    for start, target in _reachable_cases(kin, 25, seed=91):
        solution = kin.ik(start, target)
        if solution is None:
            continue
        assert kin.in_collision(solution) is False


def test_ik_refuses_a_target_that_needs_a_link_below_the_plane() -> None:
    gate_off = _kin(reject_self_collision=False)
    gate_on = _kin(reject_self_collision=False, reject_floor_collision=True)
    target = gate_off.fk(_THROUGH_FLOOR)

    assert gate_off.ik(_THROUGH_FLOOR, target, restart=False) is not None
    assert gate_on.ik(_THROUGH_FLOOR, target, restart=False) is None
    assert "floor" in (gate_on.last_ik_refusal or "")


def test_diagnose_ik_names_the_scene_that_refused() -> None:
    kin = _kin(reject_self_collision=False, reject_floor_collision=True, floor_z=0.6)
    # Everything above the 0.6 m plane is unreachable with the arm below it.
    target = kin.fk(_HOME)
    assert kin.ik(_READY, target) is None
    diagnosis = kin.diagnose_ik(_READY, target) or ""
    assert diagnosis.startswith(kin.IK_REFUSED_PREFIX)
    assert "floor" in diagnosis


def test_diagnose_ik_distinguishes_no_convergence_from_a_refusal() -> None:
    kin = _kin(reject_floor_collision=True)
    assert kin.diagnose_ik(_HOME, _unreachable_target(kin)) == kin.IK_NO_CONVERGENCE
    assert kin.ik(_HOME, _unreachable_target(kin)) is None
    assert kin.last_ik_refusal is None


def test_diagnose_ik_is_none_for_a_solvable_target() -> None:
    kin = _kin(reject_floor_collision=True)
    assert kin.diagnose_ik(_HOME, kin.fk(_READY)) is None


def test_last_ik_refusal_is_reset_by_the_next_call() -> None:
    kin = _kin(reject_self_collision=False, reject_floor_collision=True, floor_z=0.6)
    assert kin.ik(_READY, kin.fk(_HOME)) is None
    assert kin.last_ik_refusal is not None
    kin.ik(_HOME, _unreachable_target(kin))
    assert kin.last_ik_refusal is None


# ── position-only IK ────────────────────────────────────────────────────────


def test_position_ik_reaches_the_target_position() -> None:
    kin = _kin(reject_floor_collision=True)
    target = kin.fk(_READY)[:3, 3]
    solution = kin.ik_position(_HOME, target)
    assert solution is not None
    assert np.allclose(kin.fk(solution)[:3, 3], target, atol=1e-3)
    assert kin.in_collision(solution) is False


def test_position_ik_restart_false_tries_exactly_one_seed(monkeypatch) -> None:
    kin = _kin(reject_self_collision=False)
    seen: list[dict] = []
    real = BaseKinematicsManipulator.ik_position

    def _record(self, seed, target, **kwargs):
        seen.append(dict(seed))
        return real(self, seed, target, **kwargs)

    monkeypatch.setattr(BaseKinematicsManipulator, "ik_position", _record)
    far = np.array([5.0, 0.0, 0.0])
    assert kin.ik_position(_READY, far, restart=False) is None
    assert seen == [_READY]
    seen.clear()
    assert kin.ik_position(_READY, far) is None
    assert len(seen) == kin.IK_SEED_ATTEMPTS


def test_position_ik_refuses_through_the_floor_and_diagnoses_it() -> None:
    kin = _kin(reject_self_collision=False, reject_floor_collision=True, floor_z=0.6)
    target = kin.fk(_HOME)[:3, 3]
    assert kin.ik_position(_READY, target) is None
    assert "floor" in (kin.last_ik_refusal or "")
    assert "floor" in (kin.diagnose_ik_position(_READY, target) or "")
    assert (
        kin.diagnose_ik_position(_READY, np.array([5.0, 0.0, 0.0]))
        == kin.IK_NO_CONVERGENCE
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def test_mesh_package_dirs_offers_every_ancestor() -> None:
    dirs = mesh_package_dirs("/a/b/pkg/urdf/robot.urdf")
    assert dirs[:4] == ["/a/b/pkg/urdf", "/a/b/pkg", "/a/b", "/a"]


def test_the_controller_can_build_this_kinematics() -> None:
    """The seam drivers use: ArmController(build_kinematics=...)."""
    from cyberwave.driver.kinematics.arm.controller import ArmController

    controller = ArmController(
        _cfg(),
        build_kinematics=lambda config: CollisionAwareKinematics(
            config, srdf_path=str(_SRDF)
        ),
    )
    assert isinstance(controller.kinematics, CollisionAwareKinematics)
    assert controller.kinematics.collision_model_valid is True


def test_rebuilds_reuse_the_loaded_geometry_without_mutating_it(monkeypatch) -> None:
    # Scene and payload changes rebuild the solver; re-reading and re-meshing
    # every link each time cost hundreds of ms per command on a meshed arm.
    from cyberwave.driver.kinematics.arm import collision as collision_module

    monkeypatch.setattr(collision_module, "_LOADED_GEOMETRY", {})
    calls = []
    real = pinocchio.buildGeomFromUrdf

    def counting(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(pinocchio, "buildGeomFromUrdf", counting)
    first = _kin()
    cell = _primitive("cell", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0))
    scene = _kin(obstacles=(cell,))
    again = _kin()

    assert len(calls) == 1
    (cached,) = collision_module._LOADED_GEOMETRY.values()
    links = len(first._geom_model.geometryObjects)
    assert len(cached.geometryObjects) == links
    assert len(cached.collisionPairs) == 0
    assert len(scene._geom_model.geometryObjects) == links + 1
    assert "obstacle 'cell'" in (scene.collision_reason(_HOME) or "")
    assert again.collision_reason(_HOME) is None
    assert again.in_collision(_HOME) is first.in_collision(_HOME)
