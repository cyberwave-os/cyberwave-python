"""cyberwave.test.driver_testing: discovery, the section registry, and a full run on a simulated twin."""

from __future__ import annotations

import contextlib
import io
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
from cyberwave_geometry import Quaternion, Vector3, quaternion

from cyberwave.test.driver_testing import (
    Feedback,
    SectionRegistry,
    StressHarness,
    Ticket,
    cli,
    command_coverage,
    offset_along_z,
    profile_from,
    resolve_plan,
    section,
)
from cyberwave.test.driver_testing.harness import NO_MEASURED_TCP
from cyberwave.test.driver_testing.registry import REGISTRY, SectionSkipped

ARM6 = [f"joint{i}" for i in range(1, 7)]


def schema(arm, gripper=(), lower=-2.0, upper=2.0, home=None):
    joints = [
        {
            "name": n,
            "type": "revolute",
            "limits": {"lower": lower, "upper": upper},
            **({"home_position": home[n]} if home else {}),
        }
        for n in arm
    ]
    joints += [
        {"name": n, "type": "prismatic", "limits": {"lower": 0.0, "upper": 0.05}}
        for n in gripper
    ]
    joints.append({"name": "fixed_mount", "type": "fixed"})
    return {"joints": joints}


def catalog(commands, *, pose=True, kinematics=False, motion=True, navigate=False):
    out = {
        "joint": {"update": {}},
        "commands": {"supported": [{"name": c, "args": []} for c in commands]},
    }
    if pose:
        out["twin"] = {"pose": {}}
    if kinematics or motion:
        out["state"] = {}
    if kinematics:
        out["state"]["kinematics"] = {}
    if motion:
        out["state"]["motion"] = {}
    if navigate:
        out["navigate"] = {"command": {}, "status": {}}
    return out


#: The Piper's catalog (public/manifest.yaml): jogs, base rotation, planned
#: motion, obstacles, navigate, twin/pose only.
PIPER_COMMANDS = [
    "ee_pose",
    "ee_move",
    "ee_rotate_left",
    "ee_rotate_right",
    "ee_rotate_up",
    "ee_rotate_down",
    "base_rotate_left",
    "base_rotate_right",
    "grip",
    "release",
    "add_obstacles",
    "remove_obstacle",
    "set_pose",
    "grasp",
    "scene_set",
    "attach_payload",
    "detach_payload",
    "home",
]
#: A Kinova-like bridge: seven joints, a Robotiq knuckle, MoveIt only, no jogs or base nudge.
KINOVA_COMMANDS = [
    "set_pose",
    "grasp",
    "scene_set",
    "attach_payload",
    "detach_payload",
    "grip",
    "release",
    "home",
]


def piper_profile(**extra):
    return profile_from(
        "piper",
        catalog=catalog(PIPER_COMMANDS, navigate=True),
        universal_schema=schema(ARM6, ["joint7", "joint8"]),
        motion_status={"kinematics_backend": "moveit", "backend_available": True},
        **extra,
    )


def kinova_profile():
    arm = [f"joint_{i}" for i in range(1, 8)]
    universal = schema(arm)
    universal["joints"].append(
        {
            "name": "robotiq_85_left_knuckle_joint",
            "type": "revolute",
            "limits": {"lower": 0.0, "upper": 0.8},
        }
    )
    return profile_from(
        "kinova",
        catalog=catalog(KINOVA_COMMANDS, pose=False, kinematics=True, navigate=True),
        universal_schema=universal,
        motion_status={"kinematics_backend": "moveit", "backend_available": True},
    )


def plan_of(profile, **kwargs):
    return {e.id: e.skipped for e in resolve_plan(profile, **kwargs)}


# ── discovery ───────────────────────────────────────────────────────────────


def test_piper_profile_splits_arm_and_gripper_and_reads_backends():
    profile = piper_profile(extra_backends=["pinocchio", "moveit"])
    assert [j.name for j in profile.arm_joints] == ARM6
    assert profile.gripper_joint == "joint7"
    assert profile.backends == ("moveit", "pinocchio")
    assert profile.topics == {"joints", "pose", "motion", "navigate"}
    assert profile.joint("joint1").limited


def test_kinova_profile_finds_the_knuckle_and_the_kinematics_pose():
    profile = kinova_profile()
    assert len(profile.arm_joints) == 7
    assert profile.gripper_joint == "robotiq_85_left_knuckle_joint"
    assert "kinematics" in profile.topics and "pose" not in profile.topics


def test_backend_served_catalog_discovers_flat_topics_and_observed_state():
    served = {
        "commands": {"supported": [{"name": c, "args": []} for c in PIPER_COMMANDS]},
        "topics": {
            "cyberwave/joint/{twin_uuid}/update": {},
            "cyberwave/twin/{twin_uuid}/pose": {},
            "cyberwave/twin/{twin_uuid}/navigate/command": {},
        },
    }
    profile = profile_from(
        "piper",
        catalog=served,
        universal_schema=schema(ARM6, ["joint7"]),
        motion_status={"kinematics_backend": "moveit", "backend_available": True},
        observed_topics={"motion", "kinematics"},
    )
    assert profile.topics == {"joints", "pose", "navigate", "motion", "kinematics"}
    plan = plan_of(profile)
    assert plan["joints.single"] is None
    assert plan["planned.ptp@moveit"] is None
    assert plan["navigation.mission"] is None


def test_observed_navigate_topic_enables_waypoint_sections_without_catalog():
    profile = profile_from(
        "t",
        catalog=catalog(PIPER_COMMANDS),
        universal_schema=schema(ARM6, ["joint7"]),
        observed_topics={"navigate"},
    )
    assert "navigate" in profile.topics
    assert plan_of(profile)["navigation.mission"] is None


def test_explicit_joint_names_win_and_unknown_names_are_refused():
    profile = profile_from(
        "t", universal_schema=schema(ARM6, ["joint7"]), arm_joints=ARM6[:3]
    )
    assert [j.name for j in profile.arm_joints] == ARM6[:3]
    with pytest.raises(ValueError, match="not in the twin"):
        profile_from("t", universal_schema=schema(ARM6), arm_joints=["elbow"])


def test_schema_home_needs_every_arm_joint():
    home = dict.fromkeys(ARM6, 0.1)
    assert profile_from("t", universal_schema=schema(ARM6, home=home)).home() == home
    assert profile_from("t", universal_schema=schema(ARM6)).home() is None


# ── the plan ───────────────────────────────────────────────────────────────


def test_piper_plan_runs_its_sections_and_skips_the_rest_with_a_reason():
    plan = plan_of(piper_profile(extra_backends=["pinocchio"]))
    for runs in (
        "joints.single",
        "joints.sine",
        "cartesian.ee_move",
        "cartesian.ee_rotate",
        "planned.ptp@moveit",
        "planned.lin@pinocchio",
        "planned.stop",
        "scene.collision_guard",
        "scene.payload",
        "grasp.candidates",
        "gripper.endpoints",
        "base.rotate",
        "obstacles.roundtrip",
        "navigation.mission",
        "faults.refusals",
    ):
        assert plan[runs] is None, runs
    assert plan["joints.home"] is None
    # twin/pose only: the sections that need an absolute TCP never start a move.
    assert plan["planned.replace"] == NO_MEASURED_TCP
    assert plan["cartesian.ee_pose"] == NO_MEASURED_TCP


def test_kinova_plan_skips_jogs_base_and_obstacles():
    plan = plan_of(kinova_profile())
    assert plan["cartesian.ee_move"] == "catalog lacks ee_move"
    assert plan["base.rotate"] == "catalog lacks base_rotate_left, base_rotate_right"
    assert plan["obstacles.roundtrip"] == "catalog lacks add_obstacles, remove_obstacle"
    # state/kinematics stands in for twin/pose.
    assert plan["planned.ptp@moveit"] is None and plan["joints.home"] is None


def test_every_catalog_command_is_tested_or_says_why_not():
    profile = profile_from(
        "t",
        catalog=catalog(["ee_pose", "ee_rotate_up", "grip", "release", "wave_hello"]),
        universal_schema=schema(ARM6, ["joint7"]),
    )
    rows = {r.command: r for r in command_coverage(profile, resolve_plan(profile))}
    assert rows["grip"].status == "tested" and rows["grip"].sections == (
        "gripper.endpoints",
    )
    assert rows["wave_hello"].reason == "no section exercises it"
    # ee_pose needs the measured TCP; this twin only publishes twin/pose.
    assert rows["ee_pose"].status == "untested"
    assert rows["ee_pose"].reason == (
        "cartesian.ee_pose: no measured TCP on state/kinematics: "
        "compose CartesianStateMixin with use_cartesian_state = True"
    )
    # Left out on purpose is not untested.
    deselected = command_coverage(profile, resolve_plan(profile, skip=["gripper"]))
    assert {r.command: r.status for r in deselected}["grip"] == "not selected"


def test_a_section_that_skips_itself_at_run_time_tested_nothing():
    profile = piper_profile()
    plan = resolve_plan(profile, sections=["grasp"])
    phases = [
        {"section": "grasp.candidates", "outcome": "skipped", "reason": "moveit down"},
        {
            "section": "grasp.execute@moveit",
            "outcome": "skipped",
            "reason": "moveit down",
        },
    ]
    row = next(
        r for r in command_coverage(profile, plan, phases) if r.command == "grasp"
    )
    assert row.status == "untested"
    assert (
        row.reason == "grasp.candidates: moveit down; grasp.execute@moveit: moveit down"
    )


def test_an_untested_command_blocks_the_run_unless_allowed(tmp_path):
    twin = Mock(
        uuid="sim",
        capabilities={},
        _data={"universal_schema": schema(ARM6, ["joint7"])},
    )
    twin.commands.get_schema.return_value = catalog(
        ["grip", "release", "wave_hello"], motion=False
    )
    args = ["--uuid", "sim", "--report", str(tmp_path)]
    with (
        patch("cyberwave.Cyberwave", return_value=fake_client(twin)),
        patch("sys.stdin", io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()) as out,
        contextlib.redirect_stderr(io.StringIO()) as err,
    ):
        assert cli.main([*args, "--yes"]) == 1
        assert "no section can test wave_hello" in err.getvalue()
        assert "untested      wave_hello" in out.getvalue()
        # Allowed: the gate passes and the run reaches the confirmation.
        assert cli.main([*args, "--allow-untested", "wave_hello"]) == 1
        assert "without --yes" in err.getvalue()
    twin.joints.set.assert_not_called()


def test_a_twin_without_planned_motion_plans_no_planned_section():
    profile = profile_from(
        "so101",
        catalog=catalog(["grip", "release"], pose=False, motion=False),
        universal_schema=schema(ARM6[:5], ["gripper_jaw"]),
    )
    plan = plan_of(profile)
    assert plan["planned.ptp"] == "catalog lacks set_pose"
    assert plan["joints.multi"] is None and plan["gripper.endpoints"] is None


def test_sections_and_skip_select_by_area_id_and_backend():
    profile = piper_profile(extra_backends=["pinocchio"])
    plan = plan_of(profile, sections=["planned", "gripper.endpoints"], skip=["@moveit"])
    assert plan["planned.ptp@pinocchio"] is None
    assert plan["planned.ptp@moveit"] == "skipped by --skip"
    assert plan["gripper.endpoints"] is None
    assert plan["joints.single"] == "not selected by --sections"
    # A selection never revives a section the twin cannot run.
    assert (
        plan_of(profile, sections=["cartesian.ee_pose"])["cartesian.ee_pose"]
        == NO_MEASURED_TCP
    )
    with pytest.raises(ValueError, match="Unknown section 'wheels'"):
        resolve_plan(profile, sections=["wheels"])


def test_a_new_capability_is_one_decorated_function():
    registry = SectionRegistry()

    @section(
        "conveyor.belt",
        area="conveyor",
        summary="belt",
        commands={"belt_start"},
        registry=registry,
    )
    def belt(harness):
        return {"ok": True}

    with pytest.raises(ValueError, match="already registered"):
        section("conveyor.belt", area="conveyor", summary="again", registry=registry)(
            belt
        )
    entries = resolve_plan(piper_profile(), registry=registry)
    assert [(e.id, e.skipped) for e in entries] == [
        ("conveyor.belt", "catalog lacks belt_start")
    ]
    assert len(REGISTRY) > 15  # the SDK's own registry is untouched


# ── feedback ───────────────────────────────────────────────────────────────


def parse_joints(payload):
    return SimpleNamespace(positions=payload.get("positions", {}))


def kinematics(xyzw, frame="base_link"):
    return {
        "cartesian_pose": {
            "reference_frame": frame,
            "position": dict(zip("xyz", xyzw[:3])),
            "orientation": dict(zip("xyzw", xyzw[3:])),
        }
    }


def test_kinematics_tcp_replaces_twin_pose_and_frames_never_mix():
    feedback = Feedback(parse_joints)
    feedback.receive(
        "pose",
        {
            "source_type": "edge",
            "frame_id": "base",
            "child_frame_id": "flange",
            "pose_xyzw": [0, 0, 0.5, 0, 0, 0, 1],
        },
    )
    feedback.receive(
        "pose",
        {
            "source_type": "edge",
            "frame_id": "base",
            "child_frame_id": "other",
            "pose_xyzw": [0, 0, 0.9, 0, 0, 0, 1],
        },
    )
    assert feedback.snapshot("pose", 1)[2] == 0.5
    assert not feedback.pose_is_tcp
    feedback.receive("kinematics", kinematics([0, 0, 0.3, 0, 0, 0, 1]))
    feedback.receive(
        "pose",
        {
            "source_type": "edge",
            "frame_id": "base",
            "child_frame_id": "flange",
            "pose_xyzw": [0, 0, 0.5, 0, 0, 0, 1],
        },
    )
    assert feedback.pose_is_tcp and feedback.snapshot("pose", 1)[2] == 0.3
    assert feedback.pose_frame == ("base_link", "tcp")


def test_results_are_matched_by_motion_id_not_arrival():
    feedback = Feedback(parse_joints)
    first = Ticket("set_pose", time.monotonic())
    feedback.receive(
        "motion", {"command": "set_pose", "motion_id": 1, "motion_state": "executing"}
    )
    second = Ticket("set_pose", time.monotonic())
    feedback.receive(
        "motion",
        {
            "command": "set_pose",
            "motion_id": 2,
            "motion_state": "planning",
            "last_result": {"outcome": "cancelled", "motion_id": 1, "timestamp_ms": 1},
        },
    )
    assert feedback.terminal(second) is None
    assert feedback.terminal(first)[1]["motion_id"] == 1


# ── safety ─────────────────────────────────────────────────────────────────


def make_harness(profile=None, **overrides):
    args = cli.arguments(
        [
            "--uuid",
            "twin",
            "--phase-timeout-s",
            "2",
            "--sine-duration-s",
            "4",
            "--no-resume-s",
            "0.05",
            "--manual-lease-s",
            "0.05",
        ]
    )
    args.sine_joints = list(ARM6)
    for key, value in overrides.items():
        setattr(args, key, value)
    profile = profile or profile_from(
        "twin",
        catalog=catalog(PIPER_COMMANDS),
        universal_schema=schema(ARM6, ["joint7"]),
        motion_status={"kinematics_backend": "pinocchio", "backend_available": True},
    )
    client = SimpleNamespace(mqtt=SimpleNamespace(connected=True))
    return StressHarness(client, Mock(), Feedback(parse_joints), args, profile)


def test_gripper_settling_uses_jaw_tolerance_and_latest_measurement():
    harness = make_harness()
    start = time.monotonic()
    positions = [
        (0.1, 0.0),
        (0.2, 0.003),
        (0.3, 0.006),
        (0.4, 0.009),
        (0.5, 0.012),
        (0.6, 0.015),
        (0.7, 0.018),
        (0.8, 0.0184),
        (0.9, 0.0184),
        (1.0, 0.0184),
        (1.1, 0.0184),
        (1.2, 0.0184),
    ]
    settled_at = []

    def deliver(predicate, _timeout, _label):
        for offset, position in positions:
            harness.feedback.latest["joints"] = (
                start + offset,
                {"joint7": position},
            )
            if predicate():
                settled_at.append(offset)
                return
        pytest.fail("jaw never settled")

    harness.feedback.wait = deliver
    result = harness.wait_still(
        start, names=("joint7",), tolerance=harness.args.gripper_tolerance_m
    )
    assert settled_at[0] >= 1.1
    assert result["joint7"] == pytest.approx(0.0184)


def test_targets_are_clamped_inside_the_limits_and_steps_turn_back():
    harness = make_harness()
    margin = harness.args.joint_limit_margin_rad
    assert harness.clamp({"joint1": 5.0})["joint1"] == pytest.approx(2.0 - margin)
    assert harness.step_toward_centre("joint1", 1.9, 0.2) == pytest.approx(1.7)
    assert harness.step_toward_centre("joint1", -1.9, 0.2) == pytest.approx(-1.7)
    harness.profile = profile_from(
        "t", universal_schema=schema(ARM6, lower=-0.1, upper=0.1)
    )
    with pytest.raises(SectionSkipped, match="no 0.200 rad of room"):
        harness.step_toward_centre("joint1", 0.0, 0.2)


def test_targets_outside_the_workspace_box_are_never_sent():
    harness = make_harness()
    with pytest.raises(SectionSkipped, match="outside the workspace box"):
        harness.move([0.0, 0.0, 1.5, 0, 0, 0, 1])
    harness.arm.commands.set_pose.assert_not_called()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--sine-frequency-hz", "0.5", "peak joint speed"),
        ("--cartesian-step-m", "0.2", "exceeds --max-cartesian-step-m"),
        ("--teleop-rate-hz", "20", "at most 10"),
        ("--workspace-min", "1,0,0", "below --workspace-max"),
    ],
)
def test_unsafe_settings_are_refused(flag, value, message, capsys):
    with pytest.raises(SystemExit):
        cli.arguments(["--uuid", "t", flag, value])
    assert message in capsys.readouterr().err


# ── a full run on a simulated twin ───────────────────────────────────────────


class SimulatedArm:
    """A twin whose driver honours the planned-motion contract, measured at 200 Hz.

    It tests the harness's orchestration, not a real driver.
    """

    HOME: ClassVar[dict[str, float]] = dict(zip(ARM6, (0.0, 0.5, -0.5, 0.0, 0.4, 0.0)))

    def __init__(self, feedback, tcp_feed=True):
        self.feedback = feedback
        #: False: only twin/pose, of a flange 0.1 m behind the TCP.
        self.tcp_feed = tcp_feed
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.joints = {**self.HOME, "joint7": 0.04}
        self.pose = [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]
        self.status = {
            "kinematics_backend": "pinocchio",
            "backend_available": True,
            "motion_state": "idle",
            "attached_payload": None,
        }
        self.stamp = self.motion_counter = 0
        self.active = None
        self.mission = None
        self.scene = []
        self.revision = None
        self.sent = []
        self.metadata = {
            "world_obstacles": {
                "geometries": [
                    {
                        "id": "wall",
                        "primitive": "box",
                        "dimensions": [0.1, 0.1, 0.1],
                        "pose": {
                            "position": {"x": -1, "y": 0, "z": 0},
                            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                        },
                    }
                ]
            }
        }
        self.nav_status = {}
        commands = {
            c: (lambda c=c, **data: self.command(c, data)) for c in PIPER_COMMANDS
        }
        self.twin = SimpleNamespace(
            uuid="sim",
            metadata=self.metadata,
            refresh=lambda: None,
            commands=SimpleNamespace(**commands),
            joints=SimpleNamespace(set=self.set_joints),
            publish_command=lambda name, data, **options: self.command(
                name, {**data, **options}
            ),
            navigation=SimpleNamespace(
                follow_path=self.follow_path,
                stop=self.nav_stop,
                wait_for_completion=self.wait_nav,
            ),
        )
        self.thread = threading.Thread(target=self.tick, daemon=True)

    # driver side
    def publish(self, **values):
        self.status.update(values)
        self.feedback.receive("motion", dict(self.status))

    def stamped(self, result):
        self.stamp += 1
        return {**result, "timestamp_ms": self.stamp}

    def refuse(self, command, code, detail="", motion_id=None):
        result = {"command": command, "code": code, "detail": detail}
        if motion_id is not None:
            result["motion_id"] = motion_id
        self.publish(last_refusal=self.stamped(result))

    def accept(self, command, **extra):
        self.publish(
            last_result=self.stamped(
                {"command": command, "outcome": "accepted", **extra}
            )
        )

    def cancel(self):
        if self.active:
            motion_id = self.active["motion_id"]
            command = self.active["command"]
            self.active = None
            self.publish(
                motion_state="cancelled",
                last_refusal=self.stamped(
                    {"command": command, "code": "cancelled", "motion_id": motion_id}
                ),
                last_result=self.stamped(
                    {
                        "command": command,
                        "outcome": "cancelled",
                        "motion_id": motion_id,
                    }
                ),
            )

    def blocked(self, xyz):
        boxes = [*self.scene, *self.metadata["world_obstacles"]["geometries"]]
        for box in boxes:
            centre = box["pose"]["position"]
            half = box["dimensions"][0] / 2
            if all(abs(v - centre[k]) <= half for v, k in zip(xyz, "xyz")):
                return box["id"]
        return None

    def command(self, name, data):
        assert data.pop("source_type") == "tele"
        self.sent.append(name)
        with self.lock:
            handler = getattr(self, "on_" + name, None)
            if handler is not None:
                handler(data)

    def on_stop(self, data):
        self.cancel()

    def on_home(self, data):
        self.cancel()
        self.joints.update(self.HOME)

    def on_set_pose(self, data):
        backend = data.get("backend", "pinocchio")
        if backend != "pinocchio":
            return self.refuse(
                "set_pose", "backend_unavailable", f"{backend}: not running"
            )
        if "scene_revision" in data and data["scene_revision"] != self.revision:
            return self.refuse(
                "set_pose", "scene_revision_unknown", data["scene_revision"]
            )
        expected = data.get("expected_joints")
        if expected and any(abs(expected[n] - self.joints[n]) > 0.02 for n in expected):
            return self.refuse(
                "set_pose", "stale_robot_state", "expected_joints differ"
            )
        pose = data["pose"]
        assert set(pose["orientation"]) == set("xyzw")
        if data.get("mode") == "relative":
            assert [pose["orientation"][k] for k in "xyzw"] == [0.0, 0.0, 0.0, 1.0]
            target = [
                self.pose[i] + pose["position"][key] for i, key in enumerate("xyz")
            ] + self.pose[3:]
        else:
            target = [pose["position"][k] for k in "xyz"] + [
                pose["orientation"][k] for k in "xyzw"
            ]
        self.motion_counter += 1
        motion_id = self.motion_counter
        self.publish(command="set_pose", motion_id=motion_id, motion_state="planning")
        hit = self.blocked(target[:3]) or self.blocked(self.pose[:3])
        if hit:
            return self.publish(
                motion_state="failed",
                last_refusal=self.stamped(
                    {
                        "command": "set_pose",
                        "code": "planning_failed",
                        "motion_id": motion_id,
                        "detail": getattr(
                            self, "collision_detail", f"contact with obstacle {hit!r}"
                        ),
                    }
                ),
            )
        if data.get("plan_only"):
            return self.publish(
                motion_state="idle",
                last_result=self.stamped(
                    {
                        "command": "set_pose",
                        "outcome": "planned",
                        "motion_id": motion_id,
                        "planner": "pinocchio",
                    }
                ),
            )
        self.cancel()
        self.start("set_pose", motion_id, [(0.4, target)])

    def start(self, command, motion_id, waypoints, **result):
        """Execute *waypoints* ``[(after_s, pose)]``, then report *result*."""
        now = time.monotonic()
        self.active = {
            "command": command,
            "waypoints": [(now + after, pose) for after, pose in waypoints],
            "motion_id": motion_id,
            "result": result,
        }
        self.publish(command=command, motion_id=motion_id, motion_state="executing")

    def on_grasp(self, data):
        backend = data.get("backend", "pinocchio")
        if backend != "pinocchio":
            return self.refuse(
                "grasp", "backend_unavailable", f"{backend}: not running"
            )
        assert data["reference_frame"] == "tool"
        self.motion_counter += 1
        reachable = [
            i
            for i, c in enumerate(data["candidates"])
            if abs(c["pose"]["position"]["z"]) < 1.0
        ]
        if not data.get("plan_only"):
            # Tool-frame candidate: PTP to the pregrasp, back along the tool's
            # +Z, then a straight approach onto the grasp pose. No lift, no grip.
            grasp = list(self.pose)  # the candidate is the tool's own pose
            pregrasp = offset_along_z(grasp, -data["approach_distance_m"])
            self.cancel()
            return self.start(
                "grasp",
                self.motion_counter,
                [(0.2, pregrasp), (0.4, grasp)],
                selected_candidate=reachable[0],
            )
        self.publish(
            command="grasp",
            motion_id=self.motion_counter,
            motion_state="idle",
            last_result=self.stamped(
                {
                    "command": "grasp",
                    "outcome": "planned",
                    "motion_id": self.motion_counter,
                    "selected_candidate": reachable[0],
                }
            ),
        )

    def jog(self, name, change):
        change()
        # Plain SDK jogs publish measured pose but no planned-motion status.
        if name == "ee_pose":
            self.publish(command=name, motion_state="idle")

    def on_ee_move(self, data):
        self.jog("ee_move", lambda: self.pose.__setitem__(2, self.pose[2] + data["up"]))

    def turn(self, axis, angle):
        """Turn the tool about its own *axis* (about the TCP: it stays put)."""
        step = quaternion.from_axis_angle(Vector3(*axis), angle)
        turned = quaternion.multiply(Quaternion.from_xyzw(self.pose[3:]), step)
        self.pose[3:] = [turned.x, turned.y, turned.z, turned.w]

    def on_ee_rotate_left(self, data):
        self.jog("ee_rotate_left", lambda: self.turn((0, 0, 1), data["angle"]))

    def on_ee_rotate_right(self, data):
        self.jog("ee_rotate_right", lambda: self.turn((0, 0, 1), -data["angle"]))

    def on_ee_rotate_up(self, data):
        self.jog("ee_rotate_up", lambda: self.turn((0, 1, 0), data["angle"]))

    def on_ee_rotate_down(self, data):
        self.jog("ee_rotate_down", lambda: self.turn((0, 1, 0), -data["angle"]))

    def on_ee_pose(self, data):
        x, y, z, roll, pitch, yaw = data["pose"]
        turned = quaternion.from_rpy(roll, pitch, yaw)
        self.jog(
            "ee_pose",
            lambda: self.pose.__init__(
                [x, y, z, turned.x, turned.y, turned.z, turned.w]
            ),
        )

    def on_base_rotate_left(self, data):
        self.joints["joint1"] += data["angle"]

    def on_base_rotate_right(self, data):
        self.joints["joint1"] -= data["angle"]

    def on_grip(self, data):
        self.joints["joint7"] = 0.0

    def on_release(self, data):
        self.joints["joint7"] = 0.04

    def on_scene_set(self, data):
        self.scene, self.revision = data["objects"], data["scene_revision"]
        self.accept("scene_set")
        self.publish(active_scene_revision=self.revision)

    def on_attach_payload(self, data):
        self.accept("attach_payload")
        self.publish(attached_payload=data["object_id"])

    def on_detach_payload(self, data):
        self.accept("detach_payload")
        self.publish(attached_payload=None)

    def on_add_obstacles(self, data):
        self.metadata["world_obstacles"] = {"geometries": data["geometries"]}

    def on_remove_obstacle(self, data):
        geometries = self.metadata["world_obstacles"]["geometries"]
        self.metadata["world_obstacles"] = {
            "geometries": [g for g in geometries if g["id"] != data["id"]]
        }

    def set_joints(self, values, **options):
        assert options == {"degrees": False, "source_type": "tele"}
        with self.lock:
            if set(values) & set(ARM6):
                self.cancel()
            self.joints.update(values)

    def follow_path(self, waypoints, **options):
        assert options["skip_nav_anchor_transform"]
        with self.lock:
            action = f"nav-{len(self.nav_status)}"
            self.nav_status[action] = None
            self.mission = {
                "id": action,
                "points": [[w["position"][k] for k in "xyz"] for w in waypoints],
            }
        return {"action_id": action}

    def nav_stop(self, **options):
        with self.lock:
            if self.mission:
                self.nav_status[self.mission["id"]] = {"status": "cancelled"}
                self.mission = None

    def wait_nav(self, action_id, timeout=10, raise_on_failure=True):
        deadline = time.monotonic() + timeout
        while self.nav_status.get(action_id) is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return self.nav_status[action_id]

    def tick(self):
        while not self.done.wait(0.005):
            with self.lock:
                waypoints = self.active["waypoints"] if self.active else []
                if waypoints and time.monotonic() >= waypoints[0][0]:
                    self.pose = list(waypoints.pop(0)[1])
                if self.active and not waypoints:
                    active, self.active = self.active, None
                    self.publish(
                        motion_state="idle",
                        last_result=self.stamped(
                            {
                                "command": active["command"],
                                "outcome": "succeeded",
                                "motion_id": active["motion_id"],
                                **active["result"],
                            }
                        ),
                    )
                if self.mission:
                    goal = self.mission["points"][0]
                    for axis in range(3):  # 0.2 m/s along each axis
                        self.pose[axis] += max(
                            -0.001, min(0.001, goal[axis] - self.pose[axis])
                        )
                    if math.dist(goal, self.pose[:3]) < 1e-6:
                        self.mission["points"].pop(0)
                        if not self.mission["points"]:
                            self.nav_status[self.mission["id"]] = {
                                "status": "completed"
                            }
                            self.mission = None
                self.feedback.receive(
                    "joints", {"source_type": "edge", "positions": dict(self.joints)}
                )
                if self.tcp_feed:
                    self.feedback.receive("kinematics", kinematics(self.pose))
                else:
                    flange = [*self.pose[:2], self.pose[2] - 0.1, *self.pose[3:]]
                    self.feedback.receive(
                        "pose",
                        {
                            "source_type": "edge",
                            "frame_id": "base_link",
                            "child_frame_id": "flange",
                            "pose_xyzw": flange,
                        },
                    )
                self.feedback.receive("motion", dict(self.status))


@pytest.fixture
def simulated():
    profile = profile_from(
        "sim",
        catalog=catalog(PIPER_COMMANDS, pose=False, kinematics=True, navigate=True),
        universal_schema=schema(ARM6, ["joint7"], home=SimulatedArm.HOME),
        motion_status={"kinematics_backend": "pinocchio", "backend_available": True},
        extra_backends=["moveit"],
    )
    harness = make_harness(profile, navigate_link="tcp", gripper_tolerance_m=0.002)
    arm = SimulatedArm(harness.feedback)
    harness.arm = arm.twin
    arm.thread.start()
    yield harness, arm, profile
    arm.done.set()
    arm.thread.join()


def test_every_section_passes_on_a_simulated_twin(simulated):
    harness, arm, profile = simulated
    plan = resolve_plan(profile)
    with contextlib.redirect_stdout(io.StringIO()):
        cli.run_plan(harness, plan)
    outcomes = {
        p["section"]: (p["outcome"], p.get("error") or p.get("reason"))
        for p in harness.phases
    }
    failed = {k: v for k, v in outcomes.items() if v[0] == "failed"}
    assert not failed
    # moveit is reported, but the driver refuses it: skipped at run time with the refusal.
    assert outcomes["planned.ptp@moveit"] == (
        "skipped",
        "backend moveit unavailable: moveit: not running",
    )
    assert outcomes["planned.ptp@pinocchio"][0] == "passed"
    assert (
        outcomes["base.rotate"][0] == "passed"
        and outcomes["navigation.cancel"][0] == "passed"
    )
    assert outcomes["return"][0] == "passed"
    # Every command is exercised: ee_pose, all four rotations and an executed grasp too.
    assert outcomes["cartesian.ee_pose"][0] == "passed"
    assert outcomes["cartesian.ee_rotate"][0] == "passed"
    assert outcomes["grasp.execute@pinocchio"][0] == "passed"
    assert {"ee_rotate_up", "ee_rotate_down"} <= set(arm.sent)
    coverage = command_coverage(profile, plan, harness.phases)
    assert [r.command for r in coverage if r.status != "tested"] == []
    grasp = next(r for r in coverage if r.command == "grasp")
    assert grasp.sections == ("grasp.candidates", "grasp.execute@pinocchio")
    # The world obstacle list ends as it started.
    assert [g["id"] for g in arm.metadata["world_obstacles"]["geometries"]] == ["wall"]


def test_with_only_a_flange_pose_moves_are_relative_and_absolute_sections_skip():
    profile = profile_from(
        "sim",
        catalog=catalog(PIPER_COMMANDS),
        universal_schema=schema(ARM6, ["joint7"]),
        motion_status={"kinematics_backend": "pinocchio", "backend_available": True},
    )
    # The fixture must reach the TCP from the flange: an edge over twice the offset.
    harness = make_harness(profile, probe_box_m=0.3)
    arm = SimulatedArm(harness.feedback, tcp_feed=False)
    harness.arm = arm.twin
    arm.thread.start()
    try:
        plan = resolve_plan(
            profile, sections=["planned", "scene", "cartesian", "faults"]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            cli.run_plan(harness, plan)
    finally:
        arm.done.set()
        arm.thread.join()
    outcomes = {
        p["section"]: (p["outcome"], p.get("error") or p.get("reason"))
        for p in harness.phases
    }
    assert not {k: v for k, v in outcomes.items() if v[0] == "failed"}
    assert outcomes["planned.ptp@pinocchio"][0] == "passed"
    assert outcomes["planned.replace"] == ("skipped", NO_MEASURED_TCP)
    assert outcomes["scene.collision_guard"][0] == "passed"


@pytest.fixture
def flange_only():
    """A simulated twin that reports only a flange pose on twin/pose."""
    profile = profile_from(
        "sim",
        catalog=catalog(PIPER_COMMANDS),
        universal_schema=schema(ARM6, ["joint7"]),
        motion_status={"kinematics_backend": "pinocchio", "backend_available": True},
    )
    harness = make_harness(profile)
    arm = SimulatedArm(harness.feedback, tcp_feed=False)
    harness.arm = arm.twin
    arm.thread.start()
    harness.feedback.wait(
        lambda: {"joints", "pose", "motion"} <= set(harness.feedback.latest), 2, "feed"
    )
    yield harness, arm
    arm.done.set()
    arm.thread.join()


def test_a_section_that_cannot_finish_skips_before_it_moves(flange_only):
    harness, arm = flange_only
    start = harness.pose()
    with pytest.raises(SectionSkipped, match="no measured TCP"):
        harness.command_replacement()
    assert arm.sent == [] and arm.active is None
    # The first joint has no room for the manual step: skipped before the set_pose.
    harness.profile = profile_from(
        "t", universal_schema=schema(ARM6, lower=-0.1, upper=0.1)
    )
    with pytest.raises(SectionSkipped, match="no 0.200 rad of room"):
        harness.teleop_takeover()
    assert arm.sent == []
    assert harness.pose()[2] == pytest.approx(start[2])


def test_a_skip_after_motion_still_ends_in_stop_and_names_what_was_sent(flange_only):
    harness, arm = flange_only

    def careless():
        harness.step(0.02)
        raise SectionSkipped("changed my mind")

    with contextlib.redirect_stdout(io.StringIO()):
        harness.phase("careless", careless, section="careless")
    entry = harness.phases[-1]
    assert entry["outcome"] == "skipped"
    assert entry["sent_before_skip"] == ["set_pose"]
    assert arm.sent == ["set_pose", "stop"] and arm.active is None


def test_an_operator_scene_or_payload_is_never_replaced(flange_only):
    harness, arm = flange_only
    arm.publish(active_scene_revision="operator-table", attached_payload="bin")
    with pytest.raises(
        SectionSkipped, match="task scene \\(operator-table\\) is active"
    ):
        harness.scene_collision_guard()
    with pytest.raises(SectionSkipped, match="payload 'bin' is attached"):
        harness.payload()
    assert arm.sent == []
    # A scene a previous stress run left behind is ours to replace.
    arm.publish(active_scene_revision="stress-0123")
    harness.args.probe_box_m = 0.3  # the box must reach the TCP from the flange
    assert harness.scene_collision_guard()["refusal"]["code"] == "planning_failed"
    arm.publish(attached_payload=None)
    assert harness.payload() == {"object_id": "stress-payload"}


def test_home_without_declared_endpoint_first_moves_away(simulated):
    harness, arm, profile = simulated
    profile.arm_joints = tuple(
        type(j)(j.name, j.type, j.lower, j.upper, None) for j in profile.arm_joints
    )
    harness.feedback.wait(lambda: "joints" in harness.feedback.latest, 2.0, "joints")
    arm.joints["joint1"] = 0.1
    harness.feedback.wait(
        lambda: abs(harness.joints()["joint1"] - 0.1) < 0.01,
        2.0,
        "offset joint feedback",
    )
    result = harness.home()
    assert result["expected_home"] is None
    assert "home" in arm.sent
    assert any(e["kind"] == "target" for e in harness.feedback.events)
    assert result["measured_home"] == pytest.approx(arm.HOME, abs=0.035)


def test_scene_guard_accepts_generic_planner_refusal_after_clear_plan(simulated):
    harness, arm, _profile = simulated
    arm.collision_detail = "INVALID_MOTION_PLAN (pilz_ptp)"
    assert harness.scene_collision_guard()["refusal"]["detail"] == arm.collision_detail


def test_waypoint_mission_cancels_on_interrupt(simulated):
    harness, arm, _profile = simulated
    harness.feedback.wait(lambda: harness.feedback.pose_frame is not None, 2.0, "pose")
    arm.wait_nav = lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    harness.arm.navigation.wait_for_completion = arm.wait_nav
    with pytest.raises(KeyboardInterrupt):
        harness.waypoint_mission()
    assert arm.mission is None
    assert harness.active_mission_id is None


def test_a_failed_navigation_section_cancels_its_mission(simulated):
    """The return mission times out: the section fails and the mission is cancelled."""
    harness, arm, _profile = simulated
    harness.args.continue_on_failure = True
    harness.feedback.wait(
        lambda: {"joints", "pose", "motion"} <= set(harness.feedback.latest), 2, "feed"
    )
    calls = []
    wait_nav = arm.twin.navigation.wait_for_completion

    def wait_for_completion(action_id, **options):
        calls.append(action_id)
        if len(calls) == 2:  # the mission back to the start
            raise TimeoutError("return mission timed out")
        return wait_nav(action_id, **options)

    arm.twin.navigation.wait_for_completion = wait_for_completion
    with contextlib.redirect_stdout(io.StringIO()):
        harness.phase("cancel", harness.navigation_cancel, section="navigation.cancel")
    assert harness.phases[-1]["outcome"] == "failed"
    assert calls == ["nav-0", "nav-1"]
    assert arm.nav_status["nav-1"] == {"status": "cancelled"}
    assert arm.mission is None and harness.active_mission_id is None
    assert arm.sent[-1] == "stop"


def test_a_failed_section_ends_in_stop_and_the_next_one_runs(simulated):
    harness, arm, profile = simulated
    harness.args.continue_on_failure = True
    plan = [
        e for e in resolve_plan(profile, sections=["gripper.endpoints", "joints.multi"])
    ]
    arm.on_grip = lambda data: None  # a driver that drops grip
    with contextlib.redirect_stdout(io.StringIO()):
        cli.run_plan(harness, plan)
    outcomes = {p["section"]: p["outcome"] for p in harness.phases}
    assert (
        outcomes["gripper.endpoints"] == "failed"
        and outcomes["joints.multi"] == "passed"
    )
    assert "stop" in arm.sent[arm.sent.index("grip") :]


# ── the command line ───────────────────────────────────────────────────────


def fake_client(twin):
    mqtt = SimpleNamespace(
        connected=True, connect=Mock(), subscribe=Mock(), unsubscribe=Mock()
    )
    return SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(topic_prefix=""),
        twin=Mock(return_value=twin),
        disconnect=Mock(),
    )


def test_list_prints_the_plan_and_moves_nothing():
    twin = Mock(
        uuid="sim",
        capabilities={},
        _data={"universal_schema": schema(ARM6, ["joint7"])},
    )
    twin.commands.get_schema.return_value = catalog(PIPER_COMMANDS, motion=False)
    client = fake_client(twin)
    with (
        patch("cyberwave.Cyberwave", return_value=client),
        contextlib.redirect_stdout(io.StringIO()) as out,
    ):
        assert cli.main(["--uuid", "sim", "--list"]) == 0
    text = out.getvalue()
    assert "run   joints.single" in text and "skip  planned.ptp" in text
    twin.commands.set_pose.assert_not_called()
    twin.joints.set.assert_not_called()
    twin.publish_command.assert_not_called()
    client.disconnect.assert_called_once()


def test_list_uses_backend_served_catalog_and_observed_motion():
    twin = Mock(
        uuid="sim",
        capabilities={},
        _data={"universal_schema": schema(ARM6, ["joint7"])},
    )
    twin.commands.get_schema.return_value = {
        "commands": {"supported": [{"name": c, "args": []} for c in PIPER_COMMANDS]},
        "topics": {
            "cyberwave/joint/{twin_uuid}/update": {},
            "cyberwave/twin/{twin_uuid}/pose": {},
        },
    }
    client = fake_client(twin)

    def subscribe(topic, callback, **_kwargs):
        if topic.endswith("/state/motion"):
            callback({"kinematics_backend": "moveit", "backend_available": True})

    client.mqtt.subscribe.side_effect = subscribe
    with (
        patch("cyberwave.Cyberwave", return_value=client),
        contextlib.redirect_stdout(io.StringIO()) as out,
    ):
        assert cli.main(["--uuid", "sim", "--list"]) == 0
    assert "run   planned.ptp@moveit" in out.getvalue()
    assert "\n0 of" not in out.getvalue()
    twin.commands.set_pose.assert_not_called()


def test_a_live_run_needs_yes(tmp_path):
    twin = Mock(
        uuid="sim",
        capabilities={},
        _data={"universal_schema": schema(ARM6, ["joint7"])},
    )
    twin.commands.get_schema.return_value = catalog(["grip", "release"], motion=False)
    with (
        patch("cyberwave.Cyberwave", return_value=fake_client(twin)),
        patch("sys.stdin", io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()) as err,
    ):
        assert cli.main(["--uuid", "sim", "--report", str(tmp_path)]) == 1
    assert "without --yes" in err.getvalue()
    twin.joints.set.assert_not_called()


def test_report_has_phases_trace_and_config(simulated, tmp_path):
    harness, _arm, profile = simulated
    plan = resolve_plan(profile, sections=["joints.multi"])
    with contextlib.redirect_stdout(io.StringIO()):
        cli.run_plan(harness, plan)
    from cyberwave.test.driver_testing import write_report

    write_report(
        tmp_path / "out.json",
        outcome="passed",
        error=None,
        config={"uuid": "sim"},
        phases=harness.phases,
        events=harness.feedback.events,
        joint_names=ARM6,
        plan=[e.to_json() for e in plan],
    )
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["config"] == {"uuid": "sim"}
    assert {p["outcome"] for p in report["phases"]} == {"passed", "skipped"}
    assert report["plan"][0]["outcome"] in ("planned", "skipped")
    assert "joints" in (tmp_path / "events.jsonl").read_text()
    assert (
        Path(tmp_path / "joints.csv").read_text().startswith("monotonic_s,kind,joint1")
    )
