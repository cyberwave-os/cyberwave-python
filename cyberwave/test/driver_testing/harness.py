"""The stress harness: drives a twin through the public SDK and judges it by measurement.

:class:`StressHarness` holds the primitives every section uses -- sending a
command and matching its result by ``motion_id``, streaming joint targets,
waiting for measured joints or poses to arrive and hold -- and one method per
generic section. A robot-specific harness subclasses it and
overrides what is specific to that robot.

The harness is generic over two pose sources:

* ``state/kinematics`` -- the driver's measured TCP in its planning frame. The
  pose *is* the ``set_pose`` target frame, so sections send absolute targets.
* ``twin/pose`` alone -- some link of the arm in some frame. Sections then move
  by relative displacements (``mode: relative`` in the planning frame) and check
  the measured displacement, which is the same for every point of the tool
  while its orientation holds. Sections that need an absolute TCP target are
  skipped with that reason.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .feedback import (
    Feedback,
    Teleop,
    Ticket,
    euler_pose,
    rotation_angle,
    target_pose,
    tracking_metrics,
)
from .registry import SectionSkipped

logger = logging.getLogger(__name__)

#: Text in a refusal's detail that proves a collision check refused it; a
#: generic IK miss proves nothing about collision checking.
_COLLISION_WORDS = ("collis", "contact with obstacle")


#: Why a section that needs an absolute TCP target cannot run: the driver does
#: not publish its measured TCP, which the SDK's ``CartesianStateMixin``
#: provides (``use_cartesian_state = True``) on ``state/kinematics``.
NO_MEASURED_TCP = (
    "no measured TCP on state/kinematics: compose CartesianStateMixin "
    "with use_cartesian_state = True"
)

#: ``step()`` without a backend: the harness's :attr:`StressHarness.step_backend`.
_STEP_BACKEND = object()


def is_collision_refusal(result: Mapping[str, Any]) -> bool:
    detail = str(result.get("detail", "")).lower()
    return bool(result.get("code")) and any(word in detail for word in _COLLISION_WORDS)


class StressHarness:
    """Primitives and generic sections for one twin; see the module docstring.

    *args* is the parsed command line (see :func:`cyberwave.test.driver_testing.cli.arguments`);
    *profile* the :class:`~cyberwave.test.driver_testing.discovery.TwinProfile` it was
    planned on (``None`` for a robot-specific harness that hard-codes both).
    """

    #: Arm joints, when a subclass knows them; otherwise from the profile.
    arm_joints: tuple[str, ...] = ()
    #: Backend of the arm moves sections use as load (``None``: the driver's default).
    step_backend: str | None = None

    def __init__(
        self, client: Any, arm: Any, feedback: Feedback, args: Any, profile: Any = None
    ) -> None:
        self.client, self.arm, self.feedback, self.args = client, arm, feedback, args
        self.profile = profile
        if profile is not None and not self.arm_joints:
            self.arm_joints = tuple(j.name for j in profile.arm_joints)
        self.phases: list[dict[str, Any]] = []
        self.stream: Teleop | None = None
        self.scene_revision = "stress-" + uuid.uuid4().hex
        #: Moves name the revision only once a scene_set made it active.
        self.scene_set = False
        self.last_teleop_s = 0.0
        self.last_teleop_targets: dict[str, float] = {}
        #: Measured joints and pose at preflight: where the run returns to.
        self.start_joints: dict[str, float] | None = None
        self.start_pose: list[float] | None = None
        self.active_mission_id: str | None = None

    # ── configuration helpers ───────────────────────────────────────────────

    def option(self, name: str, default: Any = None) -> Any:
        """An argument a robot-specific command line may not define."""
        return getattr(self.args, name, default)

    @property
    def pose_is_tcp(self) -> bool:
        """Whether the measured pose is the TCP in the ``set_pose`` planning frame."""
        return self.feedback.pose_is_tcp

    @property
    def default_backend(self) -> str | None:
        return getattr(self.profile, "default_backend", None)

    def joint_limits(self, name: str) -> tuple[float, float] | None:
        """Schema limits shrunk by ``--joint-limit-margin-rad``, if the joint has any."""
        joint = self.profile.joint(name) if self.profile is not None else None
        if joint is None or not joint.limited:
            return None
        margin = self.option("joint_limit_margin_rad", 0.0)
        lower, upper = joint.lower + margin, joint.upper - margin
        if lower >= upper:
            return None
        return lower, upper

    def clamp(self, values: Mapping[str, float]) -> dict[str, float]:
        """Every target inside its joint's limits (minus the margin)."""
        clamped = {}
        for name, value in values.items():
            limits = self.joint_limits(name)
            clamped[name] = (
                value if limits is None else min(max(value, limits[0]), limits[1])
            )
        return clamped

    def step_toward_centre(self, name: str, value: float, step: float) -> float:
        """*value* moved by *step* in whichever direction stays inside the limits."""
        limits = self.joint_limits(name)
        if limits is None:
            return value + step
        lower, upper = limits
        if value + step <= upper and (
            value - step < lower or value <= (lower + upper) / 2
        ):
            return value + step
        if value - step >= lower:
            return value - step
        raise SectionSkipped(
            f"{name} has no {step:.3f} rad of room inside its limits at {value:.3f}"
        )

    def step_away_from_centre(self, name: str, value: float, step: float) -> float:
        """Pick a safe probe away from the usual centre-valued home posture."""
        limits = self.joint_limits(name)
        if limits is None:
            return value + step
        lower, upper = limits
        direction = 1 if value >= (lower + upper) / 2 else -1
        for candidate in (value + direction * step, value - direction * step):
            if lower <= candidate <= upper:
                return candidate
        raise SectionSkipped(
            f"{name} has no {step:.3f} rad of room inside its limits at {value:.3f}"
        )

    def check_workspace(self, *poses: Sequence[float]) -> None:
        """Refuse a target outside the configured Cartesian box, before sending it."""
        low, high = self.option("workspace_min"), self.option("workspace_max")
        if low is None or high is None:
            return
        for pose in poses:
            if not all(lo <= v <= hi for v, lo, hi in zip(pose[:3], low, high)):
                raise SectionSkipped(
                    f"target {[round(v, 3) for v in pose[:3]]} is outside the workspace box "
                    f"{list(low)}..{list(high)}"
                )

    # ── sending ──────────────────────────────────────────────────────────────

    def command(self, name: str, **data: Any) -> float:
        if not self.client.mqtt.connected:
            raise RuntimeError("MQTT disconnected during test")
        sent = self.feedback.record("command", {"command": name, "data": data})
        if name == "stop":
            # Management verbs are not in the catalog: publish directly.
            self.arm.publish_command("stop", data, source_type="tele")
        else:
            getattr(self.arm.commands, name)(source_type="tele", **data)
        return sent

    def set_joints(self, values: Mapping[str, float], *, force: bool = False) -> None:
        """Rate-limited joint teleop; arm deltas under the minimum are dropped."""
        if not self.client.mqtt.connected:
            raise RuntimeError("MQTT disconnected during teleop")
        measured = self.feedback.snapshot("joints", self.args.feedback_timeout_s)
        values = self.clamp(values)
        values = {
            name: value
            for name, value in values.items()
            if force
            or name not in self.arm_joints
            or max(
                abs(value - self.last_teleop_targets.get(name, measured[name])),
                abs(value - measured[name]),
            )
            >= self.args.min_teleop_delta_rad - 1e-9
        }
        if not values:
            return
        time.sleep(
            max(
                0.0,
                self.last_teleop_s + 1 / self.args.teleop_rate_hz - time.monotonic(),
            )
        )
        self.feedback.record("target", values.copy())
        self.arm.joints.set(values, degrees=False, source_type="tele")
        self.last_teleop_s = time.monotonic()
        self.last_teleop_targets.update(values)

    def move(
        self,
        target: Sequence[float],
        backend: str | None = None,
        motion: str = "ptp",
        plan_only: bool = False,
        *,
        relative: bool = False,
        expected_joints: bool | None = None,
    ) -> Ticket:
        """One ``set_pose``, planned by *backend* (``None``: the driver's default).

        *target* is an absolute ``pose_xyzw`` or, with *relative*, an ``[x, y, z]``
        displacement in the planning frame. In-process (pinocchio) moves carry
        ``expected_joints`` from fresh measured feedback; other backends take
        their start state at the driver.
        """
        if relative:
            data: dict[str, Any] = {
                "pose": target_pose([*target, 0.0, 0.0, 0.0, 1.0]),
                "mode": "relative",
                "reference_frame": "base",
            }
        else:
            self.check_workspace(target)
            data = {
                "pose": target_pose(target),
                "reference_frame": self.scene_frame(),
            }
        data.update(motion=motion, plan_only=plan_only)
        if backend is not None:
            data["backend"] = backend
        if expected_joints is None:
            expected_joints = (backend or self.default_backend) == "pinocchio"
        if expected_joints:
            data["expected_joints"] = self.joints()
        if self.scene_set:
            data["scene_revision"] = self.scene_revision
        return Ticket("set_pose", self.command("set_pose", **data))

    def step(
        self,
        dz: float,
        backend: Any = _STEP_BACKEND,
        motion: str = "ptp",
        plan_only: bool = False,
    ) -> tuple[Ticket, list[float]]:
        """Move the tool *dz* up in the planning frame; returns the ticket and the expected measured pose."""
        if backend is _STEP_BACKEND:
            backend = self.step_backend
        start = self.pose()
        expected = start.copy()
        expected[2] += dz
        self.check_workspace(expected)
        if self.pose_is_tcp:
            return self.move(expected, backend, motion, plan_only), expected
        return self.move(
            [0.0, 0.0, dz], backend, motion, plan_only, relative=True
        ), expected

    # ── measured state ─────────────────────────────────────────────────────

    def joints(self) -> dict[str, float]:
        values = self.feedback.snapshot("joints", self.args.feedback_timeout_s)
        missing = [name for name in self.arm_joints if name not in values]
        if missing:
            raise RuntimeError(f"Measured feedback lacks arm joints {missing}")
        return {name: values[name] for name in self.arm_joints}

    def pose(self) -> list[float]:
        return self.feedback.snapshot("pose", self.args.feedback_timeout_s)

    def motion_status(self) -> dict[str, Any]:
        return self.feedback.latest.get("motion", (0.0, {}))[1]

    def wait_joints(
        self,
        target: Mapping[str, float],
        after_s: float,
        stable_s: float = 0.4,
        tolerance: float | None = None,
    ) -> None:
        tolerance = self.args.joint_tolerance_rad if tolerance is None else tolerance
        since = None

        def reached() -> bool:
            nonlocal since
            received, values = self.feedback.latest.get("joints", (0, {}))
            now = time.monotonic()
            fresh = received > after_s and now - received < self.args.feedback_timeout_s
            close = all(
                name in values and abs(values[name] - value) <= tolerance
                for name, value in target.items()
            )
            since = (since or received) if fresh and close else None
            return since is not None and received - since >= stable_s

        self.feedback.wait(
            reached, self.args.phase_timeout_s, "Joint target not reached and held"
        )

    def wait_motion_idle(self, command: str, after_s: float) -> None:
        status_commands = (
            ("set_pose", "ee_pose", command)
            if command.startswith("ee_")
            else (command,)
        )

        def finished() -> bool:
            received, motion = self.feedback.latest.get("motion", (0, {}))
            if received <= after_s or motion.get("command") not in status_commands:
                return False
            state = motion.get("motion_state")
            if state == "failed":
                raise RuntimeError(f"{command} failed: {motion.get('last_refusal')}")
            return state == "idle"

        self.feedback.wait(
            finished, self.args.phase_timeout_s, f"{command} did not finish"
        )

    def wait_still(
        self,
        after_s: float,
        stable_s: float = 0.4,
        names: Sequence[str] | None = None,
        tolerance: float | None = None,
    ) -> dict[str, float]:
        """Wait until measured *names* (default: the arm) stop moving; returns where."""
        names = tuple(names or self.arm_joints)
        tolerance = self.args.joint_tolerance_rad if tolerance is None else tolerance
        held: dict[str, float] | None = None
        since = None
        last_received = 0.0

        def stopped() -> bool:
            nonlocal held, since, last_received
            received, values = self.feedback.latest.get("joints", (0, {}))
            if (
                received <= after_s
                or time.monotonic() - received >= self.args.feedback_timeout_s
            ):
                return False
            if not all(name in values for name in names):
                return False
            if received != last_received:
                if held is None or any(
                    abs(values[name] - held[name]) > tolerance / 2 for name in names
                ):
                    held = {name: values[name] for name in names}
                    since = received
                last_received = received
            return since is not None and received - since >= stable_s

        self.feedback.wait(stopped, self.args.phase_timeout_s, "Arm did not settle")
        return {name: self.feedback.latest["joints"][1][name] for name in names}

    def wait_pose(
        self, target: Sequence[float], after_s: float, *, orientation: bool = True
    ) -> None:
        def reached() -> bool:
            received, values = self.feedback.latest.get("pose", (0, []))
            if (
                received <= after_s
                or time.monotonic() - received >= self.args.feedback_timeout_s
            ):
                return False
            if math.dist(values[:3], target[:3]) > self.args.position_tolerance_m:
                return False
            return (
                not orientation
                or rotation_angle(values, target) <= self.args.orientation_tolerance_rad
            )

        self.feedback.wait(
            reached, self.args.phase_timeout_s, "Measured TCP target not reached"
        )

    # ── results, matched by motion_id ─────────────────────────────────────

    def outcome(self, ticket: Ticket) -> tuple[float, dict[str, Any]]:
        """The terminal result for *ticket*, waited for and claimed."""
        found = None

        def arrived() -> bool:
            nonlocal found
            found = self.feedback.terminal(ticket)
            return found is not None

        self.feedback.wait(
            arrived,
            self.args.phase_timeout_s,
            f"No terminal result for {ticket.command} sent at {ticket.sent:.3f}",
        )
        assert found is not None
        return found

    def result(self, ticket: Ticket, outcome: str = "succeeded") -> dict[str, Any]:
        _received, result = self.outcome(ticket)
        # Local cancellation uses last_refusal.code; supersede uses last_result.outcome.
        if result.get("outcome", result.get("code")) != outcome:
            raise RuntimeError(
                f"{ticket.command}: expected {outcome}, received {result}"
            )
        return result

    def takeover(self, ticket: Ticket, sent_s: float) -> dict[str, Any]:
        with self.feedback.condition:
            found = self.feedback.terminal(ticket)
        if found is None:
            return {"old_command_result": "not_reported"}
        received, result = found
        outcome = result.get("outcome", result.get("code"))
        if received < sent_s:
            raise RuntimeError(
                f"{ticket.command} ended before the replacement was sent: {result}"
            )
        if outcome != "cancelled":
            raise RuntimeError(
                f"{ticket.command} ended after replacement with {result}"
            )
        return {
            "old_command_result": "cancelled",
            "cancellation_confirmation_s": received - sent_s,
        }

    def executing(self, ticket: Ticket) -> None:
        def started() -> bool:
            ended = self.feedback.terminal(ticket, claim=False)
            if ended is not None:
                raise RuntimeError(f"Motion ended before takeover: {ended[1]}")
            motion_id = self.feedback.motion_id(ticket)
            return motion_id is not None and any(
                running == motion_id for _received, running in self.feedback.executing
            )

        self.feedback.wait(
            started, self.args.phase_timeout_s, "Motion never entered execution"
        )

    def planned(self, ticket: Ticket, backend: str | None) -> dict[str, Any]:
        """A plan-only result; an unavailable backend skips the section."""
        _received, result = self.outcome(ticket)
        if result.get("code") == "backend_unavailable":
            raise SectionSkipped(
                f"backend {backend or self.default_backend} unavailable: {result.get('detail')}"
            )
        if result.get("outcome") != "planned":
            raise RuntimeError(f"{ticket.command}: expected planned, received {result}")
        return result

    # ── phases ───────────────────────────────────────────────────────────────

    def phase(
        self,
        name: str,
        function: Callable[[], Any],
        *,
        required: bool = False,
        section: str | None = None,
    ) -> None:
        print(f"\n{len(self.phases) + 1:02d} {name}", flush=True)
        start_s = self.feedback.record("phase_start", {"name": name})
        entry: dict[str, Any] = {"name": name, "start_monotonic_s": start_s}
        if section is not None:
            entry["section"] = section
        self.phases.append(entry)
        try:
            entry["metrics"] = function() or {}
            entry["outcome"] = "passed"
        except SectionSkipped as exc:
            entry.update(outcome="skipped", reason=str(exc))
            moved = self.motion_sent_since(start_s)
            if moved:
                # "Skipped" promises nothing happened: a section that skips
                # after sending motion still ends in stop and a measured hold,
                # and the report names what it sent.
                entry["sent_before_skip"] = moved
                try:
                    self.stop_and_hold()
                except Exception as hold_error:
                    entry.update(
                        outcome="failed",
                        error=f"Skip cleanup failed: {hold_error}; skip reason: {exc}",
                    )
                    raise
        except Exception as exc:
            entry.update(outcome="failed", error=str(exc))
            try:
                self.stop_and_hold()
            except Exception as hold_error:
                entry["error"] += f"; stop/hold failed: {hold_error}"
                raise
            if required or not self.args.continue_on_failure:
                raise
        finally:
            entry["end_monotonic_s"] = self.feedback.record("phase_end", entry.copy())
        print(
            json.dumps(
                entry.get(
                    "metrics",
                    {"error": entry.get("error"), "reason": entry.get("reason")},
                ),
                sort_keys=True,
            ),
            flush=True,
        )

    #: Commands that change state but never move the arm.
    NON_MOTION_COMMANDS = frozenset(
        {
            "stop",
            "scene_set",
            "attach_payload",
            "detach_payload",
            "add_obstacles",
            "remove_obstacle",
        }
    )

    def motion_sent_since(self, start_s: float) -> list[str]:
        """Joint targets, waypoint missions and non-plan-only commands sent since *start_s*."""
        with self.feedback.condition:
            events = [e for e in self.feedback.events if e["monotonic_s"] >= start_s]
        sent = []
        for event in events:
            if event["kind"] in ("target", "waypoint_command"):
                sent.append(event["kind"])
            elif event["kind"] == "command":
                name, data = event["payload"]["command"], event["payload"]["data"]
                if name not in self.NON_MOTION_COMMANDS and not data.get("plan_only"):
                    sent.append(name)
        return sent

    def stop_and_hold(self) -> None:
        """``stop``, then a measured hold: what follows every failed section.

        A waypoint mission is cancelled first: whether a driver's arm ``stop``
        also ends a mission is up to the driver.
        """
        self.stop_navigation()
        sent = self.command("stop")
        held = self.wait_still(sent)
        self.wait_joints(held, time.monotonic(), stable_s=self.args.no_resume_s)

    # ── preflight ────────────────────────────────────────────────────────────

    def preflight(self) -> dict[str, Any]:
        """Measured start state, inside the limits and the box, held still."""
        needed = ["joints"]
        topics: frozenset[str] = getattr(self.profile, "topics", frozenset())
        if {"pose", "kinematics"} & topics:
            needed.append("pose")
        if "motion" in topics:
            needed.append("motion")
        self.feedback.wait(
            lambda: all(k in self.feedback.latest for k in needed),
            self.args.phase_timeout_s,
            f"No complete driver feedback (waiting for {needed})",
        )
        joints = self.joints()
        outside = {
            name: value
            for name, value in joints.items()
            if (limits := self.joint_limits(name)) is not None
            and not (
                limits[0] - self.args.joint_tolerance_rad
                <= value
                <= limits[1] + self.args.joint_tolerance_rad
            )
        }
        if outside:
            raise RuntimeError(
                f"Measured joints outside their limits (minus margin): {outside}"
            )
        self.command("stop")
        self.start_joints = self.wait_still(time.monotonic())
        metrics: dict[str, Any] = {"measured_start_joints": self.start_joints}
        if "pose" in needed:
            self.start_pose = self.pose()
            self.check_workspace(self.start_pose)
            metrics.update(
                measured_start_pose=self.start_pose,
                pose_frame=list(self.feedback.pose_frame or ()),
                pose_is_tcp=self.pose_is_tcp,
            )
        if "motion" in needed:
            status = self.motion_status()
            metrics["motion_status"] = {
                key: status.get(key)
                for key in ("kinematics_backend", "backend_available", "motion_state")
            }
        attached = getattr(getattr(self.arm, "policy", None), "attached", None)
        if attached is not None:
            metrics["controller_type"] = str(getattr(attached, "controller_type", None))
        return metrics

    def return_to_start(self) -> dict[str, Any]:
        """Measured return to the joints recorded at preflight."""
        if self.start_joints is None:
            raise SectionSkipped("no measured start posture")
        sent = time.monotonic()
        self.set_joints(self.start_joints, force=True)
        self.wait_joints(self.start_joints, sent)
        return {"measured_joints": self.joints()}

    # ── joints ─────────────────────────────────────────────────────────────

    def single_joints(self) -> dict[str, Any]:
        """Each arm joint alone to a target inside its limits, then back."""
        baseline = self.joints()
        reached = {}
        for name in self.arm_joints:
            target = {
                name: self.step_toward_centre(
                    name, baseline[name], self.args.amplitude_rad
                )
            }
            sent = time.monotonic()
            self.set_joints(target)
            self.wait_joints(target, sent)
            # The other joints must not move while one is commanded.
            self.wait_joints(baseline | target, time.monotonic(), stable_s=0.2)
            sent = time.monotonic()
            self.set_joints({name: baseline[name]})
            self.wait_joints(baseline, sent)
            reached[name] = target[name]
        return {"targets": reached}

    def multi_joint(self) -> dict[str, Any]:
        """Every arm joint at once to a target inside its limits, then back."""
        baseline = self.joints()
        target = {
            name: self.step_toward_centre(name, value, self.args.amplitude_rad)
            for name, value in baseline.items()
        }
        sent = time.monotonic()
        self.set_joints(target)
        self.wait_joints(target, sent)
        sent = time.monotonic()
        self.set_joints(baseline, force=True)
        self.wait_joints(baseline, sent)
        return {"target": target}

    def sine(self, duration_s: float | None = None) -> dict[str, Any]:
        duration_s = duration_s or self.args.sine_duration_s
        baseline = self.joints()
        if self.profile is not None:
            # The sine reaches ±amplitude around the baseline: keep that inside the limits.
            for name in self.args.sine_joints:
                limits = self.joint_limits(name)
                if limits and not (
                    limits[0] <= baseline[name] - self.args.amplitude_rad
                    and baseline[name] + self.args.amplitude_rad <= limits[1]
                ):
                    raise SectionSkipped(
                        f"{name} at {baseline[name]:.3f} has no ±{self.args.amplitude_rad} rad of room"
                    )
        self.stream = Teleop(
            self, baseline, self.args.amplitude_rad, duration_s
        ).start()
        try:
            self.stream.thread.join()
            self.stream.close()
            metrics = tracking_metrics(
                self.feedback.events,
                self.stream.start_s,
                self.stream.end_s,
                self.args.sine_joints,
            )
        finally:
            self.stream.close()
            self.stream = None
        # MQTT delivery can lag or coalesce measured positions. The stream is
        # load, while explicit endpoint/takeover phases prove control behavior.
        sent = time.monotonic()
        self.set_joints(baseline, force=True)
        self.wait_joints(baseline, sent, stable_s=self.args.no_resume_s)
        return metrics

    def partial_targets(self) -> dict[str, Any]:
        """A newer single-joint target replaces an older one at once: latest target wins."""
        baseline = self.joints()
        first = self.arm_joints[0]
        step = self.args.amplitude_rad
        up = self.step_toward_centre(first, baseline[first], step)
        sign = 1 if up > baseline[first] else -1
        down = baseline[first] - sign * step
        limits = self.joint_limits(first)
        if limits is not None and not limits[0] <= down <= limits[1]:
            down = baseline[first]
        target = baseline.copy()
        target[first] = up
        sent = time.monotonic()
        self.set_joints(target)
        self.feedback.wait(
            lambda: (
                self.feedback.latest.get("joints", (0, {}))[0] > sent
                and sign
                * (
                    self.feedback.latest["joints"][1].get(first, baseline[first])
                    - baseline[first]
                )
                >= self.args.min_teleop_delta_rad / 2
            ),
            self.args.phase_timeout_s,
            "First joint target made no measured progress",
        )
        # New intent must replace this axis immediately, with no FIFO backlog.
        sent = time.monotonic()
        self.set_joints({first: down}, force=True)
        target[first] = down
        self.wait_joints(target, sent)
        self.wait_joints(target, time.monotonic(), stable_s=self.args.no_resume_s)
        sent = time.monotonic()
        self.set_joints(baseline, force=True)
        self.wait_joints(baseline, sent)
        return {}

    def home(self) -> dict[str, Any]:
        """``home``: the arm settles, at the schema's home when it declares one."""
        expected = self.option("home_joints") or (
            self.profile.home() if self.profile is not None else None
        )
        if not expected:
            # A robot already home must still prove that home can move it.
            before = self.joints()
            first = self.arm_joints[0]
            nudge = self.step_away_from_centre(
                first, before[first], self.args.min_teleop_delta_rad
            )
            sent = time.monotonic()
            self.set_joints({first: nudge}, force=True)
            self.wait_joints({first: nudge}, sent)
            time.sleep(self.args.manual_lease_s)
        before = self.joints()
        sent = self.command("home")
        if expected:
            self.wait_joints(expected, sent)
        else:
            self.feedback.wait(
                lambda: (
                    self.feedback.latest.get("joints", (0, {}))[0] > sent
                    and any(
                        abs(self.feedback.latest["joints"][1].get(n, v) - v)
                        > self.args.joint_tolerance_rad
                        for n, v in before.items()
                    )
                ),
                self.args.phase_timeout_s,
                "home produced no measured arm movement",
            )
        held = self.wait_still(sent)
        time.sleep(self.args.manual_lease_s)
        return {"measured_home": held, "expected_home": expected}

    # ── Cartesian jogs ─────────────────────────────────────────────────────

    def jog(self) -> dict[str, Any]:
        """``ee_move`` up and its inverse; measured displacement and measured restore."""
        baseline = self.pose()
        target = baseline.copy()
        target[2] += self.args.cartesian_step_m
        self.check_workspace(target)
        sent = self.command("ee_move", up=self.args.cartesian_step_m, frame="base")
        self.wait_pose(target, sent)
        self.wait_still(sent)
        sent = self.command("ee_move", up=-self.args.cartesian_step_m, frame="base")
        self.wait_pose(baseline, sent)
        self.wait_still(sent)
        return {"measured_end_pose": self.pose()}

    #: Each tool rotation command with its inverse.
    ROTATION_PAIRS = (
        ("ee_rotate_left", "ee_rotate_right"),
        ("ee_rotate_up", "ee_rotate_down"),
    )

    def rotation_pairs(self) -> list[tuple[str, str]]:
        commands: frozenset[str] = getattr(self.profile, "commands", frozenset())
        pairs = [pair for pair in self.ROTATION_PAIRS if set(pair) <= commands]
        unpaired = sorted(
            c
            for pair in self.ROTATION_PAIRS
            for c in pair
            if c in commands and not set(pair) <= commands
        )
        if unpaired:
            # A rotation without its inverse cannot be undone by measurement here.
            raise RuntimeError(f"{', '.join(unpaired)} has no inverse in the catalog")
        if not pairs:
            raise SectionSkipped("catalog has no ee_rotate_* command with its inverse")
        return pairs

    def tool_rotation(self) -> dict[str, Any]:
        """Every ``ee_rotate_*`` pair: measured orientation change, then restore by the inverse."""
        measured = {}
        for there, back in self.rotation_pairs():
            measured[there] = self.rotate_and_back(there, back)
            measured[back] = self.rotate_and_back(back, there)
        return {"measured_rotation_rad": measured}

    def rotate_and_back(self, there: str, back: str) -> float:
        """*there* by the step angle (measured), then *back* to the measured baseline."""
        angle = self.args.rotation_step_rad
        baseline = self.pose()
        sent = self.command(there, angle=angle)

        def rotated() -> bool:
            received, pose = self.feedback.latest.get("pose", (0, []))
            if (
                received <= sent
                or time.monotonic() - received > self.args.feedback_timeout_s
            ):
                return False
            return rotation_angle(baseline, pose) >= angle / 2

        self.feedback.wait(
            rotated, self.args.phase_timeout_s, f"{there} produced no measured rotation"
        )
        self.wait_still(sent)
        turned = self.pose()
        measured = rotation_angle(baseline, turned)
        if abs(measured - angle) > self.args.orientation_tolerance_rad / 2:
            raise RuntimeError(
                f"{there}: measured {measured:.3f} rad, commanded {angle:.3f}"
            )
        # Only a TCP pose stays put: a flange pose orbits the TCP as the tool turns.
        if (
            self.pose_is_tcp
            and math.dist(baseline[:3], turned[:3]) > self.args.position_tolerance_m
        ):
            raise RuntimeError(f"{there} moved the measured TCP")
        sent = self.command(back, angle=angle)
        self.wait_pose(baseline, sent)
        self.wait_still(sent)
        return measured

    def absolute_pose(self) -> dict[str, Any]:
        """``ee_pose`` to the TCP raised by a step, measured, then ``ee_pose`` back to the start."""
        if not self.pose_is_tcp:
            raise SectionSkipped(NO_MEASURED_TCP)
        baseline = self.pose()
        target = baseline.copy()
        target[2] += self.args.cartesian_step_m
        self.check_workspace(target)
        for goal in (target, baseline):
            sent = self.command(
                "ee_pose", pose=euler_pose(goal), reference_frame=self.scene_frame()
            )
            self.wait_pose(goal, sent)
            self.wait_motion_idle("ee_pose", time.monotonic())
        return {"measured_end_pose": self.pose()}

    # ── planned motion ─────────────────────────────────────────────────────

    def cartesian(
        self, backend: str | None, motion_type: str = "ptp"
    ) -> dict[str, Any]:
        """``set_pose`` *motion_type* on *backend*: plan only, execute, measured arrival, return."""
        baseline = self.pose()
        ticket, target = self.step(
            self.args.cartesian_step_m, backend, motion_type, plan_only=True
        )
        plan = self.planned(ticket, backend)
        sent = time.monotonic()
        ticket, target = self.step(self.args.cartesian_step_m, backend, motion_type)
        done = self.result(ticket)
        self.wait_pose(target, sent)
        sent = time.monotonic()
        if self.pose_is_tcp:
            ticket = self.move(baseline, backend, motion_type)
        else:
            ticket = self.move(
                [0.0, 0.0, -self.args.cartesian_step_m],
                backend,
                motion_type,
                relative=True,
            )
        self.result(ticket)
        self.wait_pose(baseline, sent)
        return {
            "planner": done.get("planner") or plan.get("planner"),
            "motion_id": done.get("motion_id"),
        }

    def command_replacement(self, stop: bool = False) -> dict[str, Any]:
        """A newer ``set_pose`` replaces an executing one (or ``stop`` ends it, measured hold)."""
        if not stop and not self.pose_is_tcp:
            # Before the move: once it is in flight, skipping would leave it running.
            raise SectionSkipped(NO_MEASURED_TCP)
        baseline = self.pose()
        active, _ = self.step(self.args.takeover_step_m)
        self.executing(active)
        if stop:
            sent = self.command("stop")
            held = self.wait_still(sent)
            self.wait_joints(held, time.monotonic(), stable_s=self.args.no_resume_s)
            metrics = self.takeover(active, sent)
            sent = time.monotonic()
            ticket, fresh = self.step(self.args.cartesian_step_m)
            self.result(ticket)  # Stop must not leave the controller unusable.
            self.wait_pose(fresh, sent)
            return metrics
        replacement = self.replace_with(baseline)
        self.wait_pose(baseline, replacement.sent)
        self.wait_motion_idle(replacement.command, time.monotonic())
        held = self.joints()
        self.wait_joints(held, time.monotonic(), stable_s=self.args.no_resume_s)
        return self.takeover(active, replacement.sent)

    def replace_with(self, baseline: list[float]) -> Ticket:
        """The command that replaces an executing move, sending the tool back to *baseline*."""
        if not self.pose_is_tcp:
            raise SectionSkipped(NO_MEASURED_TCP)
        # The replacement takes its start state at the driver: an in-flight
        # arm moves faster than an expected_joints freshness gate allows.
        return self.move(baseline, None, expected_joints=False)

    def teleop_takeover(self, motion_type: str = "ptp") -> dict[str, Any]:
        """A joint target displaces an executing ``set_pose`` and holds its measured endpoint."""
        baseline = self.joints()
        first = self.arm_joints[0]
        # Every check that can skip runs before the move starts.
        manual = {
            first: self.step_toward_centre(
                first, baseline[first], self.args.amplitude_rad
            )
        }
        active, _ = self.step(
            self.option("preemption_step_m", self.args.takeover_step_m),
            motion=motion_type,
        )
        self.executing(active)
        sent = time.monotonic()
        self.set_joints(manual)
        self.wait_joints(manual, sent, stable_s=self.args.no_resume_s)
        metrics = self.takeover(active, sent)
        sent = time.monotonic()
        self.set_joints({first: baseline[first]})
        self.wait_joints({first: baseline[first]}, sent)
        return metrics

    def refusals(self) -> dict[str, Any]:
        """Guards answer with a refusal code, never a timeout."""
        reported = {}
        probes: list[tuple[str, dict[str, Any]]] = [
            (
                "scene_revision_unknown",
                {"scene_revision": "stress-unknown-" + uuid.uuid4().hex},
            )
        ]
        if self.default_backend == "pinocchio" or "pinocchio" in getattr(
            self.profile, "backends", ()
        ):
            # Half a radian from where the arm measurably is, inside the limits.
            stale = {
                name: self.step_toward_centre(name, value, 0.5)
                for name, value in self.joints().items()
            }
            probes.append(
                (
                    "stale_robot_state",
                    {"expected_joints": stale, "backend": "pinocchio"},
                )
            )
        for code, extra in probes:
            data = {
                "pose": target_pose(
                    [0.0, 0.0, self.args.cartesian_step_m, 0.0, 0.0, 0.0, 1.0]
                ),
                "mode": "relative",
                "reference_frame": "base",
                "plan_only": True,
                **extra,
            }
            ticket = Ticket("set_pose", self.command("set_pose", **data))
            _received, result = self.outcome(ticket)
            if result.get("code") != code:
                raise RuntimeError(f"expected refusal {code}, received {result}")
            reported[code] = result.get("detail")
        return {"refusals": reported}

    # ── scene, payload and grasp ───────────────────────────────────────────

    def probe_box(self, pose: Sequence[float], identifier: str) -> dict[str, Any]:
        """A box around the tool and its next step up: any plan into it must collide."""
        centre = list(pose)
        centre[2] += self.args.cartesian_step_m / 2
        return {
            "id": identifier,
            "primitive": "box",
            "dimensions": [self.option("probe_box_m", 0.1)] * 3,
            "pose": target_pose(centre[:3] + [0.0, 0.0, 0.0, 1.0]),
        }

    def scene_frame(self) -> str:
        frame = (self.feedback.pose_frame or ("",))[0]
        if not frame:
            raise SectionSkipped(
                "the measured pose names no frame to place a fixture in"
            )
        return frame

    def send_scene(self, objects: list[dict[str, Any]]) -> dict[str, Any]:
        self.scene_revision = "stress-" + uuid.uuid4().hex
        sent = self.command(
            "scene_set",
            scene_revision=self.scene_revision,
            objects=objects,
            allow_touch=[],
            frame=self.scene_frame(),
        )
        result = self.result(Ticket("scene_set", sent), "accepted")
        self.scene_set = True
        return result

    def scene_collision_guard(self) -> dict[str, Any]:
        """``scene_set`` with a fixture at the tool; a plan-only move into it is refused for collision.

        Skipped when the driver holds a task scene this tool did not install:
        ``scene_set`` replaces it, and the driver cannot report its objects
        back for a restore, so running would strip the collision objects
        whoever loaded them relies on.
        """
        revision = self.motion_status().get("active_scene_revision")
        if revision and not str(revision).startswith("stress-"):
            raise SectionSkipped(
                f"a task scene ({revision}) is active and cannot be restored after the test; "
                "clear it with scene_set first to test scene_set"
            )
        # Establish that this exact target plans in the clear. MoveIt/Pilz can
        # report INVALID_MOTION_PLAN for a collision without saying "collision".
        clear_ticket, _ = self.step(self.args.cartesian_step_m, plan_only=True)
        self.planned(clear_ticket, None)
        fixture = self.probe_box(self.pose(), "stress-scene-probe")
        try:
            self.send_scene([fixture])
            ticket, _ = self.step(self.args.cartesian_step_m, plan_only=True)
            _received, result = self.outcome(ticket)
        finally:
            self.send_scene([])
        if not is_collision_refusal(result):
            if result.get("code") != "planning_failed":
                raise RuntimeError(f"Fixture did not refuse the clear plan: {result}")
            restored_ticket, _ = self.step(self.args.cartesian_step_m, plan_only=True)
            self.planned(restored_ticket, None)
        return {"refusal": result, "fixture": fixture}

    def payload(self) -> dict[str, Any]:
        """``attach_payload`` / ``detach_payload``: ``state/motion`` reports the payload, then none."""
        attached = self.motion_status().get("attached_payload")
        if attached:
            raise SectionSkipped(
                f"payload {attached!r} is attached and cannot be re-attached after the test; "
                "detach it first to test attach_payload"
            )
        object_id = "stress-payload"
        sent = self.command(
            "attach_payload", dimensions=[0.02, 0.02, 0.02], object_id=object_id
        )
        try:
            self.result(Ticket("attach_payload", sent), "accepted")
            self.feedback.wait(
                lambda: self.motion_status().get("attached_payload") == object_id,
                self.args.phase_timeout_s,
                "state/motion never reported the attached payload",
            )
        finally:
            sent = self.command("detach_payload", object_id=object_id)
        self.result(Ticket("detach_payload", sent), "accepted")
        self.feedback.wait(
            lambda: self.motion_status().get("attached_payload") is None,
            self.args.phase_timeout_s,
            "state/motion still reports a payload after detach",
        )
        return {"object_id": object_id}

    def grasp_search(self) -> dict[str, Any]:
        """A plan-only ``grasp``: an unreachable first candidate falls through to a reachable one."""
        candidates = [
            {
                "pose": target_pose([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]),
                "approach": [0.0, 0.0, 1.0],
            },
            {
                "pose": target_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "approach": [0.0, 0.0, 1.0],
            },
        ]
        sent = self.command(
            "grasp", candidates=candidates, reference_frame="tool", plan_only=True
        )
        result = self.planned(Ticket("grasp", sent), None)
        if result.get("selected_candidate") != 1:
            raise RuntimeError(
                f"grasp selected candidate {result.get('selected_candidate')}, expected 1: {result}"
            )
        return {"selected_candidate": 1, "planner": result.get("planner")}

    def grasp_execute(self, backend: str | None) -> dict[str, Any]:
        """An executed ``grasp`` on the tool's own pose, in free space.

        The driver moves to the pregrasp pose ``approach_distance_m`` back along
        the approach axis, then approaches onto the grasp pose; the lift is only
        checked, and the gripper is left alone. Passes on a ``succeeded`` result
        for that candidate, a measured back-off of about the approach distance,
        a measured return to the start pose, and a gripper that did not move.
        """
        approach = self.option("grasp_approach_m", self.args.cartesian_step_m)
        start = self.pose()
        low, high = self.option("workspace_min"), self.option("workspace_max")
        if low is not None and high is not None:
            # The approach axis is the tool's: keep a sphere of that radius inside the box.
            self.check_workspace(
                *(
                    [
                        *(
                            v + sign * approach * (i == axis)
                            for i, v in enumerate(start[:3])
                        ),
                        *start[3:],
                    ]
                    for axis in range(3)
                    for sign in (-1, 1)
                )
            )
        gripper = getattr(self.profile, "gripper_joint", None)
        jaw = (
            self.feedback.latest.get("joints", (0, {}))[1].get(gripper)
            if gripper
            else None
        )
        data: dict[str, Any] = {
            "candidates": [
                {
                    "pose": target_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    "approach": [0.0, 0.0, 1.0],
                }
            ],
            "reference_frame": "tool",
            "approach_distance_m": approach,
        }
        if backend is not None:
            data["backend"] = backend
        if self.scene_set:
            data["scene_revision"] = self.scene_revision
        plan = self.command("grasp", **{**data, "plan_only": True})
        self.planned(Ticket("grasp", plan), backend)
        sent = self.command("grasp", **data)
        ticket = Ticket("grasp", sent)
        _received, result = self.outcome(ticket)
        if result.get("code") == "backend_unavailable":
            raise SectionSkipped(
                f"backend {backend or self.default_backend} unavailable: {result.get('detail')}"
            )
        if result.get("outcome") != "succeeded" or result.get(
            "selected_candidate"
        ) not in (0, None):
            raise RuntimeError(
                f"grasp: expected succeeded on candidate 0, received {result}"
            )
        self.wait_pose(start, sent)
        with self.feedback.condition:
            poses = [
                self.feedback_pose(e)
                for e in self.feedback.events
                if e["kind"] == "pose" and e["monotonic_s"] >= sent
            ]
        backoff = max((math.dist(p[:3], start[:3]) for p in poses if p), default=0.0)
        if backoff < approach * 0.6:
            raise RuntimeError(
                f"grasp reported success without a measured approach: {backoff:.4f} m"
            )
        if gripper and jaw is not None:
            now = self.joints_with(gripper)[gripper]
            if abs(now - jaw) > self.args.gripper_tolerance_m:
                raise RuntimeError(f"grasp moved the gripper: {jaw:.4f} -> {now:.4f}")
        return {
            "motion_id": result.get("motion_id"),
            "measured_backoff_m": backoff,
            "planner": result.get("planner"),
        }

    def _jaw_moved(self, name: str, before: float, sent: float) -> Callable[[], bool]:
        """Measured *name* has moved from *before* since *sent*."""

        def moved() -> bool:
            received, values = self.feedback.latest.get("joints", (0, {}))
            return received > sent and abs(values.get(name, before) - before) >= (
                self.option("gripper_min_travel", 0.005) / 2
            )

        return moved

    def joints_with(self, *names: str | None) -> dict[str, float]:
        """Fresh measured joints, including *names* outside the arm."""
        values = self.feedback.snapshot("joints", self.args.feedback_timeout_s)
        return {n: values[n] for n in names if n is not None and n in values}

    # ── gripper ────────────────────────────────────────────────────────────

    def gripper(self, overlap: bool = False) -> dict[str, Any]:
        """``release`` / ``grip`` / ``release``: measured endpoints, and the arm unaffected."""
        name = self.profile.gripper_joint
        active = None
        if overlap:
            active, _ = self.step(self.args.takeover_step_m)
            self.executing(active)
        endpoints: list[float] = []
        for command in ("release", "grip", "release"):
            before = self.joints_with(name)[name]
            sent = self.command(command)
            if endpoints:
                self.feedback.wait(
                    self._jaw_moved(name, before, sent),
                    self.args.phase_timeout_s,
                    f"{command} produced no measured gripper movement",
                )
            endpoints.append(
                self.wait_still(
                    sent, names=(name,), tolerance=self.args.gripper_tolerance_m
                )[name]
            )
        opened, closed, reopened = endpoints
        if abs(opened - closed) < self.option("gripper_min_travel", 0.005):
            raise RuntimeError(
                f"grip did not move {name}: open {opened:.4f}, closed {closed:.4f}"
            )
        if abs(reopened - opened) > self.args.gripper_tolerance_m:
            raise RuntimeError(
                f"release did not reopen {name}: {opened:.4f} then {reopened:.4f}"
            )
        if active is not None:
            self.result(active)  # The gripper must not cancel the arm.
        return {"open": opened, "closed": closed}

    # ── base rotation ──────────────────────────────────────────────────────

    def base_rotation(self) -> dict[str, Any]:
        """``base_rotate_left`` then ``_right``: one joint turns by the angle and back."""
        angle = self.args.rotation_step_rad
        baseline = self.joints()
        first = self.arm_joints[0]
        order = ("base_rotate_left", "base_rotate_right")
        if self.step_toward_centre(first, baseline[first], angle) < baseline[first]:
            order = order[::-1]
        sent = self.command(order[0], angle=angle)
        self.feedback.wait(
            lambda: any(
                abs(self.feedback.latest["joints"][1].get(n, baseline[n]) - baseline[n])
                >= angle / 2
                for n in self.arm_joints
            ),
            self.args.phase_timeout_s,
            f"{order[0]} produced no measured joint motion",
        )
        turned = self.wait_still(sent)
        deltas = {n: turned[n] - baseline[n] for n in self.arm_joints}
        base = max(deltas, key=lambda n: abs(deltas[n]))
        if abs(abs(deltas[base]) - angle) > self.args.joint_tolerance_rad:
            raise RuntimeError(
                f"{order[0]}: {base} turned {deltas[base]:.3f} rad, commanded {angle:.3f}"
            )
        moved = {
            n: d
            for n, d in deltas.items()
            if n != base and abs(d) > self.args.joint_tolerance_rad
        }
        if moved:
            raise RuntimeError(f"{order[0]} also moved {moved}")
        sent = self.command(order[1], angle=angle)
        self.wait_joints(baseline, sent)
        return {"base_joint": base, "measured_delta_rad": deltas[base]}

    # ── world obstacles ────────────────────────────────────────────────────

    def world_obstacles(self) -> list[dict[str, Any]]:
        self.arm.refresh()
        metadata = self.arm.metadata or {}
        saved = (
            metadata.get("world_obstacles") or metadata.get("pinocchio_obstacles") or {}
        )
        return list(saved.get("geometries", []))

    def wait_world_obstacles(self, expected: list[dict[str, Any]]) -> None:
        deadline = time.monotonic() + self.args.phase_timeout_s
        while self.world_obstacles() != expected:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Driver did not persist the requested world obstacle scene"
                )
            time.sleep(0.5)

    def obstacles(self) -> dict[str, Any]:
        """Install a fixture with ``add_obstacles``, be refused inside it, remove it, restore the list."""
        original = self.world_obstacles()
        fixture = self.probe_box(self.pose(), "stress-obstacle-probe")
        installed = [g for g in original if g.get("id") != fixture["id"]] + [fixture]
        metrics: dict[str, Any] = {"original_ids": [g.get("id") for g in original]}
        try:
            self.command(
                "add_obstacles", frame_id=self.scene_frame(), geometries=installed
            )
            self.wait_world_obstacles(installed)
            if "set_pose" in getattr(
                self.profile, "commands", ()
            ) and "pinocchio" in getattr(self.profile, "backends", ()):
                ticket, _ = self.step(
                    self.args.cartesian_step_m, "pinocchio", plan_only=True
                )
                _received, result = self.outcome(ticket)
                if not is_collision_refusal(result):
                    raise RuntimeError(
                        f"Plan into the obstacle was not refused for collision: {result}"
                    )
                metrics["refusal"] = result
            else:
                metrics["refusal"] = "not checked: no in-process (pinocchio) set_pose"
            self.command("remove_obstacle", id=fixture["id"])
            self.wait_world_obstacles(
                [g for g in installed if g["id"] != fixture["id"]]
            )
        finally:
            if self.world_obstacles() != original:
                self.command(
                    "add_obstacles", frame_id=self.scene_frame(), geometries=original
                )
                self.wait_world_obstacles(original)
        return metrics

    # ── navigation ─────────────────────────────────────────────────────────

    def navigation_start(self) -> list[float]:
        """The measured pose of the link the ``navigate`` service plans for."""
        link = self.option("navigate_link")
        child = (self.feedback.pose_frame or ("", ""))[1]
        if not link:
            raise SectionSkipped(
                "pass --navigate-link <the link the navigate service moves>"
            )
        if link != child:
            raise SectionSkipped(
                f"navigate moves {link} but the measured pose is {child}"
            )
        return self.pose()

    def waypoint_mission(self, planner: str | None = None) -> dict[str, Any]:
        """Out along +Z and back: completed, with measured outward motion and return."""
        start = self.navigation_start()
        outbound = start.copy()
        outbound[2] += self.option("waypoint_step_m", self.args.cartesian_step_m)
        self.check_workspace(outbound)
        frame = self.scene_frame()
        waypoints = [_waypoint(outbound), _waypoint(start)]
        started = self.feedback.record(
            "waypoint_command", {"planner": planner, "waypoints": waypoints}
        )
        try:
            response = self.arm.navigation.follow_path(
                waypoints,
                planner=planner,
                reference_frame=frame,
                skip_nav_anchor_transform=True,
                source_type="tele",
            )
            self.active_mission_id = response["action_id"]
            status = self.arm.navigation.wait_for_completion(
                response["action_id"], timeout=self.args.phase_timeout_s
            )
            self.active_mission_id = None
            self.feedback.record("navigation_status", status)
            if status.get("status") != "completed":
                raise RuntimeError(f"Waypoint mission ended: {status}")
            self.wait_pose(start, started)
        except BaseException:
            try:
                self.stop_navigation()
            except Exception:  # the original failure is the one to report
                logger.exception("navigation stop after a failed mission")
            raise
        with self.feedback.condition:
            poses = [
                e
                for e in self.feedback.events
                if e["kind"] == "pose" and e["monotonic_s"] >= started
            ]
        peak = max((self.feedback_pose(e)[2] - start[2] for e in poses), default=0.0)
        if peak < (outbound[2] - start[2]) * 0.6:
            raise RuntimeError(
                f"Completed waypoint mission without measured outward motion: up={peak:.4f} m"
            )
        return {
            "planner": planner,
            "action_id": response["action_id"],
            "measured_peak_up_m": peak,
        }

    def feedback_pose(self, event: dict[str, Any]) -> list[float]:
        if "pose" in event:
            return event["pose"]
        payload = event["payload"]
        if "pose_xyzw" in payload:
            return payload["pose_xyzw"]
        pose = payload["cartesian_pose"]
        return [pose["position"][k] for k in "xyz"]

    def stop_navigation(self) -> None:
        """Cancel a mission independently of the arm's stop command."""
        if self.active_mission_id is not None:
            self.arm.navigation.stop(source_type="tele")
            self.active_mission_id = None

    def navigation_cancel(self) -> dict[str, Any]:
        """A mission cancelled while it moves: ``cancelled`` status and a measured hold."""
        start = self.navigation_start()
        outbound = start.copy()
        outbound[2] += self.option("waypoint_step_m", self.args.cartesian_step_m) * 2
        self.check_workspace(outbound)
        response = self.arm.navigation.follow_path(
            [_waypoint(outbound)],
            reference_frame=self.scene_frame(),
            skip_nav_anchor_transform=True,
            source_type="tele",
        )
        self.active_mission_id = response["action_id"]
        sent = time.monotonic()
        self.feedback.wait(
            lambda: (
                self.feedback.latest.get("pose", (0, start))[0] > sent
                and self.feedback.latest["pose"][1][2] - start[2]
                >= self.args.position_tolerance_m
            ),
            self.args.phase_timeout_s,
            "Mission produced no measured motion to cancel",
        )
        cancelled = time.monotonic()
        self.stop_navigation()
        status = self.arm.navigation.wait_for_completion(
            response["action_id"],
            timeout=self.args.phase_timeout_s,
            raise_on_failure=False,
        )
        if status.get("status") != "cancelled":
            raise RuntimeError(f"Stopped mission reported {status}")
        held = self.wait_still(cancelled)
        self.wait_joints(held, time.monotonic(), stable_s=self.args.no_resume_s)
        # Back to the start for the next section. Tracked like any mission,
        # so a failure here still cancels it.
        self.active_mission_id = self.arm.navigation.follow_path(
            [_waypoint(start)],
            reference_frame=self.scene_frame(),
            skip_nav_anchor_transform=True,
            source_type="tele",
        )["action_id"]
        self.arm.navigation.wait_for_completion(
            self.active_mission_id, timeout=self.args.phase_timeout_s
        )
        self.active_mission_id = None
        self.wait_pose(start, time.monotonic())
        return {"status": status.get("status")}


def _waypoint(xyzw: Sequence[float]) -> dict[str, Any]:
    """SDK navigation waypoints use rotation in ROS xyzw field order."""
    pose = target_pose(xyzw)
    pose["rotation"] = pose.pop("orientation")
    return pose
