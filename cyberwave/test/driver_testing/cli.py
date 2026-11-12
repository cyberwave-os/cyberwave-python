"""``cyberwave-stress-test``: capability-driven stress test of any arm driver, on its real twin.

    cyberwave-stress-test --uuid <twin-uuid> --list                 # the plan; nothing moves
    cyberwave-stress-test --uuid <twin-uuid> --yes                  # every section the twin supports
    cyberwave-stress-test --uuid <twin-uuid> --sections joints,planned --yes

It reads the twin's capabilities, command catalog, universal schema joints and
limits, and ``state/motion``; prints which sections it will run and why the
others are skipped; and only then, after confirmation, moves the robot. See
the "Stress-testing your driver" page of the driver docs.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .discovery import TwinProfile, discover
from .feedback import Feedback
from .harness import StressHarness
from .registry import CommandCoverage, PlanEntry, command_coverage, resolve_plan
from .report import jsonable_config, write_report

logger = logging.getLogger(__name__)

#: Numeric knobs: (default, help). Distances in m, angles in rad, times in s.
NUMERIC_OPTIONS: dict[str, tuple[float, str]] = {
    "teleop-rate-hz": (5.0, "joint teleop publish cap (at most 10)"),
    "min-teleop-delta-rad": (0.1, "smallest arm joint target change sent"),
    "amplitude-rad": (0.2, "joint step and sine amplitude"),
    "sine-frequency-hz": (0.1, "sine load frequency"),
    "sine-duration-s": (10.0, "sine load duration"),
    "cartesian-step-m": (0.015, "Cartesian step of jogs and planned moves"),
    "takeover-step-m": (
        0.05,
        "length of a move that is replaced, stopped or overlapped",
    ),
    "preemption-step-m": (0.06, "length of a move a joint target preempts"),
    "rotation-step-rad": (0.12, "tool and base rotation step"),
    "waypoint-step-m": (0.03, "navigation waypoint step"),
    "grasp-approach-m": (0.03, "approach distance of the executed grasp"),
    "probe-box-m": (0.1, "edge of the fixture boxes placed at the tool"),
    "joint-limit-margin-rad": (0.05, "targets stay this far inside the schema limits"),
    "max-joint-speed-rad-s": (
        0.25,
        "refuse a sine whose peak joint speed exceeds this",
    ),
    "max-cartesian-step-m": (0.08, "refuse any Cartesian step longer than this"),
    "max-rotation-step-rad": (0.3, "refuse any rotation step larger than this"),
    "joint-tolerance-rad": (0.035, "measured joint arrival tolerance"),
    "position-tolerance-m": (0.008, "measured TCP arrival tolerance"),
    "orientation-tolerance-rad": (0.08, "measured orientation arrival tolerance"),
    "gripper-tolerance-m": (0.002, "measured jaw endpoint repeatability"),
    "gripper-min-travel": (0.005, "smallest measured jaw travel from release to grip"),
    "phase-timeout-s": (120.0, "observation deadline of one wait"),
    "feedback-timeout-s": (2.0, "measured feedback older than this is stale"),
    "manual-lease-s": (0.7, "pause after joint teleop before commands"),
    "no-resume-s": (1.0, "hold that proves a replaced target does not resume"),
}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _xyz(value: str) -> tuple[float, float, float]:
    numbers = [float(item) for item in _csv(value)]
    if len(numbers) != 3 or not all(math.isfinite(v) for v in numbers):
        raise argparse.ArgumentTypeError("expected three finite numbers: x,y,z")
    return numbers[0], numbers[1], numbers[2]


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cyberwave-stress-test",
        description="Capability-driven stress test of an arm driver on its twin (moves the real robot).",
    )
    parser.add_argument("--uuid", required=True, help="twin UUID")
    parser.add_argument(
        "--sections",
        type=_csv,
        default=None,
        help="comma-separated section ids, areas (joints, planned, ...) or @backend",
    )
    parser.add_argument(
        "--skip", type=_csv, default=[], help="sections, areas or @backend to leave out"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the resolved plan and exit; moves nothing",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm moving a live robot without a prompt",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="after a failed section: stop, verify a measured hold, go on (preflight stays mandatory)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="report.json path or a directory (default: a new temporary directory)",
    )
    parser.add_argument(
        "--allow-untested",
        type=_csv,
        default=[],
        help="catalog commands allowed to stay untested (default: none; any other untested command fails the run)",
    )
    parser.add_argument(
        "--backends",
        type=_csv,
        default=[],
        help="planned-motion backends to run besides the driver's default (e.g. pinocchio)",
    )
    parser.add_argument(
        "--arm-joints",
        type=_csv,
        default=None,
        help="arm joints (default: from the universal schema)",
    )
    parser.add_argument(
        "--gripper-joint",
        default=None,
        help="gripper joint (default: from the universal schema)",
    )
    parser.add_argument(
        "--sine-joints",
        type=_csv,
        default=None,
        help="joints the sine load drives (default: the arm)",
    )
    parser.add_argument(
        "--navigate-link",
        default=None,
        help="link the navigate service moves; opts into waypoint tests when set",
    )
    parser.add_argument(
        "--workspace-min",
        type=_xyz,
        default=(-0.8, -0.8, 0.02),
        help="Cartesian box lower corner x,y,z in the pose frame (m); negative: --workspace-min=-0.5,-0.5,0.05",
    )
    parser.add_argument(
        "--workspace-max",
        type=_xyz,
        default=(0.8, 0.8, 1.2),
        help="Cartesian box upper corner x,y,z in the pose frame (m)",
    )
    for name, (default, help_text) in NUMERIC_OPTIONS.items():
        parser.add_argument(
            "--" + name,
            type=float,
            default=default,
            help=f"{help_text} (default {default})",
        )
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if any(lo >= hi for lo, hi in zip(args.workspace_min, args.workspace_max)):
        parser.error("--workspace-min must be below --workspace-max on every axis")
    if args.teleop_rate_hz > 10:
        parser.error("--teleop-rate-hz must be at most 10")
    if args.sine_duration_s < 4:
        parser.error("--sine-duration-s must be at least 4")
    if args.amplitude_rad < args.min_teleop_delta_rad:
        parser.error("--amplitude-rad must cover --min-teleop-delta-rad")
    peak = args.amplitude_rad * 2 * math.pi * args.sine_frequency_hz
    if peak > args.max_joint_speed_rad_s:
        parser.error(
            f"sine peak joint speed {peak:.3f} rad/s exceeds --max-joint-speed-rad-s"
        )
    for key in (
        "cartesian_step_m",
        "takeover_step_m",
        "preemption_step_m",
        "waypoint_step_m",
        "grasp_approach_m",
    ):
        if getattr(args, key) > args.max_cartesian_step_m:
            parser.error(f"--{key.replace('_', '-')} exceeds --max-cartesian-step-m")
    if args.rotation_step_rad > args.max_rotation_step_rad:
        parser.error("--rotation-step-rad exceeds --max-rotation-step-rad")
    if args.report is None:
        args.report = (
            Path(tempfile.gettempdir())
            / f"stress-{args.uuid[:8]}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
    return args


def print_plan(
    profile: TwinProfile, plan: Sequence[PlanEntry], out: Any = None
) -> None:
    out = out or sys.stdout
    print(f"Twin {profile.uuid}", file=out)
    print(
        f"  arm joints:     {', '.join(j.name for j in profile.arm_joints) or '-'}",
        file=out,
    )
    print(f"  gripper joint:  {profile.gripper_joint or '-'}", file=out)
    print(
        f"  backends:       {', '.join(profile.backends) or '-'}"
        + (
            ""
            if profile.backend_available or not profile.backends
            else " (default reports unavailable)"
        ),
        file=out,
    )
    print(f"  feedback:       {', '.join(sorted(profile.topics)) or '-'}", file=out)
    width = max((len(e.id) for e in plan), default=0)
    print("\nPlan:", file=out)
    for entry in plan:
        mark = "run " if entry.runs else "skip"
        tail = entry.section.summary if entry.runs else entry.skipped
        print(f"  {mark}  {entry.id.ljust(width)}  {tail}", file=out)
    print(f"\n{sum(e.runs for e in plan)} of {len(plan)} sections will run.", file=out)
    print_coverage(command_coverage(profile, plan), out)


def print_coverage(rows: Sequence[CommandCoverage], out: Any = None) -> None:
    """Every catalog command: tested by which sections, or why not."""
    out = out or sys.stdout
    width = max((len(r.command) for r in rows), default=0)
    print("\nCatalog commands:", file=out)
    for row in rows:
        detail = ", ".join(row.sections) if row.sections else row.reason
        print(
            f"  {row.status.ljust(12)}  {row.command.ljust(width)}  {detail}", file=out
        )


def untested(rows: Sequence[CommandCoverage], allowed: Sequence[str]) -> list[str]:
    """Untested commands that ``--allow-untested`` does not name."""
    return [
        r.command for r in rows if r.status == "untested" and r.command not in allowed
    ]


def confirm(args: argparse.Namespace) -> bool:
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print("Refusing to move a live robot without --yes.", file=sys.stderr)
        return False
    answer = input(
        f"\nThis moves twin {args.uuid}. Workspace clear, e-stop in reach? Type 'yes': "
    )
    return answer.strip().lower() == "yes"


def run_plan(harness: StressHarness, plan: Sequence[PlanEntry]) -> None:
    """Preflight, every planned section in order (skips recorded), measured return."""
    harness.phase(
        "preflight at the measured start",
        harness.preflight,
        required=True,
        section="preflight",
    )
    for entry in plan:
        if not entry.runs:
            harness.phases.append(
                {
                    "name": entry.id,
                    "section": entry.id,
                    "outcome": "skipped",
                    "reason": entry.skipped,
                }
            )
            continue
        harness.phase(
            f"{entry.id}: {entry.section.summary}",
            _runner(harness, entry),
            section=entry.id,
        )
    if "joints" in getattr(harness.profile, "topics", ()):
        harness.phase(
            "return to the measured start", harness.return_to_start, section="return"
        )


def _runner(harness: StressHarness, entry: PlanEntry) -> Callable[[], Any]:
    kwargs = {"backend": entry.backend} if entry.backend is not None else {}
    return lambda: entry.section.body(harness, **kwargs)


def _subscribe(client: Any, twin: Any, feedback: Feedback) -> list[str]:
    prefix = client.config.topic_prefix or ""
    topics = []
    for kind, path in (
        ("joints", f"cyberwave/joint/{twin.uuid}/update"),
        ("pose", f"cyberwave/twin/{twin.uuid}/pose"),
        ("kinematics", f"cyberwave/twin/{twin.uuid}/state/kinematics"),
        ("motion", f"cyberwave/twin/{twin.uuid}/state/motion"),
    ):
        topic = prefix + path
        client.mqtt.subscribe(
            topic,
            lambda payload, kind=kind: feedback.receive(kind, payload),
            subscriber_key="cyberwave-stress",
        )
        topics.append(topic)
    return topics


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    from cyberwave import Cyberwave
    from cyberwave.data.state_representation import parse_joint_mqtt_payload

    client = Cyberwave(mode="live", source_type="tele")
    feedback = Feedback(parse_joint_mqtt_payload)
    harness: StressHarness | None = None
    subscriptions: list[str] = []
    profile: TwinProfile | None = None
    plan: list[PlanEntry] = []
    outcome, error = "failed", None
    try:
        client.mqtt.connect()
        twin = client.twin(twin_id=args.uuid)
        subscriptions = _subscribe(client, twin, feedback)
        # The backend may omit state/* from its catalog. The subscribed stream
        # is the source of truth for planned-motion support.
        profile = discover(
            twin,
            arm_joints=args.arm_joints,
            gripper_joints=[args.gripper_joint] if args.gripper_joint else None,
        )
        if "set_pose" in profile.commands or "grasp" in profile.commands:
            try:
                feedback.wait(
                    lambda: "motion" in feedback.latest, 6.0, "no state/motion"
                )
            except TimeoutError:
                print(
                    "warning: no state/motion within 6 s; planned sections are skipped",
                    file=sys.stderr,
                )
        profile = discover(
            twin,
            motion_status=feedback.latest.get("motion", (0, None))[1],
            observed_topics=(
                set(feedback.latest)
                | ({"kinematics"} if feedback.pose_is_tcp else set())
                | ({"navigate"} if args.navigate_link else set())
            ),
            arm_joints=args.arm_joints,
            gripper_joints=[args.gripper_joint] if args.gripper_joint else None,
            extra_backends=args.backends,
        )
        plan = resolve_plan(profile, sections=args.sections, skip=args.skip)
        print_plan(profile, plan)
        if args.list:
            return 0
        if not any(e.runs for e in plan):
            print("Nothing to run.", file=sys.stderr)
            return 1
        gaps = untested(command_coverage(profile, plan), args.allow_untested)
        if gaps:
            print(
                f"Refusing to run: no section can test {', '.join(gaps)}. Every command the driver "
                "exposes must be tested; name the ones you accept untested with --allow-untested.",
                file=sys.stderr,
            )
            return 1
        if not confirm(args):
            return 1
        unknown = [
            n
            for n in args.sine_joints or ()
            if n not in {j.name for j in profile.arm_joints}
        ]
        if unknown:
            raise ValueError(f"--sine-joints not among the arm joints: {unknown}")
        args.sine_joints = list(
            args.sine_joints or (j.name for j in profile.arm_joints)
        )
        harness = StressHarness(client, twin, feedback, args, profile)
        run_plan(harness, plan)
        failed = [p["name"] for p in harness.phases if p.get("outcome") == "failed"]
        # A section that skipped itself at run time tested nothing.
        coverage = command_coverage(profile, plan, harness.phases)
        print_coverage(coverage)
        gaps = untested(coverage, args.allow_untested)
        problems = ([f"Failed sections: {failed}"] if failed else []) + (
            [f"Untested commands: {gaps}"] if gaps else []
        )
        outcome = "failed" if problems else "passed"
        error = "; ".join(problems) or None
    except KeyboardInterrupt:
        outcome, error = "interrupted", "Ctrl-C"
    except Exception as exc:  # noqa: BLE001 - reported, and the robot is stopped below
        error = f"{type(exc).__name__}: {exc}"
        print(f"FAILED: {error}", flush=True)
    finally:
        if harness is not None:
            try:
                if harness.stream:
                    harness.stream.close()
            except Exception as exc:  # noqa: BLE001
                outcome, error = (
                    "failed",
                    f"{error or 'Cleanup'}; teleop shutdown: {exc}",
                )
            try:
                harness.stop_navigation()
            except Exception as exc:  # noqa: BLE001
                outcome, error = "failed", f"{error or 'Cleanup'}; mission stop: {exc}"
            try:
                harness.command("stop")  # Every run, Ctrl-C included, ends in stop.
            except Exception as exc:  # noqa: BLE001
                outcome, error = (
                    "failed",
                    f"Stop could not be sent: {exc}; previous error: {error}",
                )
        for topic in subscriptions:
            try:
                client.mqtt.unsubscribe(topic, subscriber_key="cyberwave-stress")
            except Exception:  # noqa: BLE001 - disconnecting anyway
                logger.warning("could not unsubscribe %s", topic)
        try:
            client.disconnect()
        finally:
            if harness is not None:
                directory = write_report(
                    args.report,
                    outcome=outcome,
                    error=error,
                    config=jsonable_config(args),
                    phases=harness.phases,
                    events=list(feedback.events),
                    joint_names=[
                        *harness.arm_joints,
                        *(
                            [profile.gripper_joint]
                            if profile and profile.gripper_joint
                            else []
                        ),
                    ],
                    plan=[e.to_json() for e in plan],
                    profile=profile.to_json() if profile else None,
                    coverage=[
                        r.to_json()
                        for r in command_coverage(profile, plan, harness.phases)
                    ]
                    if profile
                    else None,
                    skipped=[
                        {"section": e.id, "reason": e.skipped}
                        for e in plan
                        if not e.runs
                    ],
                )
                print(f"\n{outcome.upper()}: {directory.resolve()}", flush=True)
    return 0 if outcome == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
