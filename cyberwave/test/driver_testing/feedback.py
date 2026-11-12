"""Measured feedback, command tickets and the teleop load stream of a stress run.

Everything here judges the robot by what it reports, never by what was sent:
only ``source_type=edge`` joints and poses count as measured state, and a
command's result is the one naming the ``motion_id`` the driver gave it.
"""

from __future__ import annotations

import bisect
import itertools
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

from cyberwave_geometry import Quaternion, Vector3, quaternion

#: A pose as ``[x, y, z, qx, qy, qz, qw]``, the ``pose_xyzw`` convention.
PoseXYZW = list


def target_pose(xyzw: Sequence[float], dz: float = 0.0) -> dict[str, Any]:
    """``[x, y, z, qx, qy, qz, qw]`` -> the named ``set_pose`` pose dict."""
    values = list(xyzw)
    values[2] += dz
    return {
        "position": dict(zip("xyz", values[:3])),
        "orientation": dict(zip("xyzw", values[3:])),
    }


def euler_pose(xyzw: Sequence[float]) -> list[float]:
    """``[x, y, z, qx, qy, qz, qw]`` -> the ``ee_pose`` ``[x, y, z, roll, pitch, yaw]``."""
    rpy = quaternion.to_rpy(Quaternion.from_xyzw(list(xyzw[3:])))
    return list(xyzw[:3]) + [rpy.roll, rpy.pitch, rpy.yaw]


def offset_along_z(xyzw: Sequence[float], offset_z_m: float) -> list[float]:
    """Translate a pose along its own rotated local +Z (e.g. flange -> TCP)."""
    offset = quaternion.rotate(
        Quaternion.from_xyzw(list(xyzw[3:])), Vector3(x=0.0, y=0.0, z=offset_z_m)
    )
    return [xyzw[0] + offset.x, xyzw[1] + offset.y, xyzw[2] + offset.z] + list(xyzw[3:])


def rotation_angle(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle in rad between the orientations of two ``pose_xyzw`` poses."""
    dot = abs(sum(x * y for x, y in zip(a[3:], b[3:])))
    return 2 * math.acos(min(1.0, dot))


def _finite_numbers(values: Iterable[Any]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _normalized(values: Sequence[float]) -> list[float] | None:
    norm = math.sqrt(sum(v * v for v in values[3:]))
    if norm < 1e-9:
        return None
    return list(values[:3]) + [v / norm for v in values[3:]]


class Ticket(NamedTuple):
    """One sent command: its name and when it left (monotonic s)."""

    command: str
    sent: float


class Feedback:
    """Record raw edge feedback, never the SDK's optimistic target cache.

    ``receive`` takes four kinds:

    * ``joints`` -- ``joint/{uuid}/update``, decoded by *parse_joints*;
    * ``pose`` -- ``twin/{uuid}/pose`` (``pose_xyzw``). With *pose_frames* only
      those ``(frame_id, child_frame_id)`` pairs count; without, the first pair
      seen is kept and any other is rejected, so two frames never mix. The pose
      is moved by *tool_offset_z_m* along its local Z (flange -> TCP);
    * ``kinematics`` -- ``state/kinematics`` (``cartesian_pose``): the driver's
      measured TCP in its planning frame. Once one arrives it is the pose, and
      :attr:`pose_is_tcp` turns true;
    * ``motion`` -- ``state/motion``: motions by ``motion_id``, and every new
      result by its unique ``timestamp_ms``.
    """

    def __init__(
        self,
        parse_joints: Callable[[dict[str, Any]], Any],
        tool_offset_z_m: float = 0.0,
        pose_frames: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self.parse_joints = parse_joints
        self.tool_offset_z_m = tool_offset_z_m
        self.pose_frames = list(pose_frames) if pose_frames else None
        #: The ``(frame_id, child_frame_id)`` pair the pose is in, once known.
        self.pose_frame: tuple[str, str] | None = None
        #: True once the pose comes from ``state/kinematics``: the TCP itself,
        #: in the frame ``set_pose`` plans in, so absolute targets are safe.
        self.pose_is_tcp = False
        self.condition = threading.Condition()
        self.latest: dict[str, tuple[float, Any]] = {}
        #: Every new state/motion result, in arrival order: (received, result).
        self.terminals: list[tuple[float, dict[str, Any]]] = []
        self._seen_stamps: set[tuple[str, Any]] = set()
        self._claimed: set[int] = set()
        #: Every motion the driver started: (received, command, motion_id). A
        #: ticket's motion is the first new one with its command after it was
        #: sent; its result is the one naming that motion_id.
        self.motions: list[tuple[float, Any, Any]] = []
        self._seen_motions: set[Any] = set()
        self._ticket_motions: dict[Ticket, Any] = {}
        #: (received, motion_id) whenever a motion is reported executing.
        self.executing: list[tuple[float, Any]] = []
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        kind: str,
        payload: Any,
        positions: dict[str, float] | None = None,
        *,
        pose: list[float] | None = None,
    ) -> float:
        event = {
            "monotonic_s": time.monotonic(),
            "timestamp_ms": time.time_ns() // 1_000_000,
            "kind": kind,
            "payload": payload,
        }
        if positions is not None:
            event["positions"] = positions.copy()
        if pose is not None:
            event["pose"] = list(pose)
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()
        return event["monotonic_s"]

    def _pose_from_twin_pose(self, payload: dict[str, Any]) -> list[float] | None:
        if self.pose_is_tcp:
            return None
        frames = (payload.get("frame_id"), payload.get("child_frame_id"))
        if self.pose_frames is not None:
            if frames not in self.pose_frames:
                return None
        elif self.pose_frame is not None and frames != self.pose_frame:
            return None
        values = payload.get("pose_xyzw", [])
        if (
            not isinstance(values, list)
            or len(values) != 7
            or not _finite_numbers(values)
        ):
            return None
        values = _normalized(values)
        if values is None:
            return None
        self.pose_frame = frames  # type: ignore[assignment]
        if self.tool_offset_z_m:
            values = offset_along_z(values, self.tool_offset_z_m)
        return values

    def _pose_from_kinematics(self, payload: dict[str, Any]) -> list[float] | None:
        pose = payload.get("cartesian_pose")
        if not isinstance(pose, dict):
            return None
        position = pose.get("position") or {}
        orientation = pose.get("orientation") or {}
        try:
            raw = [position[k] for k in "xyz"] + [orientation[k] for k in "xyzw"]
        except (KeyError, TypeError):
            return None
        if not _finite_numbers(raw):
            return None
        values = _normalized(raw)
        if values is None:
            return None
        self.pose_is_tcp = True
        self.pose_frame = (str(pose.get("reference_frame") or ""), "tcp")
        return values

    def receive(self, kind: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if kind in ("joints", "pose") and payload.get("source_type") != "edge":
            return
        if kind == "kinematics" and payload.get("source_type", "edge") != "edge":
            return
        if kind == "joints":
            values = self.parse_joints(payload).positions
            if not values or not all(math.isfinite(v) for v in values.values()):
                return
        elif kind in ("pose", "kinematics"):
            with self.condition:
                values = (
                    self._pose_from_twin_pose(payload)
                    if kind == "pose"
                    else self._pose_from_kinematics(payload)
                )
            if values is None:
                return
            kind = "pose"
        else:
            values = payload
        received = self.record(
            kind,
            payload,
            values if kind == "joints" else None,
            # The measured pose as the harness reads it (TCP-converted).
            pose=values if kind == "pose" else None,
        )
        with self.condition:
            self.latest[kind] = (received, values)
            if kind == "motion":
                self._receive_motion(received, payload)
            self.condition.notify_all()

    def _receive_motion(self, received: float, payload: dict[str, Any]) -> None:
        motion_id = payload.get("motion_id")
        if motion_id is not None and motion_id not in self._seen_motions:
            self._seen_motions.add(motion_id)
            self.motions.append((received, payload.get("command"), motion_id))
        if payload.get("motion_state") == "executing":
            self.executing.append((received, motion_id))
        # A cancellation sets both fields at once; last_result wins.
        for field in ("last_result", "last_refusal"):
            result = payload.get(field)
            stamp = (result or {}).get("timestamp_ms")
            if result and stamp is not None and (field, stamp) not in self._seen_stamps:
                fresh = [(field, stamp)]
                other_field = (
                    "last_refusal" if field == "last_result" else "last_result"
                )
                other = payload.get(other_field)
                if other and other.get("timestamp_ms") is not None:
                    fresh.append((other_field, other["timestamp_ms"]))
                self._seen_stamps.update(fresh)
                self.terminals.append((received, result))
                break

    def motion_id(self, ticket: Ticket) -> Any:
        """The motion the driver started for *ticket*, once it has reported it."""
        if ticket in self._ticket_motions:
            return self._ticket_motions[ticket]
        taken = set(self._ticket_motions.values())
        for received, command, motion_id in self.motions:
            if (
                received >= ticket.sent
                and command == ticket.command
                and motion_id not in taken
            ):
                self._ticket_motions[ticket] = motion_id
                return motion_id
        return None

    def terminal(
        self, ticket: Ticket, claim: bool = True
    ) -> tuple[float, dict[str, Any]] | None:
        """*ticket*'s result: the one naming its motion, or a refusal before it started."""
        motion_id = self.motion_id(ticket)
        for index, (received, result) in enumerate(self.terminals):
            if index in self._claimed or received < ticket.sent:
                continue
            if motion_id is not None:
                mine = result.get("motion_id") == motion_id
            else:
                mine = (
                    "motion_id" not in result
                    and result.get("command") == ticket.command
                )
            if mine:
                if claim:
                    self._claimed.add(index)
                return received, result
        return None

    def snapshot(self, kind: str, max_age_s: float) -> Any:
        values = None

        def fresh() -> bool:
            nonlocal values
            received, current = self.latest.get(kind, (0.0, None))
            if current is None or time.monotonic() - received > max_age_s:
                return False
            values = current.copy()
            return True

        self.wait(fresh, 10.0, f"No fresh measured {kind} feedback within 10s")
        return values

    def wait(self, predicate: Callable[[], bool], timeout_s: float, label: str) -> None:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(label)
                self.condition.wait(min(remaining, 0.05))


class Teleop:
    """Dedicated monotonic publisher: skipped ticks never become a burst.

    Streams a quantized sine on ``harness.args.sine_joints`` around *baseline*,
    alternating sign per joint, ramped in and out so it starts and ends there.
    """

    def __init__(
        self,
        harness: Any,
        baseline: dict[str, float],
        amplitude: float,
        duration_s: float,
    ) -> None:
        self.harness = harness
        self.baseline = baseline.copy()
        self.amplitude = amplitude
        self.duration_s = duration_s
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.start_s = 0.0
        self.end_s = 0.0
        self.thread = threading.Thread(target=self.run, name="cyberwave-stress-teleop")

    def start(self) -> Teleop:
        self.thread.start()
        return self

    def run(self) -> None:
        args = self.harness.args
        previous = self.baseline.copy()
        self.start_s = time.monotonic()
        deadline = self.start_s
        period = 1 / args.teleop_rate_hz
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                elapsed = now - self.start_s
                if elapsed >= self.duration_s:
                    break
                values = self.baseline.copy()
                for index, name in enumerate(args.sine_joints):
                    # Start and finish at the baseline; ramp removes a step on entry.
                    ramp = min(1.0, elapsed, self.duration_s - elapsed)
                    offset = (
                        self.amplitude
                        * ramp
                        * math.sin(2 * math.pi * args.sine_frequency_hz * elapsed)
                        * (1 if index % 2 == 0 else -1)
                    )
                    values[name] += (
                        round(offset / args.min_teleop_delta_rad)
                        * args.min_teleop_delta_rad
                    )
                if any(
                    abs(values[name] - previous[name])
                    >= args.min_teleop_delta_rad - 1e-9
                    for name in args.sine_joints
                ):
                    self.harness.set_joints(values)
                    previous = values.copy()
                deadline += period
                now = time.monotonic()
                if deadline <= now:
                    deadline += (math.floor((now - deadline) / period) + 1) * period
                self.stop_event.wait(max(0.0, deadline - now))
        except Exception as exc:  # noqa: BLE001 - re-raised by close()
            self.error = exc
        finally:
            self.end_s = time.monotonic()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join()  # No publisher can race the final stop.
        if self.error:
            raise self.error


def _positions(event: dict[str, Any]) -> dict[str, float]:
    return event.get("positions", event["payload"].get("positions", event["payload"]))


def tracking_metrics(
    events: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    names: Sequence[str],
    settle_s: float = 0.6,
) -> dict[str, Any]:
    """Diagnostics for a teleop stream: setpoint error, travel, feedback gaps.

    Never a pass gate: MQTT can delay or coalesce samples while the arm keeps
    moving. Phases pass on measured endpoints instead.
    """
    targets = [
        e
        for e in events
        if e["kind"] == "target" and start_s <= e["monotonic_s"] <= end_s
    ]
    samples = [
        e
        for e in events
        if e["kind"] == "joints" and start_s <= e["monotonic_s"] <= end_s
    ]
    times = [e["monotonic_s"] for e in targets]
    errors = []
    settled_errors = []
    for sample in samples:
        index = bisect.bisect_right(times, sample["monotonic_s"]) - 1
        if index < 0:
            continue
        measured = _positions(sample)
        for name in names:
            if name in measured and name in targets[index]["payload"]:
                error = measured[name] - targets[index]["payload"][name]
                errors.append(error)
                if sample["monotonic_s"] - times[index] >= settle_s:
                    settled_errors.append(error)
    gaps = [b["monotonic_s"] - a["monotonic_s"] for a, b in itertools.pairwise(samples)]
    tx_gaps = [b - a for a, b in itertools.pairwise(times)]
    travel: dict[str, dict[str, Any]] = {}
    for name in names:
        measured_values = [
            v for v in (_positions(e).get(name) for e in samples) if v is not None
        ]
        commanded = [e["payload"][name] for e in targets if name in e["payload"]]
        travel[name] = {
            "measured_peak_to_peak_rad": max(measured_values) - min(measured_values)
            if measured_values
            else 0,
            "target_peak_to_peak_rad": max(commanded) - min(commanded)
            if commanded
            else 0,
        }
        steps = []
        for a, b in itertools.pairwise(samples):
            va = _positions(a).get(name)
            vb = _positions(b).get(name)
            if va is not None and vb is not None:
                steps.append(abs(vb - va))
        # MQTT packets can arrive in bursts. Dividing position changes by
        # arrival gaps would report fictitious physical joint velocities.
        travel[name]["max_measured_step_rad"] = max(steps, default=None)
    return {
        "feedback_samples": len(samples),
        "published_targets": len(targets),
        "tracking_rmse_rad": math.sqrt(sum(e * e for e in errors) / len(errors))
        if errors
        else None,
        "settled_samples": len(settled_errors),
        "settled_rmse_rad": (
            math.sqrt(sum(e * e for e in settled_errors) / len(settled_errors))
            if settled_errors
            else None
        ),
        "tracking_max_error_rad": max(map(abs, errors), default=None),
        "max_feedback_gap_s": max(gaps, default=None),
        "max_publish_gap_s": max(tx_gaps, default=None),
        "travel": travel,
    }
