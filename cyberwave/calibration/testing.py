"""Doubles and conformance helpers for testing a hand-eye driver integration.

Shipped in the package, not in the SDK's own ``tests/``, because a driver's test
suite cannot import from a wheel's test directory -- and the point of these is
that a *second* robot's driver reuses them rather than reinventing near-identical
fakes.

:class:`FakeHandEyeResult`'s attribute set is worth singling out: it is exactly
what :meth:`~cyberwave.calibration.flow.HandEyeFlow.snapshot` reads off a real
result, several fields via ``getattr``. Shipping it keeps the two in step, so a
new result field that the snapshot starts reporting shows up here too.

numpy is imported lazily so this module stays importable in an environment that
can render an alert but not run a solve.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


class FakeJointSource:
    """A :class:`~cyberwave.calibration.sources.JointSource` that never moves.

    Args:
        positions: What every read returns. Keys are *bus* names, so they must
            match the keys of the config's ``joint_name_map`` -- mismatched keys
            are how a real joint-mapping bug shows up, and this fake reproduces
            it faithfully rather than papering over it.
    """

    def __init__(self, positions: Optional[Mapping[str, float]] = None) -> None:
        self.positions: Optional[Dict[str, float]] = (
            dict(positions) if positions is not None else None
        )
        #: Reads served, so a caller can assert the flow bracketed a frame grab
        #: with two of them.
        self.reads = 0
        self.disconnected = False

    def read_joint_positions(self) -> Optional[Dict[str, float]]:
        self.reads += 1
        return self.positions

    def disconnect(self) -> None:
        self.disconnected = True


class FakeFrameSource:
    """A :class:`~cyberwave.calibration.sources.FrameSource` of blank frames.

    Deliberately models no staleness handling. That is each adapter's private
    technique for honouring "this frame postdates the arm stopping"; a fake that
    reproduced one camera's buffer drain would assert the mechanism instead of
    the contract.
    """

    def __init__(self, shape: Tuple[int, int, int] = (720, 1280, 3)) -> None:
        self._shape = shape
        #: Frames served, one per capture.
        self.grabs = 0
        self.released = False

    def grab_fresh_frame(self) -> Any:
        import numpy as np

        self.grabs += 1
        return np.zeros(self._shape, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeLeaveOneOut:
    """The stability figure the verdict is keyed on, in the real result's shape."""

    def __init__(
        self,
        max_stddev_m: float = 0.0026,
        worst_sample_index: int = 2,
        subset_count: int = 5,
    ) -> None:
        self.max_stddev_m = max_stddev_m
        self.translation_stddev_m = (0.0007, 0.0006, max_stddev_m)
        self.worst_sample_index = worst_sample_index
        self.subset_count = subset_count


class FakeHandEyeResult:
    """Every attribute the snapshot reads off a real ``HandEyeResult``.

    Pass ``leave_one_out=None`` to model a solve whose subsets were all
    degenerate, which is the case that falls back to the residual-only verdict.
    """

    def __init__(
        self,
        *,
        method: str = "park",
        sample_count: int = 5,
        residual_translation_m: float = 0.0029,
        residual_rotation_deg: float = 0.31,
        leave_one_out: Any = "default",
    ) -> None:
        self.method = method
        self.sample_count = sample_count
        self.residual_translation_m = residual_translation_m
        self.max_residual_translation_m = residual_translation_m * 1.4
        self.residual_rotation_deg = residual_rotation_deg
        self.max_residual_rotation_deg = residual_rotation_deg * 1.4
        self.relative_rotation_deg = 1.42
        self.relative_translation_m = 0.0051
        self.relative_translation_ratio = 0.09
        self.gripper_to_camera = None
        self.leave_one_out = (
            FakeLeaveOneOut() if leave_one_out == "default" else leave_one_out
        )


class FakeHandEyeSession:
    """Stands in for :class:`~cyberwave.calibration.handeye.HandEyeSession`.

    Args:
        detect: When false, every ``add_sample`` raises
            :class:`~cyberwave.calibration.handeye.BoardNotDetectedError` with
            *reject_reason* -- the wording a driver forwards to the operator, so
            it is part of the contract.
    """

    def __init__(self, detect: bool = True, reject_reason: str = "no board") -> None:
        self.sample_count = 0
        self.pending_count = 0
        self.detect = detect
        self.reject_reason = reject_reason
        self.cleared = 0
        #: Joint angles of the last accepted capture, under URDF names, so a test
        #: can assert the rename happened.
        self.last_joint_positions: Optional[Dict[str, float]] = None

    @property
    def captured_count(self) -> int:
        return self.sample_count + self.pending_count

    def add_sample(self, *, joint_positions=None, image=None, **kwargs: Any) -> None:
        from .handeye import BoardNotDetectedError

        if not self.detect:
            raise BoardNotDetectedError(self.reject_reason)
        self.last_joint_positions = dict(joint_positions or {})
        self.sample_count += 1

    def solve(self, **kwargs: Any) -> FakeHandEyeResult:
        return FakeHandEyeResult()

    def clear(self) -> None:
        self.cleared += 1
        self.sample_count = 0
        self.pending_count = 0


class FakeCalibrationHandle:
    """Records what a driver wrote to ``camera_twin.calibration``."""

    def __init__(self) -> None:
        self.set_calls: List[Dict[str, Any]] = []

    def set(self, result: Any, **kwargs: Any) -> Dict[str, Any]:
        self.set_calls.append({"result": result, **kwargs})
        return {}


class FakeCameraTwin:
    """A docked camera twin, enough for attachment validation and apply."""

    def __init__(
        self,
        *,
        attach_to_twin_uuid: str = "arm-uuid",
        attach_to_link: str = "gripper",
    ) -> None:
        self.sensors = [{"type": "rgb", "parameters": {"fovy": 58.0}}]
        self.calibration = FakeCalibrationHandle()
        self._data = {
            "attach_to_twin_uuid": attach_to_twin_uuid,
            "attach_to_link": attach_to_link,
        }

    def _data_get(self, key: str) -> Any:
        return self._data.get(key)


# --- conformance -----------------------------------------------------------


def assert_joint_source_conforms(source: Any) -> None:
    """Check *source* honours the ``JointSource`` obligations.

    Semantic, not structural: a name check verifies nothing about behaviour. Call
    this against a real adapter (in a hardware-marked test, if it needs the
    device) so a second robot's implementation is cheap to trust.
    """
    positions = source.read_joint_positions()
    if positions is not None:
        if not isinstance(positions, Mapping):
            raise AssertionError(
                f"read_joint_positions returned {type(positions).__name__}, "
                "expected a mapping of joint name to radians, or None when the "
                "bus is gone."
            )
        for name, value in positions.items():
            if not isinstance(name, str):
                raise AssertionError(  # noqa: TRY004 - a test assertion, not a guard
                    f"joint key {name!r} is not a string"
                )
            if not isinstance(value, (int, float)):
                raise AssertionError(  # noqa: TRY004 - a test assertion, not a guard
                    f"joint {name!r} value {value!r} is not a number"
                )

    # Two consecutive reads must both work: the flow brackets every frame grab
    # with a pair to prove the arm did not move during it.
    source.read_joint_positions()

    # Idempotent teardown -- the flow calls this from a different thread than an
    # in-flight capture on purpose, so a wedged read can be released.
    source.disconnect()
    source.disconnect()


def assert_frame_source_conforms(source: Any) -> None:
    """Check *source* honours the ``FrameSource`` obligations."""
    frame = source.grab_fresh_frame()
    if frame is None:
        raise AssertionError(
            "grab_fresh_frame returned None; raise on failure instead so the flow "
            "can surface a coded error."
        )
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) != 3:
        raise AssertionError(
            f"grab_fresh_frame returned shape {shape!r}, expected a 3-axis "
            "(height, width, channels) array."
        )

    # A second grab must also succeed: a run takes a dozen captures.
    second = source.grab_fresh_frame()
    if getattr(second, "shape", None) != shape:
        raise AssertionError(
            "grab_fresh_frame returned a different shape on the second call. The "
            "flow refuses a mid-run mode change, because intrinsics fitted at one "
            "size do not apply at another."
        )

    source.release()
    source.release()
