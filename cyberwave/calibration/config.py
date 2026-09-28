"""Per-robot configuration for the edge-side hand-eye flow.

Everything here is a property of *a particular arm, camera and board* rather than
of hand-eye calibration itself, which is why it is configuration and not
constants. The quality thresholds in particular were measured on hardware — see
the field docs — and a value measured on one arm is not a default for another.

**Almost nothing has a default, on purpose.** A library default for
``good_stability_m`` would be a measurement nobody took, presented as one they
did; requiring it forces whoever adds a second arm to look at their own hardware.
The exceptions are the two fields that are genuinely about presentation rather
than about the robot: ``target_samples`` and ``button_flow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..exceptions import CyberwaveValidationError
from .board import Board
from .handeye import MIN_SAMPLES

#: What the UI counts toward. Not a hard cap — capturing past it is allowed, and
#: more well-spread views only help.
DEFAULT_TARGET_SAMPLES = 12


@dataclass(frozen=True)
class HandEyeFlowConfig:
    """Everything the flow needs to know about one robot's hand-eye setup."""

    #: The docked camera twin being calibrated.
    camera_twin_uuid: str
    #: The arm the camera is mounted on.
    arm_twin_uuid: str
    #: The link the camera is docked to, and the link ``base_to_gripper`` poses
    #: are measured about. Must equal the camera twin's ``attach_to_link``: the
    #: solved transform is expressed relative to whichever link the camera is
    #: docked to, so a mismatch yields a wrong answer with a *good* residual.
    #: Use :func:`~cyberwave.calibration.flow.resolve_docked_fk_frame` to derive
    #: it from the twin rather than trusting a command payload.
    fk_frame: str
    #: Bus joint name -> URDF joint name. A plain mapping rather than a name rule
    #: or a callable: it has to survive into a run recording, so a joint-mapping
    #: bug is visible in the file rather than inferred from a wrong pose. Joints
    #: that do not move the camera's link (a gripper jaw) are simply absent.
    joint_name_map: Mapping[str, str]
    #: Object with ``fk(joint_positions) -> 4x4``. ``BaseKinematicsManipulator``
    #: from the ``drivers`` extra satisfies it; so does anything else with that
    #: method. Duck-typed so this module needs no ``cyberwave.driver`` import,
    #: which is what keeps ``cyberwave.calibration`` free of pinocchio.
    kinematics: Any
    #: The calibration target. Already constructed, not a spec dict: parsing a
    #: JSON spec is the caller's business because only the caller knows what its
    #: own transport sends.
    board: Board

    # --- quality gates, metres --------------------------------------------
    #: Leave-one-out stability at or below which the verdict is "good". This is
    #: the primary gate: it re-solves with each sample dropped and takes the
    #: standard deviation of where the camera lands, so it measures whether the
    #: *answer* is stable rather than whether the observations merely agree.
    #: Measured on an SO-101: runs whose mount position was reproducible across
    #: independent sessions land at 1.9-3.5 mm; ill-conditioned ones reach
    #: 7.7-24 mm, almost entirely along the camera's optical axis.
    good_stability_m: float
    #: Above this the answer is unstable enough that applying it is likely worse
    #: than leaving the nominal offset in place.
    bad_stability_m: float
    #: Residual bar. A secondary check only: the residual cannot see a
    #: systematically wrong input (a mislabelled frame, a wrong square size)
    #: because such an error is self-consistent, and on the SO-101 it tracked
    #: working distance rather than answer quality.
    good_residual_m: float

    # --- capture ----------------------------------------------------------
    #: Joint movement across a frame grab above which the operator is warned, in
    #: radians. Not a refusal: the operator is hand-guiding the arm, and a
    #: threshold low enough to catch real drift would mostly fire on servo noise.
    settle_tolerance_rad: float
    #: Resolution asked of the camera, ``(width, height)``. The camera may refuse
    #: — intrinsics come from the frame actually received, so a substituted mode
    #: stays correct, just less forgiving of small markers.
    requested_capture_size: tuple[int, int]
    #: Distance the granted capture mode is judged at, in metres. The far end of
    #: the range an operator actually works at; closer always decodes better.
    working_distance_m: float

    #: What the UI counts toward.
    target_samples: int = DEFAULT_TARGET_SAMPLES
    #: ``metadata.buttons[].payload.flow`` discriminator, so a node's button
    #: router can tell this flow's presses from another's. The node owns its own
    #: command namespace, so this must match whatever its router compares against.
    button_flow: str = "handeye"
    #: Opaque round-trip of the caller's own board spec, carried in button
    #: payloads so a restart re-enters with the same board. Never parsed here.
    board_spec: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.joint_name_map:
            raise CyberwaveValidationError(
                "joint_name_map is empty, so no bus joint can be renamed to a URDF "
                "joint and forward kinematics cannot run. Map at least the joints "
                "that move the camera's link."
            )
        if self.target_samples < MIN_SAMPLES:
            raise CyberwaveValidationError(
                f"target_samples ({self.target_samples}) is below the {MIN_SAMPLES} "
                "captures a solve needs."
            )
        if self.bad_stability_m <= self.good_stability_m:
            raise CyberwaveValidationError(
                f"bad_stability_m ({self.bad_stability_m}) must exceed "
                f"good_stability_m ({self.good_stability_m}); otherwise no result "
                "can ever be merely marginal."
            )
        if not self.fk_frame:
            raise CyberwaveValidationError(
                "fk_frame is required: it names the link the solved transform is "
                "expressed relative to."
            )
        # Freeze the mapping so a caller cannot mutate it after construction and
        # silently change what a later capture renames. Matches ArmKinematicsConfig.
        object.__setattr__(
            self, "joint_name_map", MappingProxyType(dict(self.joint_name_map))
        )
        if self.board_spec is not None:
            object.__setattr__(
                self, "board_spec", MappingProxyType(dict(self.board_spec))
            )
        object.__setattr__(
            self, "requested_capture_size", tuple(self.requested_capture_size)
        )

    @property
    def urdf_joint_names(self) -> tuple[str, ...]:
        """URDF joints forward kinematics is run over, in map order.

        Derived rather than configured separately: two lists that have to agree
        is a latent bug, and this way a URDF name with no bus counterpart is
        impossible by construction rather than a capture-time failure.
        """
        return tuple(self.joint_name_map.values())
