"""Which robot description is loaded, and is it the right one for the arm that moves?

Two questions every arm driver has to answer before it trusts a Cartesian pose,
and that nothing else in the stack answers:

1. **Which description does the solver load?** A driver may ship a vendor URDF,
   a Cyberwave overlay (xacro) that adds a grasp TCP and payload geometry to it,
   or accept an operator-supplied file. :func:`resolve_runtime_description`
   picks one with a fixed precedence and records where it came from, so a
   collision model built from "a robot with nothing on its wrist" is visible
   rather than silent.

2. **Does that description match the hardware?** Vendors ship description
   variants that differ by centimetres at the flange, and a configuration file
   selecting one is never compared against the firmware the arm reports.
   :func:`evaluate_hardware_compatibility` is the gate: the driver supplies the
   firmware string and a table of firmware -> accepted description fingerprints,
   and gets back a verdict with an operator-facing reason. It fails closed.

The fingerprint tables and the evidence behind each entry belong to the driver;
this module only defines the mechanism. Pure apart from :func:`expand_xacro`
(which shells out to the ``xacro`` binary): no ROS import, no config loading.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

#: What :func:`urdf_fingerprint` returns for an empty or unreadable path.
UNKNOWN_FINGERPRINT = "unknown"


def urdf_fingerprint(urdf_path: str) -> str:
    """Short content hash (sha256[:12]) of the URDF, or ``"unknown"`` if unreadable.

    Content rather than path: the same file reachable at two mount points is the
    same kinematics, and two different files at the same path are not. Never
    raises -- an empty path (e.g. during manifest export, before the URDF is
    resolvable) fingerprints as ``"unknown"``.
    """
    if not urdf_path:
        return UNKNOWN_FINGERPRINT
    try:
        with open(urdf_path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except OSError:
        return UNKNOWN_FINGERPRINT


# ── Runtime description resolution ──────────────────────────────────────────


class DescriptionSource(Enum):
    """Where the loaded description came from, and what it therefore contains."""

    #: An explicit, operator-configured path. Whatever the operator pointed at.
    CONFIGURED = "configured"
    #: A Cyberwave overlay (xacro), expanded at startup. It carries whatever the
    #: overlay adds on top of the vendor description (TCP frame, payload).
    OVERLAY = "overlay"
    #: The bare vendor description, with none of the overlay's additions.
    VENDOR = "vendor"
    #: Nothing resolved.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ResolvedDescription:
    """The description FK/IK and the collision model load, with its provenance."""

    path: str
    source: DescriptionSource
    detail: str = ""
    #: Fingerprint of the description whose JOINT KINEMATICS this model has.
    #:
    #: Not the same thing as hashing ``path``. An overlay is expanded per
    #: configuration -- a measured TCP offset or a payload collision scale goes
    #: into it -- so the expanded file's hash changes with configuration and can
    #: never appear in a table of known-good variants. What the compatibility
    #: gate actually asks is "are these the arm's kinematics", and an overlay
    #: that only adds fixed frames and fixed collision geometry does not change
    #: them. So for an overlay this is the fingerprint of the vendor description
    #: it was built from.
    kinematic_fingerprint: str = ""
    #: Whether the overlay was expanded with its optional payload geometry.
    payload_present: bool = False

    @property
    def includes_payload_geometry(self) -> bool:
        """Whether this resolved overlay contains the optional payload."""
        return self.source is DescriptionSource.OVERLAY and self.payload_present

    def effective_fingerprint(self) -> str:
        """The fingerprint to gate on: the kinematic one, else the loaded file's."""
        return self.kinematic_fingerprint or urdf_fingerprint(self.path)


def expand_xacro(
    xacro_path: str,
    arguments: Mapping[str, str] | None = None,
    *,
    xacro_binary: str | None = None,
    timeout_s: float = 60.0,
) -> str:
    """Expand *xacro_path* with ``name:=value`` *arguments*; the URDF text.

    Raises ``FileNotFoundError`` when no ``xacro`` binary is available and
    ``subprocess.CalledProcessError`` / ``TimeoutExpired`` when expansion fails.
    """
    binary = xacro_binary or shutil.which("xacro")
    if not binary:
        raise FileNotFoundError("xacro is not installed")
    argv = [binary, xacro_path] + [f"{k}:={v}" for k, v in (arguments or {}).items()]
    return subprocess.run(
        argv, capture_output=True, text=True, check=True, timeout=timeout_s
    ).stdout


def resolve_runtime_description(
    *,
    urdf_path_override: str | None,
    vendor_urdf: Callable[[], str],
    overlay_xacro: str | None = None,
    xacro_args: Mapping[str, str] | None = None,
    payload_present: bool = False,
    cache_dir: str | None = None,
    output_name: str = "description.urdf",
) -> ResolvedDescription:
    """The description FK/IK and the collision model should load.

    Precedence: an explicit path wins (an operator pointing at a file means it),
    then the expanded overlay, then the bare vendor description. Never raises --
    an unresolvable description is reported as ``UNAVAILABLE``, and the caller's
    fingerprint and SRDF-link checks decide what that means for motion.

    * *vendor_urdf* resolves the vendor description path; it may raise (e.g. a
      package not on the ament index), which is reported, not propagated.
    * *overlay_xacro* is the overlay's path, or ``None`` when the driver has no
      overlay or it is not installed.
    * *xacro_args* are passed to the overlay as ``name:=value``.
    * *payload_present* is recorded on an overlay result; the driver knows which
      of its xacro arguments switches payload geometry on.
    * *cache_dir* is where the expanded URDF is written. Choose a directory from
      which the overlay's ``package://`` mesh URIs still resolve: pinocchio
      searches ancestors of the URDF's directory.

    Expanded at startup rather than baked at build time because the overlay's
    arguments (measured TCP offset, payload scale) are runtime configuration;
    baking them would require an image rebuild to change a measurement.
    """
    if urdf_path_override:
        return ResolvedDescription(
            str(urdf_path_override),
            DescriptionSource.CONFIGURED,
            kinematic_fingerprint=urdf_fingerprint(str(urdf_path_override)),
        )

    overlay = overlay_xacro if overlay_xacro and os.path.isfile(overlay_xacro) else None
    xacro_binary = shutil.which("xacro")
    if overlay and xacro_binary:
        arguments = dict(xacro_args or {})
        try:
            expanded = expand_xacro(overlay, arguments, xacro_binary=xacro_binary)
        except Exception as exc:  # noqa: BLE001 - any expansion failure falls back
            logger.warning(
                "could not expand the description overlay %s (%s); falling back "
                "to the vendor description, which has none of its additions",
                overlay,
                exc,
            )
        else:
            directory = cache_dir or os.path.join(
                tempfile.gettempdir(), "cyberwave", "kinematics.description"
            )
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, output_name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(expanded)
            rendered = " ".join(f"{k}={v}" for k, v in arguments.items())
            return ResolvedDescription(
                path,
                DescriptionSource.OVERLAY,
                f"expanded from {overlay}" + (f" with {rendered}" if rendered else ""),
                kinematic_fingerprint=_vendor_fingerprint(vendor_urdf),
                payload_present=bool(payload_present),
            )
    elif overlay and not xacro_binary:
        logger.warning("xacro is not installed; cannot expand the description overlay")

    try:
        vendor = vendor_urdf()
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return ResolvedDescription("", DescriptionSource.UNAVAILABLE, str(exc))
    return ResolvedDescription(
        vendor,
        DescriptionSource.VENDOR,
        "vendor description: none of the overlay's additions",
        kinematic_fingerprint=urdf_fingerprint(vendor),
    )


def _vendor_fingerprint(vendor_urdf: Callable[[], str]) -> str:
    """The vendor description an overlay includes, hashed; ``""`` if unresolvable."""
    try:
        return urdf_fingerprint(vendor_urdf())
    except Exception as exc:  # noqa: BLE001 - a missing vendor file is reported
        logger.warning("could not fingerprint the vendor description: %s", exc)
        return ""


# ── Firmware <-> description compatibility gate ─────────────────────────────


class CompatibilityVerdict(Enum):
    """What we actually know about this firmware/description pairing."""

    COMPATIBLE = "compatible"
    #: The arm reported no firmware at all (probe failed, or not run).
    NO_FIRMWARE = "no_firmware"
    #: Firmware reported, but absent from the compatibility table.
    UNKNOWN_FIRMWARE = "unknown_firmware"
    #: The loaded description is not one whose kinematics have been verified.
    UNKNOWN_URDF = "unknown_urdf"
    #: Both sides recognised, and this pairing is recorded as wrong.
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    """An immutable answer, carrying its own justification."""

    verdict: CompatibilityVerdict
    reason: str
    #: Whether Cartesian and planned motion may run.
    motion_allowed: bool
    #: Always True. Present so callers read intent rather than infer it from the
    #: absence of a check -- a future safety change must not accidentally cut
    #: telemetry off along with motion. Observing an arm you cannot verify is
    #: exactly how you come to verify it.
    telemetry_allowed: bool = True
    #: True when motion was permitted by explicit diagnostic override rather than
    #: by a clean verdict. Status surfaces must keep reporting this.
    overridden: bool = False

    @property
    def verified(self) -> bool:
        """Motion permitted *and* actually justified by the evidence."""
        return self.verdict is CompatibilityVerdict.COMPATIBLE


_OVERRIDE_NOTE = (
    " Motion permitted only by explicit diagnostic override; the arm is NOT "
    "verified against this description."
)


def _with_hint(reason: str, hint: str) -> str:
    return f"{reason} {hint}" if hint else reason


def evaluate_hardware_compatibility(
    firmware_version: str | None,
    urdf_fingerprint: str,
    *,
    known_fingerprints: Mapping[str, str],
    compatibility: Mapping[str, frozenset[str]],
    diagnostic_override: bool = False,
    no_firmware_hint: str = "",
    unknown_firmware_hint: str = "",
    mismatch_hint: str = "",
) -> CompatibilityDecision:
    """Decide whether this arm may be commanded through this description.

    * *firmware_version* is what the arm reports; ``None``/empty means it was
      never read.
    * *urdf_fingerprint* is :func:`urdf_fingerprint` (or a
      :class:`ResolvedDescription`'s kinematic fingerprint) of the file the
      solver will actually load -- the configured path is not good enough, two
      different files can sit at one path.
    * *known_fingerprints* maps fingerprints whose kinematics were verified to a
      human label; *compatibility* maps a firmware version to the fingerprints it
      is recorded to match. Both are driver-owned evidence: an entry re-enables
      planned motion, so it belongs next to the measurement that justified it.
    * The ``*_hint`` strings are appended to the matching reason so the operator
      is told this arm's remedy (which probe to run, which doc section to read).

    ``diagnostic_override`` unblocks motion without changing the verdict, so a
    commissioning session can move an unverified arm deliberately while every
    status surface keeps saying it is unverified.
    """
    if urdf_fingerprint not in known_fingerprints:
        verdict = CompatibilityVerdict.UNKNOWN_URDF
        reason = (
            f"URDF fingerprint {urdf_fingerprint!r} is not a verified description; "
            "its kinematics have never been checked against the vendor model."
        )
    elif not firmware_version:
        verdict = CompatibilityVerdict.NO_FIRMWARE
        reason = _with_hint(
            "Arm firmware was not read, so the loaded description cannot be "
            "confirmed to match the hardware.",
            no_firmware_hint,
        )
    elif firmware_version not in compatibility:
        verdict = CompatibilityVerdict.UNKNOWN_FIRMWARE
        reason = _with_hint(
            f"Firmware {firmware_version!r} has no recorded URDF compatibility.",
            unknown_firmware_hint,
        )
    elif urdf_fingerprint in compatibility[firmware_version]:
        verdict = CompatibilityVerdict.COMPATIBLE
        reason = (
            f"Firmware {firmware_version!r} is recorded as compatible with URDF "
            f"{urdf_fingerprint!r}."
        )
    else:
        verdict = CompatibilityVerdict.MISMATCH
        reason = _with_hint(
            f"Firmware {firmware_version!r} is recorded as compatible with "
            f"{sorted(compatibility[firmware_version])}, not with "
            f"{urdf_fingerprint!r}.",
            mismatch_hint,
        )

    clean = verdict is CompatibilityVerdict.COMPATIBLE
    return CompatibilityDecision(
        verdict=verdict,
        reason=reason if clean or not diagnostic_override else reason + _OVERRIDE_NOTE,
        motion_allowed=clean or diagnostic_override,
        telemetry_allowed=True,
        overridden=diagnostic_override and not clean,
    )
