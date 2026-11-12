"""Description fingerprinting, runtime description resolution, and the
firmware/description compatibility gate.

Vendors ship description variants that differ by about a centimetre at the
flange, and nothing in a running system compares the loaded URDF to the arm that
will execute the pose. Until a driver records a firmware -> description pairing,
the honest state is "unknown", and unknown must inhibit Cartesian and planned
motion rather than warn about it. All tables here are synthetic: real ones, and
the evidence behind them, belong to the driver.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cyberwave.driver.kinematics.arm.description import (
    UNKNOWN_FINGERPRINT,
    CompatibilityVerdict,
    DescriptionSource,
    ResolvedDescription,
    evaluate_hardware_compatibility,
    resolve_runtime_description,
    urdf_fingerprint,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_URDF = _FIXTURES / "simple_arm.urdf"

_KNOWN = "aaaaaaaaaaaa"
_OTHER_KNOWN = "bbbbbbbbbbbb"
KNOWN_FINGERPRINTS = {_KNOWN: "variant A (measured)", _OTHER_KNOWN: "variant B (old)"}
COMPATIBILITY = {"FW-1.0": frozenset({_KNOWN})}


def _evaluate(firmware: str | None, fingerprint: str, **kwargs):
    kwargs.setdefault("known_fingerprints", KNOWN_FINGERPRINTS)
    kwargs.setdefault("compatibility", COMPATIBILITY)
    return evaluate_hardware_compatibility(firmware, fingerprint, **kwargs)


# ── fingerprinting ──────────────────────────────────────────────────────────


def test_fingerprint_is_stable_and_short() -> None:
    fingerprint = urdf_fingerprint(str(_URDF))
    assert fingerprint == urdf_fingerprint(str(_URDF))
    assert len(fingerprint) == 12 and fingerprint.isalnum()


def test_fingerprint_is_content_not_path(tmp_path: Path) -> None:
    """Same file at two mount points is the same kinematics; an edit is not."""
    copy = tmp_path / "copy.urdf"
    copy.write_bytes(_URDF.read_bytes())
    assert urdf_fingerprint(str(copy)) == urdf_fingerprint(str(_URDF))
    copy.write_bytes(_URDF.read_bytes() + b"\n")
    assert urdf_fingerprint(str(copy)) != urdf_fingerprint(str(_URDF))


def test_missing_file_fingerprints_as_unknown() -> None:
    assert urdf_fingerprint("/nope/does/not/exist.urdf") == UNKNOWN_FINGERPRINT


def test_empty_path_fingerprints_as_unknown() -> None:
    """Manifest export passes "" before the URDF is resolvable; must not raise."""
    assert urdf_fingerprint("") == UNKNOWN_FINGERPRINT


# ── the compatibility gate ──────────────────────────────────────────────────


def test_a_recorded_compatible_pair_permits_motion() -> None:
    decision = _evaluate("FW-1.0", _KNOWN)
    assert decision.verdict is CompatibilityVerdict.COMPATIBLE
    assert decision.motion_allowed is True
    assert decision.verified is True
    assert decision.overridden is False


def test_a_recorded_firmware_does_not_vouch_for_another_known_variant() -> None:
    """Firmware known, URDF known, pairing wrong -- the case the gate exists for."""
    decision = _evaluate("FW-1.0", _OTHER_KNOWN)
    assert decision.verdict is CompatibilityVerdict.MISMATCH
    assert decision.motion_allowed is False
    assert _OTHER_KNOWN in decision.reason


def test_unknown_firmware_inhibits_motion() -> None:
    decision = _evaluate("FW-9.9", _KNOWN)
    assert decision.verdict is CompatibilityVerdict.UNKNOWN_FIRMWARE
    assert decision.motion_allowed is False
    assert "FW-9.9" in decision.reason


@pytest.mark.parametrize("firmware", [None, ""])
def test_absent_firmware_inhibits_motion(firmware: str | None) -> None:
    decision = _evaluate(firmware, _KNOWN)
    assert decision.verdict is CompatibilityVerdict.NO_FIRMWARE
    assert decision.motion_allowed is False


def test_unrecognised_urdf_inhibits_motion_before_firmware_is_considered() -> None:
    """An unverified description is unknown whatever the firmware says."""
    for firmware in ("FW-1.0", None):
        decision = _evaluate(firmware, "deadbeef1234")
        assert decision.verdict is CompatibilityVerdict.UNKNOWN_URDF
        assert decision.motion_allowed is False


def test_an_empty_table_fails_closed() -> None:
    decision = _evaluate("FW-1.0", _KNOWN, compatibility={})
    assert decision.verdict is CompatibilityVerdict.UNKNOWN_FIRMWARE
    assert decision.motion_allowed is False


def test_hints_carry_the_drivers_remedy_into_the_reason() -> None:
    assert "run probe" in _evaluate(None, _KNOWN, no_firmware_hint="run probe").reason
    assert (
        "see docs"
        in _evaluate("FW-9.9", _KNOWN, unknown_firmware_hint="see docs").reason
    )
    assert (
        "differ by 1 cm"
        in _evaluate("FW-1.0", _OTHER_KNOWN, mismatch_hint="differ by 1 cm").reason
    )


@pytest.mark.parametrize(
    "firmware,fingerprint",
    [(None, _KNOWN), ("FW-1.0", _KNOWN), ("FW-1.0", "deadbeef1234")],
)
def test_telemetry_stays_available_whatever_the_verdict(
    firmware: str | None, fingerprint: str
) -> None:
    """The gate inhibits motion only. Read-only observation is never blocked."""
    assert _evaluate(firmware, fingerprint).telemetry_allowed is True


def test_the_diagnostic_override_is_never_the_default() -> None:
    default = _evaluate(None, _KNOWN)
    assert default.motion_allowed is False
    assert default.overridden is False


def test_the_diagnostic_override_is_explicit_and_stays_loud() -> None:
    """It may unblock motion, but it must never look like a clean verdict."""
    decision = _evaluate(None, _KNOWN, diagnostic_override=True)
    assert decision.motion_allowed is True
    assert decision.overridden is True
    assert decision.verified is False
    assert decision.verdict is CompatibilityVerdict.NO_FIRMWARE
    assert "override" in decision.reason.lower()


def test_override_does_not_invent_compatibility_for_a_known_mismatch() -> None:
    decision = _evaluate("FW-1.0", _OTHER_KNOWN, diagnostic_override=True)
    assert decision.verdict is CompatibilityVerdict.MISMATCH
    assert decision.overridden is True


def test_override_on_a_clean_verdict_is_not_reported_as_an_override() -> None:
    decision = _evaluate("FW-1.0", _KNOWN, diagnostic_override=True)
    assert decision.verified is True
    assert decision.overridden is False
    assert "override" not in decision.reason.lower()


def test_decision_is_immutable() -> None:
    decision = _evaluate(None, _KNOWN)
    with pytest.raises((AttributeError, TypeError)):
        decision.motion_allowed = True  # type: ignore[misc]


# ── runtime description resolution ──────────────────────────────────────────


def _vendor(tmp_path: Path) -> str:
    vendor = tmp_path / "vendor" / "urdf" / "vendor.urdf"
    vendor.parent.mkdir(parents=True)
    vendor.write_bytes(_URDF.read_bytes())
    return str(vendor)


def _fake_xacro(bin_dir: Path, *, body: str, exit_code: int = 0) -> None:
    """A stand-in ``xacro`` on PATH: prints *body* and records its argv."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "xacro"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" > "{bin_dir}/argv"\n'
        f"cat <<'URDF'\n{body}\nURDF\n"
        f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def test_an_explicit_path_always_wins(tmp_path: Path) -> None:
    def _never() -> str:
        raise AssertionError("vendor must not be consulted")

    resolved = resolve_runtime_description(
        urdf_path_override=str(_URDF), vendor_urdf=_never
    )
    assert resolved.source is DescriptionSource.CONFIGURED
    assert resolved.path == str(_URDF)
    assert resolved.kinematic_fingerprint == urdf_fingerprint(str(_URDF))


def test_without_an_overlay_the_vendor_description_is_loaded(tmp_path: Path) -> None:
    vendor = _vendor(tmp_path)
    resolved = resolve_runtime_description(
        urdf_path_override=None, vendor_urdf=lambda: vendor
    )
    assert resolved.source is DescriptionSource.VENDOR
    assert resolved.path == vendor
    assert resolved.includes_payload_geometry is False
    assert resolved.effective_fingerprint() == urdf_fingerprint(vendor)


def test_an_unresolvable_description_is_reported_never_raised() -> None:
    def _missing() -> str:
        raise FileNotFoundError("vendor package not installed")

    resolved = resolve_runtime_description(urdf_path_override="", vendor_urdf=_missing)
    assert resolved.source is DescriptionSource.UNAVAILABLE
    assert resolved.path == ""
    assert "not installed" in resolved.detail


def test_the_overlay_is_expanded_and_keeps_the_vendor_kinematic_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expanded file's hash changes with configuration; the gate must not."""
    vendor = _vendor(tmp_path)
    overlay = tmp_path / "overlay.urdf.xacro"
    overlay.write_text("<robot/>")
    bin_dir = tmp_path / "bin"
    _fake_xacro(bin_dir, body="<robot name='expanded'/>")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    resolved = resolve_runtime_description(
        urdf_path_override=None,
        vendor_urdf=lambda: vendor,
        overlay_xacro=str(overlay),
        xacro_args={"tool_offset_z": "0.12", "payload_present": "true"},
        payload_present=True,
        cache_dir=str(tmp_path / "cache"),
        output_name="expanded.urdf",
    )

    assert resolved.source is DescriptionSource.OVERLAY
    assert resolved.path == str(tmp_path / "cache" / "expanded.urdf")
    assert "expanded" in Path(resolved.path).read_text()
    assert resolved.kinematic_fingerprint == urdf_fingerprint(vendor)
    assert resolved.kinematic_fingerprint != urdf_fingerprint(resolved.path)
    assert resolved.effective_fingerprint() == urdf_fingerprint(vendor)
    assert resolved.includes_payload_geometry is True
    assert "tool_offset_z:=0.12" in (bin_dir / "argv").read_text()


def test_a_failed_expansion_falls_back_to_the_vendor_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor = _vendor(tmp_path)
    overlay = tmp_path / "overlay.urdf.xacro"
    overlay.write_text("<robot/>")
    bin_dir = tmp_path / "bin"
    _fake_xacro(bin_dir, body="", exit_code=1)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    resolved = resolve_runtime_description(
        urdf_path_override=None,
        vendor_urdf=lambda: vendor,
        overlay_xacro=str(overlay),
        cache_dir=str(tmp_path / "cache"),
    )
    assert resolved.source is DescriptionSource.VENDOR
    assert resolved.path == vendor


def test_an_overlay_without_xacro_falls_back_to_the_vendor_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor = _vendor(tmp_path)
    overlay = tmp_path / "overlay.urdf.xacro"
    overlay.write_text("<robot/>")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    resolved = resolve_runtime_description(
        urdf_path_override=None, vendor_urdf=lambda: vendor, overlay_xacro=str(overlay)
    )
    assert resolved.source is DescriptionSource.VENDOR


def test_payload_geometry_is_only_claimed_for_an_overlay() -> None:
    vendor = ResolvedDescription("x", DescriptionSource.VENDOR, payload_present=True)
    assert vendor.includes_payload_geometry is False
