"""The SDK can load the shared geometry core.

Not a geometry test -- ``cyberwave-geometry`` has 420 golden vectors and its own
C++ suite for that. This is a *packaging* test, and it exists because the SDK
deleted its own quaternion maths: ``twin.py``, ``placement.py``, ``schema.py``
and ``calibration/frames.py`` all call the core, each through the lazy
``cyberwave/_geometry.py`` shim.

The laziness is what makes the failure invisible. ``import cyberwave`` succeeds
without the core, so every test, example and image here looks fine, and the
first thing that notices is a user's call to ``Quaternion.from_rpy`` -- which is
exactly the person who cannot fix it. That happened for real: the dependency sat
commented out in ``pyproject.toml`` while CI installed the core from the
checkout, so nothing in this suite could tell the difference between "declared"
and "happens to be present".

So: fail here, in a named test, with the paths it looked for.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_the_shared_geometry_core_loads() -> None:
    try:
        import cyberwave_geometry
    except ImportError as error:  # pragma: no cover - only in a broken env
        pytest.fail(
            "The SDK cannot load the shared geometry core.\n"
            f"{error}\n\n"
            "Install it with:\n"
            "  pip install cyberwave-geometry\n\n"
            "It is a declared dependency, so this usually means a partial\n"
            "install, or a platform with no wheel (wheels cover CPython\n"
            "3.10-3.14 on Linux, macOS and Windows)."
        )

    assert cyberwave_geometry.core_version(), "the core reported no version"
    assert Path(cyberwave_geometry._native.library_path).exists(), (
        f"the binding reported loading {cyberwave_geometry._native.library_path}, "
        "which does not exist"
    )


def test_the_core_computes_rather_than_merely_importing() -> None:
    """A stub or a mismatched ABI imports fine and then returns nonsense."""
    from cyberwave_geometry import GeometryError, Quaternion, Vector3
    from cyberwave_geometry import quaternion as quat

    rotated = quat.rotate_unit(quat.from_rpy(0.0, 0.0, math.pi / 2), Vector3(1.0, 0.0, 0.0))
    assert rotated.x == pytest.approx(0.0, abs=1e-12)
    assert rotated.y == pytest.approx(1.0, abs=1e-12)

    # Strictness is the contract `cyberwave.compat`-style call sites rely on:
    # they can only choose a fallback because the core refuses first.
    with pytest.raises(GeometryError):
        quat.normalize(Quaternion(x=0.0, y=0.0, z=0.0, w=0.0))


def test_the_dependency_is_declared_not_merely_installed() -> None:
    """The core being importable here says nothing about a user's install.

    A commented-out dependency resolves perfectly in this repository, because
    CI installs the core from the checkout before the SDK. The only thing that
    makes `pip install cyberwave` work for someone else is this line.
    """
    declared = re.search(
        r"^cyberwave-geometry *= *(.+)$", PYPROJECT.read_text(), re.MULTILINE
    )
    assert declared, (
        "pyproject.toml does not declare cyberwave-geometry.\n"
        "Without it, a released SDK raises ImportError from every rotation\n"
        "entry point for anyone who installed it the documented way."
    )


def test_the_shim_reports_a_usable_instruction_when_the_core_is_absent() -> None:
    """The message a user actually sees is the whole value of the lazy shim.

    It named an unpublished package for as long as the core was unpublished;
    once that stopped being true the message became the wrong instruction, and
    nothing would have caught it.
    """
    from cyberwave import _geometry

    assert "pip install cyberwave-geometry" in _geometry._MISSING
    assert "NOT yet published" not in _geometry._MISSING
