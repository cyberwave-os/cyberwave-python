"""NotSimulatedError is a CyberwaveError subclass."""

from __future__ import annotations

from cyberwave.exceptions import CyberwaveError, NotSimulatedError


def test_not_simulated_error_is_cyberwave_error() -> None:
    err = NotSimulatedError("nope")
    assert isinstance(err, CyberwaveError)
    assert str(err) == "nope"
