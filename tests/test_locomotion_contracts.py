import json
import math
from pathlib import Path

import pytest

import cyberwave
from cyberwave.locomotion_contracts import (
    LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
    LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS,
    build_locomotion_velocity_command,
    stop_locomotion_velocity_command,
)


def test_locomotion_contract_helpers_are_exported_from_package() -> None:
    assert (
        cyberwave.LOCOMOTION_VELOCITY_COMMAND_CONTRACT
        == LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    )
    assert (
        cyberwave.LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS
        == LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS
    )
    assert (
        cyberwave.stop_locomotion_velocity_command()
        == stop_locomotion_velocity_command()
    )


#: Where the schemas live in a monorepo checkout. They moved out of
#: ``cyberwave-backend/src/app/contracts`` into the shared package; this test went
#: on resolving the old path, found no directory, and skipped -- passing green for
#: every one of the drift cases it exists to catch.
_SCHEMAS_IN_CHECKOUT = "cyberwave-contracts/cyberwave_contracts/schemas"


def _schemas_dir() -> Path | None:
    """The shipped schema directory, or ``None`` when neither source is present.

    Two sources, because this file runs in two places. In the monorepo the
    checkout is authoritative, and in a standalone SDK environment the installed
    package is the only copy. ``None`` means genuinely neither -- a standalone
    environment that also has no contracts package -- which is the one context
    where there is nothing to compare against.
    """
    try:
        from cyberwave_contracts.manifest import SCHEMAS_DIR
    except ImportError:
        pass
    else:
        return SCHEMAS_DIR

    for parent in Path(__file__).resolve().parents:
        candidate = parent / _SCHEMAS_IN_CHECKOUT
        if candidate.is_dir():
            return candidate
    return None


def _schema(contract_id: str) -> dict:
    """Load one contract schema, failing rather than skipping when it should exist.

    The distinction is the whole point: absence of the *directory* means this is a
    standalone environment and there is nothing to check, but absence of a
    *schema* inside a directory that exists is drift, and skipping on it would
    pass in exactly the case worth catching.
    """
    directory = _schemas_dir()
    if directory is None:
        pytest.skip(
            "no cyberwave_contracts package and no monorepo checkout -- the "
            "standalone SDK environment, the one context with nothing to compare "
            f"against. In a checkout the schemas are at {_SCHEMAS_IN_CHECKOUT}."
        )
    path = directory / f"{contract_id}.schema.json"
    if not path.is_file():
        pytest.fail(
            f"{path} is missing, but {directory} exists. The contract this SDK "
            "copy claims to implement is gone or renamed; skipping here would "
            "report that as success."
        )
    return json.loads(path.read_text())


def test_build_locomotion_velocity_command_matches_repo_schema_required_fields() -> None:
    schema = _schema(LOCOMOTION_VELOCITY_COMMAND_CONTRACT)
    payload = build_locomotion_velocity_command(
        linear_x=0.2,
        angular_z=0.1,
        duration_ms=500,
        gait="walk",
        origin="teleop",
    ).to_payload()

    assert schema["properties"]["contract"]["const"] == (
        LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    )
    assert tuple(schema["required"]) == LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS
    assert schema["x-cyberwave-adapter-capabilities"]["stop"] == (
        stop_locomotion_velocity_command()
    )
    assert payload["contract"] == LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    assert set(LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS).issubset(payload)


def test_build_locomotion_velocity_command_allows_schema_stop_duration() -> None:
    command = build_locomotion_velocity_command(
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        duration_ms=0,
        gait="stand",
        origin="workflow",
    )

    assert command.to_payload() == {
        "linear_x": 0.0,
        "linear_y": 0.0,
        "angular_z": 0.0,
        "duration_ms": 0,
        "gait": "stand",
        "origin": "workflow",
        "contract": LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
    }


def test_build_locomotion_velocity_command_rejects_short_active_duration() -> None:
    with pytest.raises(ValueError, match="0 or at least 50"):
        build_locomotion_velocity_command(duration_ms=1)


def test_build_locomotion_velocity_command_rejects_fractional_duration() -> None:
    with pytest.raises(ValueError, match="integer duration_ms"):
        build_locomotion_velocity_command(duration_ms=500.5)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    (
        ("linear_x", {"linear_x": math.nan}),
        ("angular_z", {"angular_z": math.inf}),
        ("duration_ms", {"duration_ms": math.inf}),
    ),
)
def test_build_locomotion_velocity_command_rejects_non_finite_numbers(
    field_name: str,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=f"finite numeric {field_name}"):
        build_locomotion_velocity_command(**kwargs)


def test_stop_locomotion_velocity_command_uses_canonical_contract() -> None:
    payload = stop_locomotion_velocity_command({"origin": "workflow"})

    assert payload["contract"] == LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    assert payload["duration_ms"] == 0
    assert payload["gait"] == "stand"
    assert payload["origin"] == "workflow"
