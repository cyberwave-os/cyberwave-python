"""Cross-language wire-format contract for ``edge_health``.

Asserts that the Python SDK's ``EdgeHealthCheck`` produces a payload
whose structural shape matches the JSON fixtures under
``tests/fixtures/edge_health/``.  The C++ SDK's wire-format test in
CYB-2004 PR 2 reads fixtures with identical content from its own
``tests/`` (via the canonical monorepo path while in-tree, vendored
copy when published), giving us a deliberately-blunt cross-language
guarantee that the two publishers cannot drift on field names,
casing, or types.

The fixtures live inside this SDK's ``tests/`` rather than at a
shared monorepo root so they ship with the package when it is
published to ``cyberwave-os/cyberwave-python``.  Reaching above
``cyberwave-python/`` for any asset breaks the standalone publish
(see the run that exposed it: cyberwave-os/cyberwave-python actions
run 26058310626).

The fixture is intentionally minimal: it omits all timing /
publisher-runtime fields (``timestamp``, ``uptime_seconds``,
``frames_sent``, ``fps``, ``is_stale``, ...) so the assertion can be
made structurally without timing dependencies.  Fields that survive
the redaction are the ones the dashboard renders to the operator —
exactly the ones we cannot let drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from cyberwave.edge.health import EdgeHealthCheck


# Fixtures live in-tree so they ship with the SDK when it is mirrored
# to the standalone open-source repo.  A missing file here is a real
# failure, not a "skip silently in OSS CI" situation — that defeats
# the entire point of pinning the wire format.
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "edge_health"


# Fields that vary with publisher runtime state and therefore cannot be
# captured in a static fixture without making the test flaky.  We strip
# them from both sides before structural equality.  Adding a new
# always-changing field here is fine; removing one risks regressing
# behaviour, so consider whether the change is really publisher-runtime
# rather than a contract change.
_RUNTIME_FIELDS_TOP_LEVEL = frozenset({"timestamp", "uptime_seconds"})
_RUNTIME_FIELDS_PER_STREAM = frozenset(
    {
        "connection_state",
        "ice_connection_state",
        "frames_sent",
        "last_frame_ts",
        "fps",
        "uptime_seconds",
        "restart_count",
        "is_stale",
        "is_healthy",
    }
)


def _strip_runtime_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove publisher-runtime fields so the result is fixture-comparable."""
    cleaned = {k: v for k, v in payload.items() if k not in _RUNTIME_FIELDS_TOP_LEVEL}
    streams = cleaned.get("streams")
    if isinstance(streams, dict):
        cleaned["streams"] = {
            sid: {k: v for k, v in entry.items() if k not in _RUNTIME_FIELDS_PER_STREAM}
            if isinstance(entry, dict)
            else entry
            for sid, entry in streams.items()
        }
    return cleaned


class _FakeMQTT:
    topic_prefix = ""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


def _synthesise_payload(checker: EdgeHealthCheck, *, twin_uuid: str) -> Dict[str, Any]:
    """Build one wire payload, mirroring the publisher's compose path."""
    health_data = checker.get_health_data()
    return {
        "type": "edge_health",
        "timestamp": 1700000000.0,
        "edge_id": checker.edge_id,
        "twin_uuid": twin_uuid,
        "uptime_seconds": 0.0,
        **health_data,
    }


@pytest.mark.parametrize(
    "fixture_name, registrations",
    [
        (
            "camera_minimal.json",
            [
                (
                    "stream",
                    {
                        "kind": "camera",
                        "source": "/dev/video0",
                        "resolution": "1280x720",
                        "fps": 15,
                        "camera_type": "cv2",
                    },
                ),
            ],
        ),
        (
            "multi_stream_realsense.json",
            [
                (
                    "rgb-0",
                    {
                        "kind": "camera",
                        "source": "0",
                        "resolution": "1280x720",
                        "fps": 30,
                        "camera_type": "realsense",
                    },
                ),
                (
                    "depth-0",
                    {
                        "kind": "camera",
                        "source": "0",
                        "resolution": "640x480",
                        "fps": 30,
                        "camera_type": "realsense",
                    },
                ),
            ],
        ),
    ],
)
def test_python_publisher_matches_shared_fixture(
    fixture_name: str, registrations: list
) -> None:
    """The Python SDK's output equals the shared fixture, modulo runtime fields.

    A regression here means the Python publisher diverged from the
    contract the C++ publisher is also expected to honour.  Bring them
    back into agreement by either updating both publishers or updating
    the fixture (and the matching C++ test) — never just one.

    The multi-stream fixture additionally pins the deterministic
    ``camera_config`` shim winner (lexicographic sort on ``stream_id``)
    so multi-camera devices don't appear to flap identities on each
    heartbeat.
    """
    fixture_path = _FIXTURES_DIR / fixture_name
    fixture_payload = json.loads(fixture_path.read_text())

    checker = EdgeHealthCheck(
        mqtt_client=_FakeMQTT(),
        twin_uuids=["twin-fixture"],
        edge_id="twin-fixture",
    )
    for stream_id, config in registrations:
        checker.register_stream_config(stream_id, config)

    synthesised = _synthesise_payload(checker, twin_uuid="twin-fixture")

    assert _strip_runtime_fields(synthesised) == _strip_runtime_fields(fixture_payload)
