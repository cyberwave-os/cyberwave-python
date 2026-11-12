"""``publish_frame_poses`` against the frame_poses contract.

The golden fixtures in ``common/protobuf/fixtures/mqtt/twin/v1/`` are the
contract's payloads, so the SDK has to put the same topic and fields on the
wire, and refuse what Vector would drop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest

from cyberwave.driver.interface.args import TopicSpec
from cyberwave.driver.interface.registry import resolve_topic_path
from cyberwave.manifest.driver_config import TWIN_FRAME_POSES_TOPIC_SLUG
from cyberwave.mqtt import POSE_PROVENANCES, CyberwaveMQTTClient
from cyberwave.mqtt_client import CyberwaveMQTTClient as WrapperMQTTClient

_FIXTURES = (
    Path(__file__).resolve().parents[3] / "common/protobuf/fixtures/mqtt/twin/v1"
)
_SIM_FIXTURE = _FIXTURES / "frame_poses.json"
_HARDWARE_FIXTURE = _FIXTURES / "frame_poses_hardware.json"
_needs_fixtures = pytest.mark.skipif(
    not _SIM_FIXTURE.is_file(),
    reason="the frame_poses contract lives in the monorepo's common/protobuf",
)

TWIN = "8f1c2d3e-0000-4000-8000-000000000001"
POSITION = {"x": 1.2, "y": 0.5, "z": 0.31}
ORIENTATION = {"x": 0.0, "y": 0.0, "z": 0.04998, "w": 0.99875}
FRAME = {
    "child_frame_id": "base_link",
    "position": POSITION,
    "orientation": ORIENTATION,
}
WIRE_FRAME = {
    "child_frame_id": "base_link",
    "pose": {"position": POSITION, "orientation": ORIENTATION},
    "pose_provenance": "POSE_PROVENANCE_GROUND_TRUTH",
}
COVARIANCE = [0.0] * 36


@pytest.fixture
def paho() -> Any:
    with patch("cyberwave.mqtt.mqtt.Client") as client_cls:
        paho = client_cls.return_value
        paho.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
        yield paho


def _client(source_type: str | None = "sim", **kwargs: Any) -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_username="user",
        api_key="api_key_secret",
        auto_connect=False,
        source_type=source_type,
        **kwargs,
    )
    client.connected = True
    return client


@pytest.fixture
def client(paho: MagicMock) -> CyberwaveMQTTClient:
    return _client()


@pytest.fixture
def hardware(paho: MagicMock) -> CyberwaveMQTTClient:
    return _client("edge")


def _sent(paho: MagicMock) -> tuple[str, dict[str, Any]]:
    (call,) = paho.publish.call_args_list
    return call.args[0], json.loads(call.args[1])


def test_publishes_the_frames_on_the_twin_topic(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    before = time.time_ns()
    assert client.publish_frame_poses(TWIN, [FRAME]) is True
    after = time.time_ns()

    topic, message = _sent(paho)
    assert topic == f"cyberwave/twin/{TWIN}/frame_poses"
    assert topic == TWIN_FRAME_POSES_TOPIC_SLUG.format(twin_uuid=TWIN)
    header = message["header"]
    assert header["frame_id"] == "environment"
    assert header["twin_uuid"] == TWIN
    assert header["substrate"] == "SUBSTRATE_SIM"
    stamp = header["stamp"]
    assert stamp["clock"] == "CLOCK_DOMAIN_REALTIME"
    assert isinstance(stamp["value_nanos"], str)
    assert before <= int(stamp["value_nanos"]) <= after
    assert stamp["clock_id"] == client.client_id
    assert message["producer_instance"] == client.client_id
    assert message["frames"] == [WIRE_FRAME]
    assert "source_type" not in message
    assert "timestamp" not in message


def test_the_topic_carries_the_client_prefix(paho: MagicMock) -> None:
    client = _client(topic_prefix="dev")
    client.publish_frame_poses(TWIN, [FRAME])
    topic, _ = _sent(paho)
    assert topic == f"devcyberwave/twin/{TWIN}/frame_poses"


def test_accepts_the_previous_frame_spelling(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    frame = {
        "frame_name": "base_link",
        "position": POSITION,
        "quaternion": {"w": 0.99875, "x": 0.0, "y": 0.0, "z": 0.04998},
    }
    client.publish_frame_poses(TWIN, [frame])
    _, message = _sent(paho)
    assert message["frames"] == [WIRE_FRAME]


def test_sends_the_envelope_it_is_given(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    client.publish_frame_poses(
        TWIN,
        [FRAME],
        source_subtype="mujoco",
        producer_instance="sim-7f3a",
        producer_sequence=77190,
        workload_uuid="9b7e4f10-0000-4000-8000-0000000000aa",
        simulation_time_s=12.5,
    )
    _, message = _sent(paho)
    assert message["header"]["stamp"]["clock_id"] == "sim-7f3a"
    assert message["producer_instance"] == "sim-7f3a"
    assert message["source_subtype"] == "mujoco"
    assert message["producer_sequence"] == 77190
    assert message["workload_uuid"] == "9b7e4f10-0000-4000-8000-0000000000aa"
    assert message["simulation_time_s"] == 12.5


@_needs_fixtures
@pytest.mark.parametrize(
    ("fixture_path", "source_type"),
    [(_SIM_FIXTURE, "sim"), (_HARDWARE_FIXTURE, "edge")],
    ids=["sim", "hardware"],
)
def test_reproduces_the_contract_fixture(
    paho: MagicMock, fixture_path: Path, source_type: str
) -> None:
    fixture = json.loads(fixture_path.read_text())
    payload = fixture["payload"]
    twin = fixture["topic"].split("/")[2]
    client = _client(source_type)
    envelope = {
        key: payload[key]
        for key in (
            "source_subtype",
            "producer_instance",
            "workload_uuid",
            "simulation_time_s",
        )
        if key in payload
    }
    if "producer_sequence" in payload:
        envelope["producer_sequence"] = int(payload["producer_sequence"])
    client.publish_frame_poses(twin, payload["frames"], **envelope)

    topic, message = _sent(paho)
    assert topic == fixture["topic"]
    # session_id is the client's, stamped on every publish.
    assert set(message) - {"session_id"} <= set(payload)
    assert set(message["header"]) - {"twin_uuid"} == set(payload["header"]) - {
        "twin_uuid"
    }
    assert message["header"]["substrate"] == payload["header"]["substrate"]
    assert message["header"]["frame_id"] == payload["header"]["frame_id"]
    assert set(message["header"]["stamp"]) == set(payload["header"]["stamp"])
    assert message["frames"] == payload["frames"]


@pytest.mark.parametrize(
    ("given", "sent"),
    [
        ("odometry", "POSE_PROVENANCE_ODOMETRY"),
        ("POSE_PROVENANCE_SLAM", "POSE_PROVENANCE_SLAM"),
        ("lidar_localization", "POSE_PROVENANCE_LIDAR_LOCALIZATION"),
    ],
)
def test_maps_a_friendly_provenance_to_its_full_name(
    hardware: CyberwaveMQTTClient, paho: MagicMock, given: str, sent: str
) -> None:
    hardware.publish_frame_poses(TWIN, [FRAME], pose_provenance=given)
    _, message = _sent(paho)
    assert message["header"]["substrate"] == "SUBSTRATE_HARDWARE"
    assert message["frames"][0]["pose_provenance"] == sent


def test_every_provenance_has_a_friendly_word(
    hardware: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    estimates = [p for p in POSE_PROVENANCES if p != "POSE_PROVENANCE_GROUND_TRUTH"]
    frames = [
        {**FRAME, "pose_provenance": name[len("POSE_PROVENANCE_") :].lower()}
        for name in estimates
    ]
    hardware.publish_frame_poses(TWIN, frames)
    _, message = _sent(paho)
    assert [f["pose_provenance"] for f in message["frames"]] == estimates


def test_one_frame_under_two_provenances_with_covariance(
    hardware: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    hardware.publish_frame_poses(
        TWIN,
        [
            {**FRAME, "pose_covariance": COVARIANCE},
            {**FRAME, "pose_provenance": "slam", "pose_covariance": None},
        ],
        pose_provenance="odometry",
    )
    _, message = _sent(paho)
    odometry, slam = message["frames"]
    assert odometry["pose_provenance"] == "POSE_PROVENANCE_ODOMETRY"
    assert odometry["pose_covariance"] == {"data": COVARIANCE}
    assert slam["pose_provenance"] == "POSE_PROVENANCE_SLAM"
    assert "pose_covariance" not in slam


@pytest.mark.parametrize(
    ("source_type", "substrate", "sent"),
    [
        ("sim_tele", None, "SUBSTRATE_SIM"),
        ("tele", None, "SUBSTRATE_HARDWARE"),
        (None, None, "SUBSTRATE_HARDWARE"),
        ("edge", "sim", "SUBSTRATE_SIM"),
        ("sim", "hardware", "SUBSTRATE_HARDWARE"),
        ("edit", "SUBSTRATE_SIM", "SUBSTRATE_SIM"),
    ],
)
def test_substrate_defaults_from_the_source_type(
    paho: MagicMock, source_type: str | None, substrate: str | None, sent: str
) -> None:
    client = _client(source_type)
    client.publish_frame_poses(
        TWIN, [FRAME], substrate=substrate, pose_provenance="odometry"
    )
    _, message = _sent(paho)
    assert message["header"]["substrate"] == sent


def test_an_edit_client_has_to_name_the_substrate(paho: MagicMock) -> None:
    with pytest.raises(ValueError, match="substrate"):
        _client("edit").publish_frame_poses(TWIN, [FRAME])
    paho.publish.assert_not_called()


def test_refuses_a_non_environment_frame(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    with pytest.raises(ValueError, match="environment"):
        client.publish_frame_poses(TWIN, [FRAME], frame_id="odom")
    paho.publish.assert_not_called()


@pytest.mark.parametrize("twin_uuid", ["not-a-uuid", "", f"{TWIN}/extra"])
def test_refuses_a_twin_id_that_is_not_a_uuid(
    client: CyberwaveMQTTClient, paho: MagicMock, twin_uuid: str
) -> None:
    # Vector drops a topic whose twin segment is not a UUID.
    with pytest.raises(ValueError, match="twin_uuid must be a UUID"):
        client.publish_frame_poses(twin_uuid, [FRAME])
    paho.publish.assert_not_called()


def test_refuses_no_frames(client: CyberwaveMQTTClient, paho: MagicMock) -> None:
    with pytest.raises(ValueError, match="at least one"):
        client.publish_frame_poses(TWIN, [])
    paho.publish.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"substrate": "scene"}, "substrate"),
        ({"substrate": "SIM"}, "substrate"),
        ({"pose_provenance": "laser"}, "pose_provenance"),
        ({"pose_provenance": "unspecified"}, "pose_provenance"),
        ({"pose_provenance": "POSE_PROVENANCE_UNSPECIFIED"}, "pose_provenance"),
        ({"producer_sequence": 1.5}, "producer_sequence"),
        ({"producer_sequence": True}, "producer_sequence"),
        ({"producer_sequence": 2**63}, "producer_sequence"),
        ({"producer_sequence": -(2**63) - 1}, "producer_sequence"),
        ({"producer_sequence": str(2**63)}, "producer_sequence"),
        ({"producer_sequence": "12a"}, "producer_sequence"),
        ({"producer_sequence": "1.5"}, "producer_sequence"),
        ({"producer_sequence": " 7"}, "producer_sequence"),
        ({"producer_sequence": "009"}, "producer_sequence"),
        ({"producer_sequence": "007"}, "producer_sequence"),
        ({"producer_sequence": ""}, "producer_sequence"),
        ({"source_subtype": 5}, "source_subtype"),
        ({"workload_uuid": 5}, "workload_uuid"),
        ({"producer_instance": 5}, "producer_instance"),
        ({"producer_instance": b"sim-7f3a"}, "producer_instance"),
        ({"simulation_time_s": -1.0}, "simulation_time_s"),
        ({"simulation_time_s": float("nan")}, "simulation_time_s"),
        ({"simulation_time_s": "1.5"}, "simulation_time_s"),
    ],
)
def test_refuses_arguments_vector_would_drop(
    client: CyberwaveMQTTClient, paho: MagicMock, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        client.publish_frame_poses(TWIN, [FRAME], **kwargs)
    paho.publish.assert_not_called()


@pytest.mark.parametrize(
    "sequence", [2**63 - 1, -(2**63), 0, "9223372036854775807", "-42", "0"]
)
def test_sends_a_producer_sequence_vector_accepts_as_given(
    client: CyberwaveMQTTClient, paho: MagicMock, sequence: int | str
) -> None:
    client.publish_frame_poses(TWIN, [FRAME], producer_sequence=sequence)

    _topic, message = _sent(paho)
    assert message["producer_sequence"] == sequence


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        ({**FRAME, "child_frame_id": ""}, "child_frame_id"),
        ({k: v for k, v in FRAME.items() if k != "child_frame_id"}, "child_frame_id"),
        ({**FRAME, "position": {"x": 1.0, "y": 2.0}}, r"position\.z"),
        ({**FRAME, "position": [1.0, 2.0, 3.0]}, "position"),
        ({**FRAME, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0}}, r"orientation\.w"),
        (
            {k: v for k, v in FRAME.items() if k != "orientation"},
            "orientation",
        ),
        (
            {**FRAME, "position": {"x": float("nan"), "y": 0.0, "z": 0.0}},
            r"position\.x",
        ),
        (
            {**FRAME, "position": {"x": float("inf"), "y": 0.0, "z": 0.0}},
            r"position\.x",
        ),
        ({**FRAME, "position": {"x": "1.0", "y": 0.0, "z": 0.0}}, r"position\.x"),
        ({**FRAME, "position": {"x": True, "y": 0.0, "z": 0.0}}, r"position\.x"),
        ({**FRAME, "pose": "identity"}, "pose"),
        ({**FRAME, "pose_provenance": "ground truth"}, "pose_provenance"),
        ({**FRAME, "pose_provenance": 1}, "pose_provenance"),
        ({**FRAME, "pose_covariance": COVARIANCE[:35]}, "pose_covariance"),
        ({**FRAME, "pose_covariance": {"data": [*COVARIANCE, 0.0]}}, "covariance"),
        ({**FRAME, "pose_covariance": {}}, "pose_covariance"),
        (
            {**FRAME, "pose_covariance": [float("nan"), *COVARIANCE[1:]]},
            r"pose_covariance\[0\]",
        ),
        ({**FRAME, "pose_covariance": ["0.1", *COVARIANCE[1:]]}, "pose_covariance"),
        ("base_link", "dict"),
    ],
)
def test_refuses_a_frame_vector_would_drop(
    client: CyberwaveMQTTClient, paho: MagicMock, frame: Any, match: str
) -> None:
    second = {**FRAME, "child_frame_id": "tool0"}
    with pytest.raises(ValueError, match=match):
        client.publish_frame_poses(TWIN, [second, frame])
    paho.publish.assert_not_called()


def test_refuses_a_frame_twice_under_one_provenance(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    with pytest.raises(ValueError, match="once per provenance"):
        client.publish_frame_poses(TWIN, [FRAME, FRAME])
    paho.publish.assert_not_called()


def test_hardware_has_to_say_how_a_pose_was_estimated(
    hardware: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    with pytest.raises(ValueError, match="no pose_provenance"):
        hardware.publish_frame_poses(TWIN, [FRAME])
    paho.publish.assert_not_called()


@pytest.mark.parametrize(
    ("frames", "kwargs"),
    [
        ([FRAME], {"pose_provenance": "ground_truth"}),
        ([{**FRAME, "pose_provenance": "POSE_PROVENANCE_GROUND_TRUTH"}], {}),
        (
            [FRAME, {**FRAME, "pose_provenance": "ground_truth"}],
            {"pose_provenance": "odometry"},
        ),
    ],
)
def test_hardware_cannot_claim_ground_truth(
    hardware: CyberwaveMQTTClient,
    paho: MagicMock,
    frames: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="GROUND_TRUTH"):
        hardware.publish_frame_poses(TWIN, frames, **kwargs)
    paho.publish.assert_not_called()


def test_sends_only_the_contract_keys_as_floats(
    client: CyberwaveMQTTClient, paho: MagicMock
) -> None:
    frame = {
        "child_frame_id": "tool0",
        "position": {"x": 1, "y": 2, "z": 3, "extra": 9},
        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
        "velocity": {"x": 1.0},
    }
    client.publish_frame_poses(TWIN, [frame])
    _, message = _sent(paho)
    assert message["frames"] == [
        {
            "child_frame_id": "tool0",
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "pose_provenance": "POSE_PROVENANCE_GROUND_TRUTH",
        }
    ]


def test_a_driver_topic_spec_resolves_to_the_same_topic() -> None:
    spec = TopicSpec(
        payload_schema_ref="TwinFramePosesPayload", namespace="twin", leaf="frame_poses"
    )
    assert resolve_topic_path(spec, TWIN, prefix="dev") == (
        f"devcyberwave/twin/{TWIN}/frame_poses"
    )


def test_the_wrapper_delegates() -> None:
    wrapper = WrapperMQTTClient.__new__(WrapperMQTTClient)
    wrapper._client = MagicMock()
    wrapper._client.publish_frame_poses.return_value = True

    assert (
        wrapper.publish_frame_poses(
            TWIN, [FRAME], substrate="hardware", pose_provenance="odometry"
        )
        is True
    )
    wrapper._client.publish_frame_poses.assert_called_once_with(
        TWIN,
        [FRAME],
        substrate="hardware",
        pose_provenance="odometry",
        frame_id="environment",
        source_subtype=None,
        producer_instance=None,
        producer_sequence=None,
        workload_uuid=None,
        simulation_time_s=None,
    )
