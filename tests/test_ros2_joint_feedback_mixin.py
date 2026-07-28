"""Ros2JointFeedbackMixin: state accumulation, name map, edge payload, first-RX hook."""

from __future__ import annotations

from types import SimpleNamespace

from cyberwave.constants import SOURCE_TYPE_EDGE
from cyberwave.driver.ros2.joint_names import JointNameMap
from cyberwave.driver.ros2.joint_feedback import Ros2JointFeedbackMixin


def _msg(names, positions):
    return SimpleNamespace(name=list(names), position=list(positions))


class _FakeLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class _Driver(Ros2JointFeedbackMixin):
    use_joint_feedback = True

    def __init__(self, name_map=None, topic="joint_states_single"):
        self._name_map = name_map
        self._topic = topic
        self.subscriptions: list[tuple] = []
        self.first_feedback_calls = 0
        self._init_joint_feedback()

    # seams
    def joint_feedback_topic(self):
        return self._topic

    def joint_name_map(self):
        return self._name_map

    def on_first_joint_feedback(self):
        self.first_feedback_calls += 1

    # fake node surface
    def create_subscription(self, msg_type, topic, cb, qos):
        self.subscriptions.append((msg_type, topic, cb, qos))
        return object()

    def get_logger(self):
        return _FakeLogger()


def test_feedback_accumulates_and_reads_back():
    d = _Driver()
    d._on_joint_feedback(_msg(["j1", "j2"], [0.1, 0.2]))
    d._on_joint_feedback(_msg(["j2"], [0.3]))
    assert d._current_joint_positions() == {"j1": 0.1, "j2": 0.3}


def test_name_map_applied_with_mimic():
    nm = JointNameMap(
        ros_to_platform={"gripper": "joint7"}, mimic={"joint8": ("joint7", -1.0)}
    )
    d = _Driver(name_map=nm)
    d._on_joint_feedback(_msg(["joint1", "gripper"], [0.5, 0.02]))
    assert d._current_joint_positions() == {"joint1": 0.5, "joint7": 0.02, "joint8": -0.02}


def test_first_feedback_hook_fires_exactly_once():
    d = _Driver()
    d._on_joint_feedback(_msg(["j1"], [0.1]))
    d._on_joint_feedback(_msg(["j1"], [0.2]))
    assert d.first_feedback_calls == 1
    assert d._joint_feedback_count == 2


def test_empty_msg_ignored():
    d = _Driver()
    d._on_joint_feedback(_msg([], []))
    assert d._current_joint_positions() == {}
    assert d.first_feedback_calls == 0


def test_convert_joints_to_payload_edge_shape():
    d = _Driver()
    payload = d.convert_joints_to_payload(_msg(["j1"], [0.4]))
    assert payload is not None
    assert payload["source_type"] == SOURCE_TYPE_EDGE
    assert payload["j1"] == 0.4
    assert "timestamp" in payload


def test_convert_joints_to_payload_none_on_empty():
    assert _Driver().convert_joints_to_payload(_msg([], [])) is None


def test_convert_applies_binarize_when_present():
    class _WithBinarize(_Driver):
        def binarize_gripper(self, payload):
            return {**payload, "j1": 999.0}

    payload = _WithBinarize().convert_joints_to_payload(_msg(["j1"], [0.4]))
    assert payload["j1"] == 999.0


def test_register_callbacks_gated_on_flag():
    class _Disabled(_Driver):
        use_joint_feedback = False

    d = _Disabled()
    d.register_callbacks()
    assert d.subscriptions == []
