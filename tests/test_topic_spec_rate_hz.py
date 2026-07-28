"""TopicSpec.rate_hz exists and ROS stream cap falls through to BaseDriver default."""
import pytest

from cyberwave.driver.interface.args import TopicSpec


def test_topic_spec_has_optional_rate_hz():
    spec = TopicSpec(namespace="joint", leaf="update", payload_schema_ref="X")
    assert spec.rate_hz is None
    spec2 = TopicSpec(namespace="joint", leaf="update", payload_schema_ref="X", rate_hz=25.0)
    assert spec2.rate_hz == 25.0


@pytest.mark.parametrize("bad", [0, 0.0, -5.0])
def test_topic_spec_rejects_non_positive_rate_hz(bad):
    # 0 must NOT silently mean "unlimited" (the rate limiter treats max_hz <= 0
    # as uncapped) — it is a configuration error.
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        TopicSpec(namespace="joint", leaf="update", payload_schema_ref="X", rate_hz=bad)


from cyberwave.driver.base import BaseDriver


class _FakeDriver:
    """Exercise ros_stream_publish_max_hz resolution without rclpy."""
    STREAM_PUBLISH_MAX_HZ = 30.0

    def __init__(self, mqtt_max_hz=None):
        self._mqtt_max_hz = mqtt_max_hz

    # reuse the real resolution logic
    stream_publish_max_hz = BaseDriver.stream_publish_max_hz
    ros_stream_key = staticmethod(lambda t: f"ros:{t.lstrip('/')}")

    def ros_stream_publish_max_hz(self, ros_topic):
        return self.stream_publish_max_hz(self.ros_stream_key(ros_topic))


def test_ros_cap_defaults_to_basedriver_30():
    assert _FakeDriver().ros_stream_publish_max_hz("/joint_states") == 30.0


def test_ros_cap_uses_manifest_override():
    assert _FakeDriver(mqtt_max_hz=18.0).ros_stream_publish_max_hz("/joint_states") == 18.0
