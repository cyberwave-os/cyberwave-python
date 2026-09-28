"""Central capability → handle resolution."""

from cyberwave.twin.capability_resolve import resolve_handler_from_capabilities

GO2_CAPS = {
    "can_locomote": True,
    "has_joints": True,
    "sensors": [
        {"id": "lidar_4d", "type": "lidar_4d"},
        {"id": "front_camera", "type": "rgb"},
    ],
}


def test_resolve_lidar_and_camera_for_go2() -> None:
    lidar = resolve_handler_from_capabilities(GO2_CAPS, "lidar")
    camera = resolve_handler_from_capabilities(GO2_CAPS, "camera")
    assert lidar.available
    assert lidar.sensor_ids == ("lidar_4d",)
    assert lidar.default_sensor_id == "lidar_4d"
    assert camera.available
    assert camera.sensor_ids == ("front_camera",)
    assert camera.default_sensor_id == "front_camera"


def test_resolve_locomotion_flag() -> None:
    assert resolve_handler_from_capabilities(GO2_CAPS, "locomotion").available
    assert not resolve_handler_from_capabilities(GO2_CAPS, "flight").available


def test_resolve_unknown_handler() -> None:
    assert not resolve_handler_from_capabilities(GO2_CAPS, "hoverboard").available


def test_resolve_gps_imu_compass() -> None:
    caps = {
        "sensors": [
            {"id": "nav_gps", "type": "gps"},
            {"id": "body_imu", "type": "imu"},
            {"id": "mag", "type": "compass"},
        ]
    }
    assert resolve_handler_from_capabilities(caps, "gps").sensor_ids == ("nav_gps",)
    assert resolve_handler_from_capabilities(caps, "imu").sensor_ids == ("body_imu",)
    assert resolve_handler_from_capabilities(caps, "compass").sensor_ids == ("mag",)


def test_resolve_microphone_and_speaker() -> None:
    caps = {
        "sensors": [
            {"id": "g1-vui-mic", "type": "microphone"},
            {"id": "g1-vui-speaker", "type": "speaker"},
        ]
    }
    mic = resolve_handler_from_capabilities(caps, "microphone")
    speaker = resolve_handler_from_capabilities(caps, "speaker")
    assert mic.available
    assert mic.sensor_ids == ("g1-vui-mic",)
    assert mic.default_sensor_id == "g1-vui-mic"
    assert speaker.available
    assert speaker.sensor_ids == ("g1-vui-speaker",)
    assert speaker.default_sensor_id == "g1-vui-speaker"
    # "mic" is an accepted alias for "microphone".
    assert resolve_handler_from_capabilities(caps, "mic").sensor_ids == ("g1-vui-mic",)


def test_resolve_microphone_accepts_type_aliases() -> None:
    for alias in ("mic", "microphone", "audio_in", "audio", "audio_mono", "audio_stereo"):
        caps = {"sensors": [{"id": "s1", "type": alias}]}
        assert resolve_handler_from_capabilities(caps, "microphone").available, alias


def test_resolve_speaker_accepts_type_aliases() -> None:
    for alias in ("speaker", "loudspeaker", "speakerphone", "audio_out"):
        caps = {"sensors": [{"id": "s1", "type": alias}]}
        assert resolve_handler_from_capabilities(caps, "speaker").available, alias


def test_resolve_microphone_unavailable_when_twin_has_no_such_sensor() -> None:
    # A twin that only declares a camera must not report a microphone/speaker
    # available -- callers rely on this to skip starting a sensor the twin
    # never declared, rather than assuming every unit of a given asset model
    # has the same physically-installed sensors.
    caps = {"sensors": [{"id": "front_camera", "type": "rgb"}]}
    assert not resolve_handler_from_capabilities(caps, "microphone").available
    assert not resolve_handler_from_capabilities(caps, "speaker").available
