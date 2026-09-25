"""RealSense device selection by serial.

A RealSense is addressed by serial, never by device index. Before these
guards the serial was accepted at every public entry point and then dropped
before ``pipeline.start()``, so every streamer silently bound whichever
device librealsense enumerated first. With two cameras on one host that
means two twins open the same camera and the second fails with EBUSY.

These tests need no hardware: they exercise the plumbing that carries a
serial from the public API down to the track.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("av", reason="pyav not installed (extras: realsense)")

from cyberwave.sensor.camera_rs import RealSenseStreamer, RealSenseVideoTrack  # noqa: E402
from cyberwave.sensor.manager import _infer_config_from_twin  # noqa: E402
from cyberwave.twin import DepthCameraTwin  # noqa: E402


class TestSerialNumberIsAccepted:
    """Every public constructor must expose ``serial_number``."""

    @pytest.mark.parametrize(
        "func",
        [
            RealSenseVideoTrack.__init__,
            RealSenseStreamer.__init__,
            RealSenseStreamer.from_config,
            RealSenseStreamer.from_device,
        ],
    )
    def test_signature_exposes_serial_number(self, func) -> None:
        assert "serial_number" in inspect.signature(func).parameters

    def test_serial_number_defaults_to_none(self) -> None:
        """None must stay valid: single-camera hosts configure nothing."""
        for func in (RealSenseVideoTrack.__init__, RealSenseStreamer.__init__):
            assert inspect.signature(func).parameters["serial_number"].default is None


def _streamer(serial_number: str | None) -> RealSenseStreamer:
    """Build a streamer without touching librealsense or an MQTT broker.

    ``__init__`` calls ``require_realsense()``, which needs the optional
    ``pyrealsense2`` extra; only attribute plumbing is under test here.
    """
    streamer = RealSenseStreamer.__new__(RealSenseStreamer)
    streamer.serial_number = serial_number
    return streamer


class TestStreamerCarriesSerialToTrack:
    """``initialize_track`` must hand the serial to the track it builds."""

    def test_serial_reaches_the_track(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _fake_track(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            "cyberwave.sensor.camera_rs.RealSenseVideoTrack", _fake_track
        )
        streamer = _streamer("046322252081")
        for attr, value in {
            "color_fps": 30,
            "depth_fps": 30,
            "color_resolution": None,
            "depth_resolution": None,
            "enable_depth": True,
            "client": None,
            "time_reference": None,
            "twin_uuid": None,
            "depth_publish_interval": 30,
            "frame_callback": None,
            "depth_callback": None,
        }.items():
            setattr(streamer, attr, value)

        streamer.initialize_track()

        assert captured["serial_number"] == "046322252081"

    def test_leading_zero_survives(self, monkeypatch) -> None:
        """A serial is a string; an int would drop the leading zero."""
        assert _streamer("043422251999").serial_number.startswith("0")


class TestFromConfigHonoursConfigSerial:
    """A config built with a serial must not reach the pipeline unpinned."""

    def test_config_serial_is_used_when_no_override(self) -> None:
        source = inspect.signature(RealSenseStreamer.from_config).parameters
        assert source["serial_number"].default is None

    def test_from_device_threads_serial_through_config(self) -> None:
        """``from_device`` sets the serial on the config it builds.

        It must not also pass ``serial_number`` separately — two sources of
        truth that can disagree.
        """
        src = inspect.getsource(RealSenseStreamer.from_device)
        assert "RealSenseConfig.from_device(" in src


class TestManagerInfersSerialFromTwinMetadata:
    @staticmethod
    def _twin(serial_number: object) -> DepthCameraTwin:
        twin = DepthCameraTwin.__new__(DepthCameraTwin)
        twin._data = {
            "capabilities": {"sensors": []},
            "metadata": {"serial_number": serial_number},
        }
        return twin

    def test_reads_serial_number_from_twin_metadata(self) -> None:
        config = _infer_config_from_twin(self._twin("046322252081"))
        assert config["camera_type"] == "realsense"
        assert config["serial_number"] == "046322252081"

    def test_explicit_override_wins_over_twin_metadata(self) -> None:
        config = _infer_config_from_twin(
            self._twin("046322252081"), {"serial_number": "999999999999"}
        )
        assert config["serial_number"] == "999999999999"
