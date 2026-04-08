"""Tests for FlyingTwin hovering-status methods.

Covers:
- is_hovering() / get_hovering_status() from cached metadata
- set_hovering_status() — API call + local cache update
- takeoff() / land() / hover() automatic status updates in sim_tele mode
- takeoff() / land() / hover() do NOT touch status in live (tele) mode
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from cyberwave.twin import FlyingTwin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(*, runtime_mode: str = "simulation") -> SimpleNamespace:
    """Build a minimal client stub with the given runtime_mode."""
    mqtt = MagicMock()
    mqtt.connected = True
    twins_manager = MagicMock()
    return SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(
            runtime_mode=runtime_mode,
            source_type="sim" if runtime_mode == "simulation" else "edge",
            topic_prefix="",
        ),
        twins=twins_manager,
    )


def _make_flying_twin(
    *,
    runtime_mode: str = "simulation",
    metadata: dict | None = None,
) -> tuple[FlyingTwin, SimpleNamespace]:
    client = _make_client(runtime_mode=runtime_mode)
    data = SimpleNamespace(
        uuid="drone-uuid",
        name="Test Drone",
        metadata=metadata or {},
    )
    twin = FlyingTwin(client, data)
    return twin, client


# ---------------------------------------------------------------------------
# is_hovering() — reads local cache
# ---------------------------------------------------------------------------

class TestIsHovering:
    def test_returns_false_when_metadata_empty(self):
        twin, _ = _make_flying_twin(metadata={})
        assert twin.is_hovering() is False

    def test_returns_false_when_status_absent(self):
        twin, _ = _make_flying_twin(metadata={"drivers": {}})
        assert twin.is_hovering() is False

    def test_returns_false_when_hovering_false(self):
        twin, _ = _make_flying_twin(metadata={"status": {"controller_requested_hovering": False}})
        assert twin.is_hovering() is False

    def test_returns_true_when_hovering_true(self):
        twin, _ = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 3.0}}
        )
        assert twin.is_hovering() is True

    def test_returns_false_for_dict_data_missing_metadata(self):
        """_data stored as dict (alternative internal format)."""
        client = _make_client()
        twin = FlyingTwin(client, {"uuid": "d", "name": "D", "metadata": {}})
        assert twin.is_hovering() is False

    def test_returns_true_for_dict_data_with_hovering(self):
        client = _make_client()
        twin = FlyingTwin(
            client,
            {"uuid": "d", "name": "D", "metadata": {"status": {"controller_requested_hovering": True}}},
        )
        assert twin.is_hovering() is True


# ---------------------------------------------------------------------------
# get_hovering_status() — reads local cache
# ---------------------------------------------------------------------------

class TestGetHoveringStatus:
    def test_defaults_when_metadata_empty(self):
        twin, _ = _make_flying_twin(metadata={})
        status = twin.get_hovering_status()
        assert status == {"controller_requested_hovering": False, "controller_requested_hovering_altitude": None}

    def test_reflects_hovering_true_with_altitude(self):
        twin, _ = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.5}}
        )
        status = twin.get_hovering_status()
        assert status["controller_requested_hovering"] is True
        assert status["controller_requested_hovering_altitude"] == 2.5

    def test_reflects_hovering_false_no_altitude(self):
        twin, _ = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": False}}
        )
        status = twin.get_hovering_status()
        assert status["controller_requested_hovering"] is False
        assert status["controller_requested_hovering_altitude"] is None


# ---------------------------------------------------------------------------
# set_hovering_status() — calls API and updates local cache
# ---------------------------------------------------------------------------

class TestSetHoveringStatus:
    def test_calls_twins_update_with_merged_metadata(self):
        twin, client = _make_flying_twin(
            metadata={"drivers": {"default": {"docker_image": "img"}}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=2.0)

        client.twins.update.assert_called_once_with(
            "drone-uuid",
            metadata={
                "drivers": {"default": {"docker_image": "img"}},
                "status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.0},
            },
        )

    def test_updates_local_cache_after_api_call(self):
        twin, _ = _make_flying_twin(metadata={})
        twin.set_hovering_status(hovering=True, hovering_altitude=5.0)

        assert twin.is_hovering() is True
        assert twin.get_hovering_status()["controller_requested_hovering_altitude"] == 5.0

    def test_clears_altitude_when_landing(self):
        twin, client = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 3.0}}
        )
        twin.set_hovering_status(hovering=False)

        _, kwargs = client.twins.update.call_args
        assert "controller_requested_hovering_altitude" not in kwargs["metadata"]["status"]
        assert kwargs["metadata"]["status"]["controller_requested_hovering"] is False

    def test_preserves_existing_altitude_when_not_provided_on_land(self):
        """Altitude key is removed (not just set to None) on hovering=False."""
        twin, client = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.0}}
        )
        twin.set_hovering_status(hovering=False)
        updated_status = client.twins.update.call_args.kwargs["metadata"]["status"]
        assert "controller_requested_hovering_altitude" not in updated_status

    def test_does_not_overwrite_unrelated_altitude_when_hovering_with_explicit_value(self):
        twin, client = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": False, "controller_requested_hovering_altitude": 1.0}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=4.0)
        updated_status = client.twins.update.call_args.kwargs["metadata"]["status"]
        assert updated_status["controller_requested_hovering_altitude"] == 4.0

    def test_altitude_unchanged_when_hovering_true_and_no_altitude_passed(self):
        """Passing hovering=True without altitude leaves existing altitude intact."""
        twin, client = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": False, "controller_requested_hovering_altitude": 2.0}}
        )
        twin.set_hovering_status(hovering=True)
        updated_status = client.twins.update.call_args.kwargs["metadata"]["status"]
        # altitude was already in status and we didn't pass a new value — should be preserved
        assert updated_status["controller_requested_hovering_altitude"] == 2.0

    def test_raises_cyberwave_error_on_api_failure(self):
        from cyberwave.exceptions import CyberwaveError

        twin, client = _make_flying_twin()
        client.twins.update.side_effect = RuntimeError("network error")

        with pytest.raises(CyberwaveError, match="Failed to update hovering status"):
            twin.set_hovering_status(hovering=True, hovering_altitude=1.0)


# ---------------------------------------------------------------------------
# takeoff() in simulation mode — should auto-update hovering status
# ---------------------------------------------------------------------------

class TestTakeoffSimMode:
    def test_publishes_mqtt_takeoff_command(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff(altitude=3.0)

        client.mqtt.publish.assert_any_call(
            "twins/drone-uuid/commands/takeoff", {"altitude": 3.0}
        )

    def test_sets_hovering_true_with_altitude_in_sim_mode(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff(altitude=3.0)

        client.twins.update.assert_called_once()
        metadata = client.twins.update.call_args.kwargs["metadata"]
        assert metadata["status"]["controller_requested_hovering"] is True
        assert metadata["status"]["controller_requested_hovering_altitude"] == 3.0

    def test_uses_default_altitude_1m(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff()

        metadata = client.twins.update.call_args.kwargs["metadata"]
        assert metadata["status"]["controller_requested_hovering_altitude"] == 1.0

    def test_local_cache_reflects_hovering_after_takeoff(self):
        twin, _ = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff(altitude=2.5)

        assert twin.is_hovering() is True
        assert twin.get_hovering_status()["controller_requested_hovering_altitude"] == 2.5


# ---------------------------------------------------------------------------
# takeoff() in live mode — must NOT auto-update hovering status
# ---------------------------------------------------------------------------

class TestTakeoffLiveMode:
    def test_publishes_mqtt_takeoff_command(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.takeoff(altitude=2.0)

        client.mqtt.publish.assert_called_once_with(
            "twins/drone-uuid/commands/takeoff", {"altitude": 2.0}
        )

    def test_does_not_call_twins_update(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.takeoff(altitude=2.0)

        client.twins.update.assert_not_called()

    def test_local_cache_unchanged(self):
        twin, _ = _make_flying_twin(runtime_mode="live", metadata={})
        twin.takeoff(altitude=2.0)

        assert twin.is_hovering() is False


# ---------------------------------------------------------------------------
# land() in simulation mode — should auto-clear hovering status
# ---------------------------------------------------------------------------

class TestLandSimMode:
    def test_publishes_mqtt_land_command(self):
        twin, client = _make_flying_twin(
            runtime_mode="simulation",
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.0}},
        )
        twin.land()

        client.mqtt.publish.assert_any_call("twins/drone-uuid/commands/land", {})

    def test_sets_hovering_false_in_sim_mode(self):
        twin, client = _make_flying_twin(
            runtime_mode="simulation",
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.0}},
        )
        twin.land()

        metadata = client.twins.update.call_args.kwargs["metadata"]
        assert metadata["status"]["controller_requested_hovering"] is False
        assert "controller_requested_hovering_altitude" not in metadata["status"]

    def test_local_cache_reflects_not_hovering_after_land(self):
        twin, _ = _make_flying_twin(
            runtime_mode="simulation",
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.0}},
        )
        twin.land()

        assert twin.is_hovering() is False


# ---------------------------------------------------------------------------
# land() in live mode — must NOT auto-update hovering status
# ---------------------------------------------------------------------------

class TestLandLiveMode:
    def test_publishes_mqtt_land_command(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.land()

        client.mqtt.publish.assert_called_once_with(
            "twins/drone-uuid/commands/land", {}
        )

    def test_does_not_call_twins_update(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.land()

        client.twins.update.assert_not_called()


# ---------------------------------------------------------------------------
# hover() in simulation mode — should set hovering=True (no altitude change)
# ---------------------------------------------------------------------------

class TestHoverSimMode:
    def test_publishes_mqtt_hover_command(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.hover()

        client.mqtt.publish.assert_any_call("twins/drone-uuid/commands/hover", {})

    def test_sets_hovering_true_in_sim_mode(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.hover()

        metadata = client.twins.update.call_args.kwargs["metadata"]
        assert metadata["status"]["controller_requested_hovering"] is True

    def test_local_cache_reflects_hovering_after_hover(self):
        twin, _ = _make_flying_twin(runtime_mode="simulation")
        twin.hover()

        assert twin.is_hovering() is True


# ---------------------------------------------------------------------------
# hover() in live mode — must NOT auto-update hovering status
# ---------------------------------------------------------------------------

class TestHoverLiveMode:
    def test_does_not_call_twins_update(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.hover()

        client.twins.update.assert_not_called()


# ---------------------------------------------------------------------------
# Full workflow: takeoff → hover → land (sim mode)
# ---------------------------------------------------------------------------

class TestFullFlightWorkflow:
    def test_takeoff_hover_land_sequence_updates_status_correctly(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")

        twin.takeoff(altitude=2.0)
        assert twin.is_hovering() is True
        assert twin.get_hovering_status()["controller_requested_hovering_altitude"] == 2.0

        twin.hover()
        assert twin.is_hovering() is True

        twin.land()
        assert twin.is_hovering() is False
        assert twin.get_hovering_status()["controller_requested_hovering_altitude"] is None

    def test_mqtt_commands_published_in_order(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")

        twin.takeoff(altitude=1.5)
        twin.land()

        mqtt_calls = client.mqtt.publish.call_args_list
        topics = [c.args[0] for c in mqtt_calls]
        assert "twins/drone-uuid/commands/takeoff" in topics
        assert "twins/drone-uuid/commands/land" in topics
        assert topics.index("twins/drone-uuid/commands/takeoff") < topics.index(
            "twins/drone-uuid/commands/land"
        )
