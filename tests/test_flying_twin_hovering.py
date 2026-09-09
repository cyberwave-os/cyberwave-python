"""Tests for FlyingTwin hovering-status methods.

Covers:
- is_hovering() / get_hovering_status() from cached metadata
- set_hovering_status() — API call + local cache update
- takeoff() / land() / hover() are PLAYGROUND-compatible: they publish (and
  mirror hovering status, since there's no edge driver to do it) in a
  simulation runtime
- takeoff() / land() / hover() do NOT touch status in live (tele) mode

The MQTT-side contract (canonical command topic / payload envelope /
source-type validation) is exercised in
``test_flying_twin_drone_commands.py`` to keep this file focused on the
hovering-metadata bookkeeping that lives next to the SDK methods.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

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
    twins_manager.update.return_value = SimpleNamespace(uuid="drone-uuid", metadata={})
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={"mqtt": {"topics": {}, "commands": {"supported": []}}}
    )
    policy = MagicMock()
    policy.ensure_attached = MagicMock()

    client = SimpleNamespace(
        mqtt=mqtt,
        assets=assets,
        policy=policy,
        config=SimpleNamespace(
            runtime_mode=runtime_mode,
            source_type="sim" if runtime_mode == "simulation" else "edge",
            topic_prefix="",
        ),
        twins=twins_manager,
    )
    return client


def _make_flying_twin(
    *,
    runtime_mode: str = "simulation",
    metadata: dict | None = None,
) -> tuple[FlyingTwin, SimpleNamespace]:
    client = _make_client(runtime_mode=runtime_mode)
    data = SimpleNamespace(
        uuid="drone-uuid",
        name="Test Drone",
        asset_uuid="asset-uuid",
        metadata=metadata or {},
    )
    twin = FlyingTwin(client, data)
    twin._prepare_outbound_command = lambda: None  # type: ignore[method-assign]
    return twin, client


def _published_command(twin: FlyingTwin, command: str) -> dict:
    """Return the canonical-command payload from PR1 mock outbound log."""
    matching = [r.payload for r in twin._outbound_log if r.command == command]
    assert matching, (
        f"No mock outbound command {command!r}; log: "
        f"{[r.command for r in twin._outbound_log]}"
    )
    return matching[-1]


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
    """Intent goes through POST /flight-request, never PUT /twins.

    `PUT /twins` shallow-merges only TOP-LEVEL metadata keys, so sending
    `status` replaces the whole dict -- wiping `is_flying` / `is_flying_at` /
    `flight_mode`, which only the edge may write and which the
    redundant-takeoff guard reads. `takeoff()` calls this from `sim_tele`, so
    the clobber raced a live aircraft report. The server merges under a row
    lock, so these tests assert on the INTENT sent, and on the local cache --
    not on a metadata blob the client no longer builds a request from.
    """

    def test_sends_intent_through_flight_request(self):
        twin, client = _make_flying_twin(
            metadata={"drivers": {"default": {"docker_image": "img"}}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=2.0)

        client.twins.set_flight_request.assert_called_once_with(
            "drone-uuid",
            hovering=True,
            hovering_altitude=2.0,
        )

    def test_never_puts_the_whole_metadata_blob(self):
        """The regression this endpoint exists for."""
        twin, client = _make_flying_twin(
            metadata={"status": {"is_flying": True, "is_flying_at": 1234.5}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=2.0)

        client.twins.update.assert_not_called()
        client.twins.set_flight_request.assert_called_once()

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

        # No altitude is sent, which is what tells the server to clear it.
        client.twins.set_flight_request.assert_called_once_with(
            "drone-uuid",
            hovering=False,
            hovering_altitude=None,
        )
        assert (
            "controller_requested_hovering_altitude"
            not in twin._data.metadata["status"]
        )
        assert twin.is_hovering() is False

    def test_keeps_unrelated_metadata_in_the_local_cache(self):
        twin, _ = _make_flying_twin(
            metadata={"drivers": {"default": {"docker_image": "img"}}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=4.0)

        assert twin._data.metadata["drivers"] == {"default": {"docker_image": "img"}}

    def test_local_cache_keeps_the_aircraft_keys_beside_the_intent(self):
        """The cache update must not clobber what the endpoint deliberately keeps.

        `is_flying` / `is_flying_at` live in the same `status` dict as the
        intent, and only the edge may write them. Rewriting the dict wholesale
        from the cached snapshot would drop them locally, which is the same
        mistake the endpoint exists to avoid on the server.
        """
        twin, _ = _make_flying_twin(
            metadata={"status": {"is_flying": True, "is_flying_at": 123.0}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=2.0)

        status = twin._data.metadata["status"]
        assert status["is_flying"] is True
        assert status["is_flying_at"] == 123.0
        assert status["controller_requested_hovering"] is True

    def test_explicit_altitude_wins_over_the_cached_one(self):
        twin, client = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": False, "controller_requested_hovering_altitude": 1.0}}
        )
        twin.set_hovering_status(hovering=True, hovering_altitude=4.0)

        assert client.twins.set_flight_request.call_args.kwargs["hovering_altitude"] == 4.0
        assert twin.get_hovering_status()["controller_requested_hovering_altitude"] == 4.0

    def test_altitude_unchanged_when_hovering_true_and_no_altitude_passed(self):
        """Passing hovering=True without altitude leaves the existing one intact.

        Nothing is sent for it, so the server leaves its copy alone too.
        """
        twin, client = _make_flying_twin(
            metadata={"status": {"controller_requested_hovering": False, "controller_requested_hovering_altitude": 2.0}}
        )
        twin.set_hovering_status(hovering=True)

        assert client.twins.set_flight_request.call_args.kwargs["hovering_altitude"] is None
        assert twin.get_hovering_status()["controller_requested_hovering_altitude"] == 2.0

    def test_raises_cyberwave_error_on_api_failure(self):
        from cyberwave.exceptions import CyberwaveError

        twin, client = _make_flying_twin()
        client.twins.set_flight_request.side_effect = RuntimeError("network error")

        with pytest.raises(CyberwaveError, match="Failed to update hovering status"):
            twin.set_hovering_status(hovering=True, hovering_altitude=1.0)


# ---------------------------------------------------------------------------
# takeoff() in simulation mode — live/driver-only, raises NotSimulatedError
# ---------------------------------------------------------------------------

class TestTakeoffSimMode:
    def test_takeoff_publishes_and_sets_hovering_in_sim_mode(self):
        # Flight is PLAYGROUND-compatible: the browser playground renders
        # takeoff/land/hover directly, so they publish (rather than raise) in
        # simulation mode, and sim_tele takeoff mirrors hovering status since
        # there's no edge driver to report motors-on.
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff(altitude=3.0)

        assert _published_command(twin, "takeoff")
        client.twins.set_flight_request.assert_called()
        assert twin.is_hovering() is True

    def test_takeoff_sets_hovering_altitude_in_sim_mode(self):
        twin, _ = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff(altitude=2.5)
        assert twin.is_hovering() is True


# ---------------------------------------------------------------------------
# takeoff() in live mode — must NOT auto-update hovering status
# ---------------------------------------------------------------------------

class TestTakeoffLiveMode:
    def test_publishes_canonical_takeoff_command_with_tele_source(self):
        twin, _client = _make_flying_twin(runtime_mode="live")
        twin.takeoff(altitude=2.0)

        payload = _published_command(twin, "takeoff")
        assert payload["source_type"] == "tele"
        assert payload["data"] == {"altitude": 2.0}

    def test_does_not_call_twins_update(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.takeoff(altitude=2.0)

        client.twins.update.assert_not_called()
        client.twins.set_flight_request.assert_not_called()

    def test_local_cache_unchanged(self):
        twin, _ = _make_flying_twin(runtime_mode="live", metadata={})
        twin.takeoff(altitude=2.0)

        assert twin.is_hovering() is False


# ---------------------------------------------------------------------------
# land() in simulation mode — live/driver-only, raises NotSimulatedError
# ---------------------------------------------------------------------------

class TestLandSimMode:
    def test_land_publishes_and_clears_hovering_in_sim_mode(self):
        twin, client = _make_flying_twin(
            runtime_mode="simulation",
            metadata={"status": {"controller_requested_hovering": True, "controller_requested_hovering_altitude": 2.0}},
        )
        twin.land()

        assert _published_command(twin, "land")
        client.twins.set_flight_request.assert_called()
        assert twin.is_hovering() is False


# ---------------------------------------------------------------------------
# land() in live mode — must NOT auto-update hovering status
# ---------------------------------------------------------------------------

class TestLandLiveMode:
    def test_publishes_canonical_land_command_with_tele_source(self):
        twin, _client = _make_flying_twin(runtime_mode="live")
        twin.land()

        payload = _published_command(twin, "land")
        assert payload["source_type"] == "tele"
        assert payload["data"] == {}

    def test_does_not_call_twins_update(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.land()

        client.twins.update.assert_not_called()
        client.twins.set_flight_request.assert_not_called()


# ---------------------------------------------------------------------------
# hover() in simulation mode — live/driver-only, raises NotSimulatedError
# ---------------------------------------------------------------------------

class TestHoverSimMode:
    def test_hover_publishes_and_sets_hovering_in_sim_mode(self):
        twin, client = _make_flying_twin(runtime_mode="simulation")
        twin.hover()

        assert _published_command(twin, "hover")
        client.twins.set_flight_request.assert_called()
        assert twin.is_hovering() is True


# ---------------------------------------------------------------------------
# hover() in live mode — must NOT auto-update hovering status
# ---------------------------------------------------------------------------

class TestHoverLiveMode:
    def test_does_not_call_twins_update(self):
        twin, client = _make_flying_twin(runtime_mode="live")
        twin.hover()

        client.twins.update.assert_not_called()
        client.twins.set_flight_request.assert_not_called()


# ---------------------------------------------------------------------------
# Full workflow: takeoff → hover → land (live mode)
# ---------------------------------------------------------------------------

class TestFullFlightWorkflow:
    def test_live_takeoff_land_published_in_order_without_status_writes(self):
        twin, client = _make_flying_twin(runtime_mode="live")

        twin.takeoff(altitude=1.5)
        twin.land()

        canonical_commands = [r.command for r in twin._outbound_log]
        assert "takeoff" in canonical_commands
        assert "land" in canonical_commands
        assert canonical_commands.index("takeoff") < canonical_commands.index("land")
        # Live mode: the edge driver owns hovering status, not the SDK.
        client.twins.update.assert_not_called()
        client.twins.set_flight_request.assert_not_called()

    def test_sim_takeoff_then_land_updates_hovering_status(self):
        twin, _ = _make_flying_twin(runtime_mode="simulation")
        twin.takeoff(altitude=2.0)
        assert twin.is_hovering() is True
        twin.land()
        assert twin.is_hovering() is False
