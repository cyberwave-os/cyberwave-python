"""Tests specific to the ZenohBackend implementation.

These are skipped when ``eclipse-zenoh`` is not installed.
"""

from unittest.mock import patch

import pytest

from cyberwave.data.exceptions import BackendUnavailableError

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False

pytestmark = pytest.mark.skipif(not _has_zenoh, reason="eclipse-zenoh not installed")


@pytest.fixture
def backend():
    from cyberwave.data.zenoh_backend import ZenohBackend

    be = ZenohBackend()
    yield be
    be.close()


class TestZenohSession:
    def test_session_opens(self, backend):
        assert backend._session is not None

    def test_close_idempotent(self, backend):
        backend.close()
        backend.close()


class TestSharedMemoryConfig:
    def test_shared_memory_flag_accepted(self):
        from cyberwave.data.zenoh_backend import ZenohBackend

        try:
            be = ZenohBackend(shared_memory=True)
        except BackendUnavailableError:
            pytest.skip("POSIX shared memory not available on this platform")
        assert be._session is not None
        be.close()


class TestConnectEndpoints:
    def test_custom_endpoints_accepted(self):
        from cyberwave.data.zenoh_backend import ZenohBackend

        be = ZenohBackend(connect=["tcp/127.0.0.1:7447"])
        assert be._session is not None
        be.close()


class TestPeerLinkLogging:
    """``zid()`` answers from local state, so it stays healthy after every
    remote peer disappears. The peer-count sample is the only signal that a
    driver is publishing into an empty bus (CYB-3201).
    """

    @staticmethod
    def _probe(backend, counts: list[int | None], caplog, level="INFO"):
        """Feed *counts* to the watchdog helper, one probe each."""
        with caplog.at_level(level, logger="cyberwave.data.zenoh_backend"):
            for count in counts:
                with patch.object(backend, "_peer_link_count", return_value=count):
                    backend._log_peer_link_changes()
        return caplog.text

    def test_losing_the_last_peer_warns_immediately(self, backend, caplog):
        """The CYB-3202 signature: linked, then the worker restarts."""
        backend._peer_links = 2
        text = self._probe(backend, [0], caplog, level="WARNING")
        assert "no remote peers" in text
        assert backend._peer_links == 0

    def test_cold_start_is_quiet_until_the_grace_period_elapses(self, backend, caplog):
        """A driver that boots before its worker is alone but healthy."""
        text = self._probe(backend, [0, 0], caplog, level="WARNING")
        assert text == ""

    def test_sustained_zero_eventually_warns(self, backend, caplog):
        text = self._probe(backend, [0, 0, 0], caplog, level="WARNING")
        assert text.count("no remote peers") == 1

    def test_at_most_one_warning_per_zero_peer_episode(self, backend, caplog):
        """Otherwise a permanently isolated driver warns every 5 s forever."""
        backend._peer_links = 1
        text = self._probe(backend, [0, 0, 0, 0, 0], caplog, level="WARNING")
        assert text.count("no remote peers") == 1

    def test_rejoin_logs_info_and_rearms_the_warning(self, backend, caplog):
        backend._peer_links = 1
        text = self._probe(backend, [0, 2, 0], caplog)
        assert text.count("no remote peers") == 2
        assert "linked to 2 remote peer(s)" in text

    def test_steady_state_is_silent(self, backend, caplog):
        backend._peer_links = 1
        assert self._probe(backend, [1, 1, 1], caplog) == ""

    def test_unknown_count_never_reported_as_zero(self, backend, caplog):
        """An older binding without ``peers_zid`` must not fake a disconnect."""
        backend._peer_links = 2
        assert self._probe(backend, [None, None, None], caplog, level="WARNING") == ""
        assert backend._peer_links == 2

    def test_peer_count_survives_missing_binding_api(self, backend):
        class _NoPeersInfo:
            def zid(self):
                return "zid"

        with patch.object(backend, "_session_info", return_value=_NoPeersInfo()):
            assert backend._peer_link_count() is None

    def test_reconnect_clears_peer_state(self, backend):
        """A new session must not inherit the old one's tally or warned flag."""
        backend._peer_links = 3
        backend._zero_peer_probes = 7
        backend._zero_peer_warned = True

        with patch.object(backend, "_open_session", return_value=backend._session):
            with patch.object(backend, "_resubscribe_all"):
                backend._reconnect()

        assert backend._peer_links is None
        assert backend._zero_peer_probes == 0
        assert backend._zero_peer_warned is False


class TestImportError:
    def test_missing_zenoh_gives_clear_error(self):
        with patch.dict("sys.modules", {"zenoh": None}):
            import importlib

            from cyberwave.data import zenoh_backend

            orig = zenoh_backend._has_zenoh
            zenoh_backend._has_zenoh = False
            try:
                with pytest.raises(BackendUnavailableError, match="eclipse-zenoh"):
                    zenoh_backend.ZenohBackend()
            finally:
                zenoh_backend._has_zenoh = orig
