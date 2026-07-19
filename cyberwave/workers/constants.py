"""Shared constants for the worker subsystem."""

_MONITOR_PREFIX = "cw/_monitor"
_MONITOR_SUFFIX = "worker_stats"

MONITOR_STATS_WILDCARD = f"{_MONITOR_PREFIX}/*/{_MONITOR_SUFFIX}"
"""Zenoh wildcard that matches the stats key for every host.

Used by ``cyberwave worker monitor --all-hosts`` to subscribe to all
worker containers reachable on the local Zenoh mesh in one declaration.
"""


def build_monitor_stats_key(hostname: str) -> str:
    """Return the Zenoh key for a specific host's stats channel.

    Format: ``cw/_monitor/{hostname}/worker_stats``

    The hostname component is the value from ``/etc/hostname`` inside the
    container (set by Docker, matching
    ``docker inspect --format {{.Config.Hostname}}``).  Embedding the host
    in the key lets Zenoh filter at the routing layer — the subscriber
    receives only the messages it declared interest in, with no
    post-decode JSON comparison required.

    Published directly on the Zenoh session (not through DataBus) because
    the ``_monitor/`` prefix bypasses DataBus validation.
    """
    return f"{_MONITOR_PREFIX}/{hostname}/{_MONITOR_SUFFIX}"
