"""Shared constants for the worker subsystem."""

MONITOR_STATS_KEY = "cw/_monitor/worker_stats"
"""Raw Zenoh key expression for the monitor stats channel.

Published directly on the Zenoh session (not through DataBus) because
the key uses an underscore prefix that doesn't pass DataBus validation.
Shared between the SDK runtime (publisher) and the CLI monitor (subscriber).
"""
