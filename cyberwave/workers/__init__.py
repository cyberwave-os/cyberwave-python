"""Worker hook registry, callback context, module loader, and runtime.

Re-exports the public API surface needed by worker modules and the
worker runtime.  Client integration (``@cw.on_frame`` ergonomic API
and ``Cyberwave._hook_registry``) ships with CYB-1557.
"""

from .constants import MONITOR_STATS_KEY
from .context import HookContext
from .hooks import HookRegistration, HookRegistry, ScheduleRegistration, SynchronizedGroup
from .loader import load_workers
from .runtime import WorkerRuntime

__all__ = [
    "HookContext",
    "HookRegistration",
    "HookRegistry",
    "MONITOR_STATS_KEY",
    "ScheduleRegistration",
    "SynchronizedGroup",
    "WorkerRuntime",
    "load_workers",
]
