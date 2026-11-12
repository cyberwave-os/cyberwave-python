"""Run a compiled cloud workflow.

``run_execution`` is the runner (``python -m cyberwave.workers.cloud_runner``): it
holds the run for its whole life, blocks on waits like an edge worker, and a runner
that dies ends the run. See ``runner``.

The Django-free half of the journaling runner (``context``, ``segment``, ``ports``,
``wait_results``) lives here too, shared with the backend which owns the journal:
a segment that can resume after a crash or a long wait. Nothing launches it yet;
import it from its submodule.
"""

from .runner import RunnerContext, run_execution
from .segment import SegmentOutcome

__all__ = ["RunnerContext", "SegmentOutcome", "run_execution"]
