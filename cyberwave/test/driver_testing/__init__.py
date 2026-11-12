"""Capability-driven driver testing: ``cyberwave-stress-test --uuid <twin>`` for any arm driver.

The pieces, in the order a run uses them:

* :func:`discover` / :func:`profile_from` -- what the twin can do
  (:class:`TwinProfile`): capabilities, catalog commands, universal schema
  joints and limits, ``state/motion`` backends;
* :data:`REGISTRY` / :func:`section` / :func:`resolve_plan` -- the sections that
  match the profile, and the reason each other one is skipped;
* :class:`StressHarness` -- drives the twin through the public SDK and decides
  pass or fail from measured feedback, matching results by ``motion_id``;
* :func:`write_report` -- ``report.json``, ``events.jsonl``, ``joints.csv``.

A robot-specific harness subclasses :class:`StressHarness`; a new capability
gets a section with the :func:`section` decorator. Importing this package moves
nothing and needs no ROS.
"""

from .discovery import JointInfo, TwinProfile, discover, profile_from
from .feedback import (
    Feedback,
    Teleop,
    Ticket,
    euler_pose,
    offset_along_z,
    target_pose,
    tracking_metrics,
)
from .harness import StressHarness, is_collision_refusal
from .registry import (
    REGISTRY,
    CommandCoverage,
    PlanEntry,
    Section,
    SectionRegistry,
    SectionSkipped,
    command_coverage,
    resolve_plan,
    section,
)
from .report import write_report

__all__ = [
    "REGISTRY",
    "CommandCoverage",
    "Feedback",
    "JointInfo",
    "PlanEntry",
    "Section",
    "SectionRegistry",
    "SectionSkipped",
    "StressHarness",
    "Teleop",
    "Ticket",
    "TwinProfile",
    "command_coverage",
    "discover",
    "euler_pose",
    "is_collision_refusal",
    "offset_along_z",
    "profile_from",
    "resolve_plan",
    "section",
    "target_pose",
    "tracking_metrics",
    "write_report",
]
