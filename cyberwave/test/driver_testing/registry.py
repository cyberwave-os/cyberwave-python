"""The map from what a driver exposes to the test section that exercises it.

Every :class:`Section` declares what it needs from the twin (catalog commands,
feedback topics, capability flags, joints, a planned-motion backend) and the
callable that drives it. :func:`resolve_plan` matches the registry against a
:class:`~cyberwave.test.driver_testing.discovery.TwinProfile` and returns every section with
either nothing in its way or the reason it is skipped -- before anything moves.

Adding a section for a new capability is one decorated function::

    from cyberwave.test.driver_testing import section

    @section("conveyor.belt", area="conveyor", summary="Belt runs and stops",
             commands={"belt_start", "belt_stop"}, topics={"joints"})
    def belt(harness):
        ...
        return {"measured_speed": speed}  # metrics for the report
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .discovery import TwinProfile

#: A section body: drives the twin through the harness, returns report metrics,
#: raises on failure and :class:`SectionSkipped` when the run reveals it cannot
#: apply (e.g. a backend that turns out to be unavailable).
SectionBody = Callable[..., "dict[str, Any] | None"]


#: How a missing topic family is named in a skip reason.
_TOPIC_NAMES = {
    "joints": "joint/update feedback",
    "pose": "twin/pose or state/kinematics feedback",
    "kinematics": "state/kinematics feedback",
    "motion": "state/motion status",
    "navigate": "navigate block",
}


class SectionSkipped(Exception):
    """Raised by a section that cannot apply to this twin; the reason is reported."""


@dataclass(frozen=True)
class Section:
    """One test section and what it needs from the twin."""

    id: str
    area: str
    summary: str
    body: SectionBody
    #: Every one of these catalog commands.
    commands: frozenset[str] = frozenset()
    #: At least one of these catalog commands (empty: no such need).
    any_commands: frozenset[str] = frozenset()
    #: Topic families from the catalog (``joints``, ``pose``, ``motion``, ...).
    topics: frozenset[str] = frozenset()
    #: ``twin.capabilities`` flags that must be truthy.
    capabilities: frozenset[str] = frozenset()
    needs_arm_joints: bool = False
    needs_gripper_joint: bool = False
    #: Run once per planned-motion backend, as ``<id>@<backend>``.
    per_backend: bool = False
    #: Further check on the profile: a skip reason, or ``None``.
    check: Callable[[TwinProfile], str | None] | None = None
    #: Catalog commands this section exercises, for the coverage table.
    #: Default: every command in ``commands`` and ``any_commands``.
    covers: frozenset[str] | None = None

    @property
    def covered_commands(self) -> frozenset[str]:
        return (
            self.covers
            if self.covers is not None
            else self.commands | self.any_commands
        )

    def skip_reason(self, profile: TwinProfile) -> str | None:
        """Why this section cannot run on *profile*, or ``None`` when it can."""
        missing = sorted(self.commands - profile.commands)
        if missing:
            return f"catalog lacks {', '.join(missing)}"
        if self.any_commands and not (self.any_commands & profile.commands):
            return f"catalog has none of {', '.join(sorted(self.any_commands))}"
        available = set(profile.topics)
        if "kinematics" in available:
            # Either pose feed will do: state/kinematics (the TCP) or twin/pose.
            available.add("pose")
        missing = sorted(self.topics - available)
        if missing:
            return "catalog declares no " + ", ".join(
                _TOPIC_NAMES.get(t, t) for t in missing
            )
        flags = sorted(f for f in self.capabilities if not profile.capabilities.get(f))
        if flags:
            return f"twin capabilities lack {', '.join(flags)}"
        if self.needs_arm_joints and not profile.arm_joints:
            return "universal schema declares no arm joints"
        if self.needs_gripper_joint and not profile.gripper_joint:
            return "universal schema declares no gripper joint"
        if self.per_backend and not profile.backends:
            return "state/motion reports no planned-motion backend"
        if self.check is not None:
            return self.check(profile)
        return None


@dataclass(frozen=True)
class PlanEntry:
    """One section as it will run (or why it will not)."""

    id: str
    section: Section
    backend: str | None = None
    skipped: str | None = None
    #: Left out by ``--sections`` / ``--skip`` rather than by the twin.
    deselected: bool = False

    @property
    def runs(self) -> bool:
        return self.skipped is None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "area": self.section.area,
            "summary": self.section.summary,
            "backend": self.backend,
            "outcome": "planned" if self.runs else "skipped",
            "reason": self.skipped,
        }


@dataclass
class SectionRegistry:
    """Ordered sections; the order is the run order."""

    sections: list[Section] = field(default_factory=list)

    def register(self, entry: Section) -> Section:
        if any(existing.id == entry.id for existing in self.sections):
            raise ValueError(f"Section {entry.id!r} is already registered")
        self.sections.append(entry)
        return entry

    def __iter__(self) -> Iterator[Section]:
        return iter(self.sections)

    def __len__(self) -> int:
        return len(self.sections)

    def get(self, section_id: str) -> Section:
        for entry in self.sections:
            if entry.id == section_id:
                return entry
        raise KeyError(section_id)

    @property
    def areas(self) -> list[str]:
        return list(dict.fromkeys(entry.area for entry in self.sections))


#: The SDK's sections, filled by :mod:`cyberwave.test.driver_testing.sections`.
REGISTRY = SectionRegistry()


def section(
    id: str,
    *,
    area: str,
    summary: str,
    commands: Iterable[str] = (),
    any_commands: Iterable[str] = (),
    topics: Iterable[str] = (),
    capabilities: Iterable[str] = (),
    needs_arm_joints: bool = False,
    needs_gripper_joint: bool = False,
    per_backend: bool = False,
    check: Callable[[TwinProfile], str | None] | None = None,
    covers: Iterable[str] | None = None,
    registry: SectionRegistry | None = None,
) -> Callable[[SectionBody], SectionBody]:
    """Register the decorated ``body(harness, **variant)`` as a section."""

    def decorate(body: SectionBody) -> SectionBody:
        (registry if registry is not None else REGISTRY).register(
            Section(
                id=id,
                area=area,
                summary=summary,
                body=body,
                commands=frozenset(commands),
                any_commands=frozenset(any_commands),
                topics=frozenset(topics),
                capabilities=frozenset(capabilities),
                needs_arm_joints=needs_arm_joints,
                needs_gripper_joint=needs_gripper_joint,
                per_backend=per_backend,
                check=check,
                covers=None if covers is None else frozenset(covers),
            )
        )
        return body

    return decorate


def _matches(token: str, entry: PlanEntry) -> bool:
    base = entry.section.id
    return (
        token in (entry.id, base, entry.section.area)
        or base.startswith(token + ".")
        or (entry.backend is not None and token == f"@{entry.backend}")
    )


def resolve_plan(
    profile: TwinProfile,
    *,
    sections: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    registry: SectionRegistry | None = None,
) -> list[PlanEntry]:
    """Every registered section against *profile*, in run order.

    *sections* / *skip* take section ids (``planned.ptp``), ids with a backend
    (``planned.ptp@moveit``), areas (``planned``) or a bare backend
    (``@moveit``). A section left out by them is reported skipped, with the
    flag as its reason. An unknown token raises ``ValueError``.
    """
    registry = registry if registry is not None else _default_registry()
    entries: list[PlanEntry] = []
    for entry in registry:
        reason = entry.skip_reason(profile)
        if entry.per_backend and profile.backends:
            for backend in profile.backends:
                backend_reason = reason
                if (
                    backend == profile.default_backend
                    and profile.motion_status is not None
                    and not profile.backend_available
                ):
                    backend_reason = f"default backend {backend} reports unavailable"
                entries.append(
                    PlanEntry(f"{entry.id}@{backend}", entry, backend, backend_reason)
                )
        else:
            entries.append(PlanEntry(entry.id, entry, None, reason))
    for token in (*(sections or ()), *skip):
        if not any(_matches(token, e) for e in entries) and not any(
            token in (s.id, s.area) or s.id.startswith(token + ".") for s in registry
        ):
            raise ValueError(
                f"Unknown section {token!r}; known areas: {', '.join(registry.areas)}"
            )
    resolved = []
    for e in entries:
        if e.runs and sections and not any(_matches(t, e) for t in sections):
            e = PlanEntry(
                e.id,
                e.section,
                e.backend,
                "not selected by --sections",
                deselected=True,
            )
        elif e.runs and any(_matches(t, e) for t in skip):
            e = PlanEntry(
                e.id, e.section, e.backend, "skipped by --skip", deselected=True
            )
        resolved.append(e)
    return resolved


@dataclass(frozen=True)
class CommandCoverage:
    """Whether one catalog command is exercised, and by which sections.

    ``status`` is ``tested`` (a section covering it runs, or ran), ``not
    selected`` (only ``--sections`` / ``--skip`` left it out) or ``untested``
    (no section covers it, or every one that does is skipped for this twin;
    ``reason`` says why).
    """

    command: str
    status: str
    sections: tuple[str, ...] = ()
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "sections": list(self.sections),
            "reason": self.reason,
        }


def command_coverage(
    profile: TwinProfile,
    plan: Sequence[PlanEntry],
    phases: Sequence[dict[str, Any]] | None = None,
) -> list[CommandCoverage]:
    """Every catalog command against the sections that exercise it.

    Before a run, a command is tested when a section covering it is planned.
    After a run, pass the harness's *phases*: a command is then tested only
    when a section covering it passed or failed. A section that skipped itself
    at run time (an unavailable backend, no measured TCP) tested nothing.
    """
    ran_outcome = {p.get("section"): p for p in phases or ()}
    rows = []
    for command in sorted(profile.commands):
        covering = [e for e in plan if command in e.section.covered_commands]

        def ran(entry: PlanEntry) -> bool:
            if not entry.runs:
                return False
            if phases is None:
                return True
            return ran_outcome.get(entry.id, {}).get("outcome") in ("passed", "failed")

        tested = [e for e in covering if ran(e)]
        if tested:
            rows.append(CommandCoverage(command, "tested", tuple(e.id for e in tested)))
        elif any(e.deselected for e in covering):
            deselected = tuple(e.id for e in covering if e.deselected)
            rows.append(CommandCoverage(command, "not selected", deselected))
        elif covering:
            reasons = [
                f"{e.id}: {e.skipped or ran_outcome.get(e.id, {}).get('reason') or 'did not run'}"
                for e in covering
            ]
            rows.append(CommandCoverage(command, "untested", (), "; ".join(reasons)))
        else:
            rows.append(
                CommandCoverage(command, "untested", (), "no section exercises it")
            )
    return rows


def _default_registry() -> SectionRegistry:
    from . import sections  # noqa: F401 - registers the SDK's sections

    return REGISTRY
