"""Test helper: drive a ``python-statemachine`` class through every transition.

New state machines get a test that calls :func:`assert_every_transition_runs`,
so each declared edge is exercised at least once. Together with the library's
own checks when the class is defined (every state reachable, no dead ends),
this covers the whole machine.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

from statemachine import StateMachine


def _edges(machine_cls: type[StateMachine]) -> set[tuple[str, str, str]]:
    return {
        (transition.source.id, event, transition.target.id)
        for state in machine_cls.states
        for transition in state.transitions
        for event in transition.events
    }


def _path_to(
    machine_cls: type[StateMachine], start: str, goal: str
) -> list[str] | None:
    """Shortest list of events leading from *start* to *goal*."""
    edges = _edges(machine_cls)
    queue: deque[tuple[str, list[str]]] = deque([(start, [])])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for source, event, target in sorted(edges):
            if source == state and target not in seen:
                seen.add(target)
                queue.append((target, [*path, event]))
    return None


def assert_every_transition_runs(factory: Callable[[], StateMachine]) -> None:
    """Fire every declared transition from a fresh machine built by *factory*.

    For each edge, the helper reaches its source state by the shortest path,
    sends the event, and checks the machine lands on the edge's target.
    """
    machine_cls = type(factory())
    initial = machine_cls.initial_state.id
    for source, event, target in sorted(_edges(machine_cls)):
        path = _path_to(machine_cls, initial, source)
        assert path is not None, f"state {source!r} is unreachable"
        machine = factory()
        for step in path:
            machine.send(step)
        machine.send(event)
        current = {state.id for state in machine.configuration}
        assert current == {target}, (
            f"{event!r} from {source!r} reached {sorted(current)}, expected {target!r}"
        )
