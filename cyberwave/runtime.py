from __future__ import annotations

import asyncio
from typing import Awaitable, Generic, Optional, TypeVar, Union

T = TypeVar("T")


def _loop_running() -> bool:
    """Return True when an asyncio event loop is currently running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    return loop.is_running()


async def _consume(awaitable: Awaitable[T]) -> T:
    return await awaitable


class CyberwaveTask(Generic[T]):
    """Wrapper that can be awaited or resolved synchronously.

    When a coroutine is produced while an event loop is already running, the
    SDK returns a :class:`CyberwaveTask` instead of forcing callers to touch
    :mod:`asyncio` directly.  The task can be ``await``-ed inside async code or
    resolved synchronously through :meth:`wait` / :meth:`result` once control is
    back on the main thread.
    """

    def __init__(self, awaitable: Awaitable[T]):
        self._awaitable = awaitable
        self._task: Optional[asyncio.Task[T]] = None
        self._result: Optional[T] = None
        self._exception: Optional[BaseException] = None
        self._consumed = False

    def _ensure_task(self) -> asyncio.Task[T]:
        if self._consumed:
            if self._task is not None:
                return self._task
            raise RuntimeError("CyberwaveTask already resolved")

        if self._task is None:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._execute())
        return self._task

    async def _execute(self) -> T:
        if self._consumed:
            raise RuntimeError("CyberwaveTask already resolved")
        self._consumed = True
        try:
            self._result = await self._awaitable
            return self._result
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            self._exception = exc
            raise

    def __await__(self):  # type: ignore[override]
        return self._ensure_task().__await__()

    def done(self) -> bool:
        task = self._task
        return bool(task and task.done())

    def wait(self) -> T:
        """Synchronously resolve the task when no event loop is running."""
        return self.result()

    def result(self) -> T:
        if self._result is not None:
            return self._result
        if self._exception is not None:
            raise self._exception
        if _loop_running():
            raise RuntimeError(
                "Cannot block for a CyberwaveTask while the asyncio loop is running;"
                " await it instead."
            )
        if self._consumed and self._task is not None:
            return self._task.result()
        self._consumed = True
        return asyncio.run(_consume(self._awaitable))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        state = "pending"
        if self._exception is not None:
            state = f"failed({self._exception!r})"
        elif self._result is not None:
            state = f"completed({self._result!r})"
        elif self.done():
            state = "completed"
        return f"CyberwaveTask({state})"


def run(awaitable: Awaitable[T]) -> Union[T, CyberwaveTask[T]]:
    """Resolve an awaitable, returning a value or a :class:`CyberwaveTask`."""
    if _loop_running():
        return CyberwaveTask(awaitable)
    return asyncio.run(_consume(awaitable))
