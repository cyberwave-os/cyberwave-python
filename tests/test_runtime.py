from __future__ import annotations

import asyncio

from cyberwave.runtime import CyberwaveTask, run


async def _sample_value() -> int:
    await asyncio.sleep(0)
    return 42


def test_run_returns_value_in_sync_context():
    assert run(_sample_value()) == 42


def test_run_returns_task_in_async_context():
    async def main() -> int:
        task = run(_sample_value())
        assert isinstance(task, CyberwaveTask)
        return await task

    assert asyncio.run(main()) == 42


def test_task_wait_resolves_after_async_context():
    async def main() -> CyberwaveTask[int]:
        task = run(_sample_value())
        assert isinstance(task, CyberwaveTask)
        return task

    task = asyncio.run(main())
    assert isinstance(task, CyberwaveTask)
    assert task.wait() == 42


def test_task_result_raises_when_loop_active():
    async def main() -> bool:
        task = run(_sample_value())
        assert isinstance(task, CyberwaveTask)
        try:
            task.result()
        except RuntimeError:
            await task
            return True
        return False

    assert asyncio.run(main()) is True
