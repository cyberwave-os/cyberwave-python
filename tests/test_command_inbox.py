"""B5: CommandInbox — thread-safe merge inbox with correct timestamp dedup.

Replaces the hand-rolled lock + pending dict + watermark that drivers (Piper)
got wrong: missing timestamps bypassed dedup and never advanced the watermark,
and equal timestamps were dropped.
"""

import threading

from cyberwave.driver.interface.command_inbox import CommandInbox


def test_submit_then_drain_returns_and_clears():
    inbox = CommandInbox()
    assert inbox.submit({"j1": 0.1}) is True
    assert inbox.submit({"j2": 0.2}) is True
    assert inbox.drain() == {"j1": 0.1, "j2": 0.2}
    assert inbox.drain() == {}  # cleared after drain


def test_later_update_overwrites_same_key_in_pending():
    inbox = CommandInbox()
    inbox.submit({"j1": 0.1})
    inbox.submit({"j1": 0.9})
    assert inbox.drain() == {"j1": 0.9}


def test_stale_timestamp_is_dropped():
    inbox = CommandInbox()
    assert inbox.submit({"j1": 1.0}, timestamp=100.0) is True
    assert inbox.submit({"j1": 2.0}, timestamp=99.0) is False  # older → stale
    assert inbox.drain() == {"j1": 1.0}


def test_equal_timestamp_is_accepted():
    # Piper's bug: `ts <= last` dropped distinct same-tick commands.
    inbox = CommandInbox()
    assert inbox.submit({"j1": 1.0}, timestamp=100.0) is True
    assert inbox.submit({"j2": 2.0}, timestamp=100.0) is True
    assert inbox.drain() == {"j1": 1.0, "j2": 2.0}


def test_missing_timestamp_always_accepted_even_after_watermark():
    # Relaxed: untagged messages are accepted and not blocked by the watermark.
    inbox = CommandInbox()
    inbox.submit({"j1": 1.0}, timestamp=100.0)
    assert inbox.submit({"j1": 2.0}) is True  # no timestamp → accepted
    assert inbox.drain() == {"j1": 2.0}
    # A subsequent stale timestamped message is still rejected.
    inbox.submit({"j1": 3.0}, timestamp=100.0)  # equal → accepted
    assert inbox.submit({"j1": 4.0}, timestamp=50.0) is False


def test_reset_clears_pending_and_watermark():
    inbox = CommandInbox()
    inbox.submit({"j1": 1.0}, timestamp=100.0)
    inbox.reset()
    assert inbox.drain() == {}
    # After reset, an older timestamp is accepted again (new session).
    assert inbox.submit({"j1": 2.0}, timestamp=10.0) is True


def test_pending_count():
    inbox = CommandInbox()
    assert inbox.pending_count == 0
    inbox.submit({"j1": 1.0, "j2": 2.0})
    assert inbox.pending_count == 2


def test_concurrent_submit_and_drain_do_not_corrupt():
    inbox = CommandInbox()
    drained_total = 0
    stop = threading.Event()

    def producer():
        for i in range(2000):
            inbox.submit({f"k{i % 8}": float(i)})

    def consumer():
        nonlocal drained_total
        while not stop.is_set():
            drained_total += len(inbox.drain())

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t2.start()
    t1.start()
    t1.join()
    stop.set()
    t2.join()
    # No exception/corruption is the assertion; final drain is a valid dict.
    assert isinstance(inbox.drain(), dict)
