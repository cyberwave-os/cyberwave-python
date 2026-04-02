"""Tests for the worker module loader."""

import builtins
import sys

import pytest

from cyberwave.workers.loader import load_workers


@pytest.fixture(autouse=True)
def _cleanup_builtins():
    """Remove ``builtins.cw`` after each test."""
    yield
    if hasattr(builtins, "cw"):
        delattr(builtins, "cw")
    # Clean worker modules from sys.modules
    for key in list(sys.modules):
        if key.startswith("cyberwave_worker_"):
            del sys.modules[key]


def test_load_workers_finds_py_files(tmp_path):
    (tmp_path / "worker_a.py").write_text("LOADED_A = True\n")
    (tmp_path / "worker_b.py").write_text("LOADED_B = True\n")

    count = load_workers(tmp_path, cw_instance="fake_cw")
    assert count == 2


def test_load_workers_skips_underscore(tmp_path):
    (tmp_path / "_helper.py").write_text("HELPER = True\n")
    (tmp_path / "worker.py").write_text("W = 1\n")

    count = load_workers(tmp_path, cw_instance="fake_cw")
    assert count == 1
    assert "cyberwave_worker__helper" not in sys.modules
    assert "cyberwave_worker_worker" in sys.modules


def test_load_workers_bad_file_continues(tmp_path):
    (tmp_path / "good.py").write_text("G = 1\n")
    (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n")
    (tmp_path / "also_good.py").write_text("AG = 2\n")

    count = load_workers(tmp_path, cw_instance="fake_cw")
    assert count == 2  # good and also_good loaded, bad skipped


def test_load_workers_missing_dir(tmp_path):
    nonexistent = tmp_path / "missing"
    count = load_workers(nonexistent, cw_instance="fake_cw")
    assert count == 0


def test_cw_injected_as_builtin(tmp_path):
    sentinel = object()
    (tmp_path / "check_cw.py").write_text(
        "import builtins; assert hasattr(builtins, 'cw')\n"
    )

    count = load_workers(tmp_path, cw_instance=sentinel)
    assert count == 1
    assert builtins.cw is sentinel  # type: ignore[attr-defined]


def test_load_workers_alphabetical_order(tmp_path):
    """Worker modules are loaded in alphabetical order."""
    (tmp_path / "z_last.py").write_text(
        "import sys; sys.modules[__name__]._ORDER = 'z'\n"
    )
    (tmp_path / "a_first.py").write_text(
        "import sys; sys.modules[__name__]._ORDER = 'a'\n"
    )
    (tmp_path / "m_middle.py").write_text(
        "import sys; sys.modules[__name__]._ORDER = 'm'\n"
    )

    load_workers(tmp_path, cw_instance="fake_cw")

    for name in [
        "cyberwave_worker_a_first",
        "cyberwave_worker_m_middle",
        "cyberwave_worker_z_last",
    ]:
        assert name in sys.modules


def test_load_workers_cw_usable_in_module(tmp_path):
    """Worker code can use the bare ``cw`` variable."""
    (tmp_path / "use_cw.py").write_text("val = cw.upper()\n")

    count = load_workers(tmp_path, cw_instance="hello")
    assert count == 1
    mod = sys.modules["cyberwave_worker_use_cw"]
    assert mod.val == "HELLO"
