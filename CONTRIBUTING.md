# Contributing to the Cyberwave Python SDK

Thanks for your interest in contributing! This guide covers how to set up the
project, run the tests, and submit changes.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

The SDK requires **Python 3.10+** and uses [Poetry](https://python-poetry.org/) for
dependency management (the same toolchain CI runs).

```bash
git clone https://github.com/cyberwave-os/cyberwave-python.git
cd cyberwave-python

# Install Poetry if you don't have it: https://python-poetry.org/docs/#installation
poetry install            # installs the package + dev dependencies
```

Optional features live behind extras — install the ones you need, e.g.:

```bash
poetry install -E camera -E ml -E zenoh
```

## Running tests

```bash
poetry run pytest tests/ -v
poetry run python tests/test_imports.py
```

Please add or update tests for any behavior you change. Type checking uses `mypy`:

```bash
poetry run mypy cyberwave
```

## Submitting changes

1. Fork the repository and create a feature branch off `main`.
2. Make your change with focused commits and clear messages.
3. Ensure the test suite passes locally.
4. Open a pull request describing **what** changed and **why**. Link any related issue.

For larger changes, please open an
[issue](https://github.com/cyberwave-os/cyberwave-python/issues) first so we can
discuss the approach before you invest significant time.

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/cyberwave-os/cyberwave-python/issues). Include
the SDK version (`python -c "import cyberwave; print(cyberwave.__version__)"`), your
Python version and OS, and a minimal reproduction where possible.

## Questions

Join the community on [Discord](https://discord.gg/dfGhNrawyF) or browse the docs at
[docs.cyberwave.com](https://docs.cyberwave.com).
