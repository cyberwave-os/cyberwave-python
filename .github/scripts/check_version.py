"""Verify that the SDK version is declared consistently.

Checks that ``pyproject.toml`` and ``cyberwave/_version.py`` agree and, when
``--tag`` is given, that the release tag (``vX.Y.Z``) matches them too.

Usage:
    python .github/scripts/check_version.py
    python .github/scripts/check_version.py --tag v0.7.2
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Regex rather than tomllib so this runs on Python 3.10. The section pattern
# stops at the next line-leading table header, so arrays like ``authors = [``
# inside [tool.poetry] don't end it early.
POETRY_SECTION_PATTERN = re.compile(r"^\[tool\.poetry\]\s*$(.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
STATIC_PATTERN = re.compile(r'^STATIC_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)


def _search(text: str, pattern: re.Pattern[str], source: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"Could not find a version in {source}")
    return match.group(1)


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = _search(text, POETRY_SECTION_PATTERN, "pyproject.toml [tool.poetry]")
    return _search(section, VERSION_PATTERN, "pyproject.toml [tool.poetry]")


def _static_version() -> str:
    text = (ROOT / "cyberwave" / "_version.py").read_text(encoding="utf-8")
    return _search(text, STATIC_PATTERN, "cyberwave/_version.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag to check, e.g. v0.7.2")
    args = parser.parse_args()

    versions = {
        "pyproject.toml": _pyproject_version(),
        "cyberwave/_version.py": _static_version(),
    }
    if args.tag:
        versions[f"tag {args.tag}"] = args.tag.removeprefix("v")

    if len(set(versions.values())) != 1:
        print("Version mismatch:", file=sys.stderr)
        for source, version in versions.items():
            print(f"  {source}: {version}", file=sys.stderr)
        return 1

    print(f"Version {versions['pyproject.toml']} is consistent across {', '.join(versions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
