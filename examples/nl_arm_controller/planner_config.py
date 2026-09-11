"""Shared LLM planner configuration."""

from __future__ import annotations

import os


DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.4-mini"


def get_openrouter_model() -> str:
    """Return the OpenRouter model selected by the environment or its default."""
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)