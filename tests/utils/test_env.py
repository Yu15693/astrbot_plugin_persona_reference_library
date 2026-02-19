from __future__ import annotations

import os

import pytest

from src.utils.errors import PluginException


def is_env_enabled(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"{name} is required for live integration test.")
    return value


def is_retryable_plugin_exception(exc: BaseException) -> bool:
    return isinstance(exc, PluginException) and exc.retryable
