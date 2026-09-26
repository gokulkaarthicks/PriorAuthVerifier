"""Unit tests for runtime DISABLE_SAFETENSORS_CONVERSION bootstrap."""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _restore_config_module() -> None:
    """Reload config after each test so other suites see the default guard."""
    yield
    import app.core.config as config

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    importlib.reload(config)
    config.get_settings.cache_clear()


def test_disable_safetensors_conversion_set_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    assert "DISABLE_SAFETENSORS_CONVERSION" not in os.environ

    import app.core.config as config

    importlib.reload(config)

    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_disable_safetensors_conversion_preserves_existing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "0")

    import app.core.config as config

    importlib.reload(config)

    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "0"
