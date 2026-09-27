"""Optional local debug-trace dumps (DEBUG_TRACE_ENABLED=true)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TRACE_DIR = Path("debug_trace")


def is_enabled() -> bool:
    return bool(get_settings().DEBUG_TRACE_ENABLED)


def write_json(name: str, payload: Any, *, analysis_id: str | None = None) -> None:
    """Write a JSON artifact when debug tracing is enabled.

    Never called for normal request logs. May contain prompts/context locally.
    """
    if not is_enabled():
        return
    try:
        directory = _TRACE_DIR
        if analysis_id:
            directory = directory / analysis_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(
            json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("debug_trace_written", name=name, analysis_id=analysis_id)
    except Exception as exc:  # noqa: BLE001  # never break analysis for traces
        logger.warning("debug_trace_write_failed", name=name, error=str(exc))
