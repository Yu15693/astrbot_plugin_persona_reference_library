from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any


def summarize_log_value(value: Any) -> Any:
    """将复杂对象压缩为更适合日志输出的结构。"""
    if isinstance(value, dict):
        return {key: summarize_log_value(item) for key, item in value.items()}
    if isinstance(value, list):
        limit = 3
        items = [summarize_log_value(item) for item in value[:limit]]
        if len(value) > limit:
            items.append(f"<+{len(value) - limit} items>")
        return items
    if isinstance(value, str):
        if value.startswith("data:"):
            return f"<data-url len={len(value)}>"
        if len(value) > 200:
            return f"{value[:200]}...(truncated)"
    return value


@dataclass(slots=True)
class StructuredLogEmitter:
    """结构化日志输出器，默认会压缩复杂字段。"""

    logger: logging.Logger
    compress: bool = True

    def _emit(self, level: int, event: str, detail: dict[str, Any]) -> None:
        if not self.logger.isEnabledFor(level):
            return
        payload: Any = summarize_log_value(detail) if self.compress else detail
        message = {
            "event": event,
            "detail": payload,
        }
        self.logger.log(level, "%s", json.dumps(message, ensure_ascii=False, default=str))

    def debug(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.DEBUG, event, detail)

    def info(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.INFO, event, detail)

    def warning(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.WARNING, event, detail)

    def error(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.ERROR, event, detail)
