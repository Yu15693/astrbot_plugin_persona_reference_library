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
        limit = 5
        items = [summarize_log_value(item) for item in value[:limit]]
        if len(value) > limit:
            items.append(f"<+{len(value) - limit} items>")
        return items
    if isinstance(value, str):
        if value.startswith("data:"):
            return f"<data-url len={len(value)}>"
        if len(value) > 400:
            return f"{value[:400]}...(truncated)"
    return value


def resolve_runtime_logger(name: str | None) -> logging.Logger:
    """返回运行时 logger：有宿主时优先使用宿主 logger，否则回退标准 logging。"""
    try:
        from astrbot.api import logger as host_logger

        if isinstance(host_logger, logging.Logger):
            return host_logger
    except Exception:
        pass

    return logging.getLogger(name)


def get_structured_logger(
    name: str | None = None,
    *,
    compress: bool = True,
) -> StructuredLogEmitter:
    """创建结构化日志输出器。"""
    return StructuredLogEmitter(
        logger=resolve_runtime_logger(name),
        compress=compress,
    )


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

        self.logger.log(
            level,
            "%s",
            json.dumps(message, ensure_ascii=False, default=str),
            stacklevel=3,
        )

    def debug(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.DEBUG, event, detail)

    def info(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.INFO, event, detail)

    def warning(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.WARNING, event, detail)

    def error(self, event: str, detail: dict[str, Any]) -> None:
        self._emit(logging.ERROR, event, detail)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 3)
        self.logger.exception(message, *args, **kwargs)


logger = get_structured_logger()
