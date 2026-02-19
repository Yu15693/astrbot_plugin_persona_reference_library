from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .log import summarize_log_value


class PluginErrorCode(str, Enum):
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"


class PluginException(Exception):
    def __init__(
        self,
        code: PluginErrorCode,
        message: str,
        retryable: bool,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.detail = dict(detail) if detail else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        base = f"[{self.code.value}] {self.message} (retryable={self.retryable})"
        if not self.detail:
            return base
        detail_json = json.dumps(
            summarize_log_value(self.detail), ensure_ascii=False, default=str
        )
        return f"{base} detail={detail_json}"
