from __future__ import annotations

import base64
import binascii

from ..utils.io import data_url_to_base64
from .normalize import compact_whitespace, normalize_mime, normalize_text


def normalize_base64_payload(value: str) -> str:
    """规范化并校验 base64 负载，返回纯 base64 字符串。"""
    normalized = normalize_text(value)
    if normalized.startswith("base64://"):
        normalized = normalized.removeprefix("base64://")
    normalized = compact_whitespace(normalized)
    if not normalized:
        raise ValueError("base64 payload is empty.")
    try:
        raw_bytes = base64.b64decode(normalized, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("base64 payload is invalid.") from exc
    return base64.b64encode(raw_bytes).decode("ascii")


def data_url_payload_to_base64(data_url: str) -> str:
    """将 data URL 转换为规范化 base64 负载。"""
    return data_url_to_base64(data_url)


def build_data_url(mime: str, base64_payload: str) -> str:
    """根据 MIME 与 base64 负载组装 data URL。"""
    normalized_mime = normalize_mime(mime)
    if not normalized_mime:
        raise ValueError("mime is required to build data URL.")
    return f"data:{normalized_mime};base64,{base64_payload}"
