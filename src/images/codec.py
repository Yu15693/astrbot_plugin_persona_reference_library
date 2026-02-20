from __future__ import annotations

import base64
import binascii
from typing import Literal

from ..utils.url import is_data_url, is_http_url
from .normalize import compact_whitespace, normalize_mime, normalize_text

ParsedImageValueKind = Literal["http_url", "data_url", "base64"]


def decode_base64_payload(value: str) -> bytes:
    """规范化并校验 base64 负载，返回原始字节。"""
    normalized = normalize_text(value)
    if normalized.startswith("base64://"):
        normalized = normalized.removeprefix("base64://")
    normalized = compact_whitespace(normalized)
    if not normalized:
        raise ValueError("base64 payload is empty.")
    try:
        return base64.b64decode(normalized, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("base64 payload is invalid.") from exc


def normalize_base64_payload(value: str) -> str:
    """规范化并校验 base64 负载，返回纯 base64 字符串。"""
    return base64.b64encode(decode_base64_payload(value)).decode("ascii")


def _split_data_url(data_url: str) -> tuple[str, str]:
    """拆分 data URL，返回 `(header, payload)`。"""
    normalized = normalize_text(data_url)
    if not normalized.startswith("data:"):
        raise ValueError("data_url must start with 'data:'.")
    if "," not in normalized:
        raise ValueError("data_url must contain ',' separator.")
    return normalized.split(",", 1)


def data_url_to_base64(data_url: str) -> str:
    """将 base64 data URL 转换为规范化 base64 负载。"""
    return base64.b64encode(data_url_to_bytes(data_url)).decode("ascii")


def data_url_to_bytes(data_url: str) -> bytes:
    """将 base64 data URL 转换为原始字节。"""
    header, payload = _split_data_url(data_url)
    if ";base64" not in header.lower():
        raise ValueError("data_url must contain ';base64'.")
    return decode_base64_payload(payload)


def extract_data_url_mime(data_url: str, default_mime: str = "") -> str:
    """从 data URL 头部提取 MIME。"""
    try:
        header, _ = _split_data_url(data_url)
    except ValueError:
        return default_mime
    mime = normalize_mime(header[5:].split(";", 1)[0])
    return mime or default_mime


def build_data_url(mime: str, base64_payload: str) -> str:
    """根据 MIME 与 base64 负载组装 data URL。"""
    normalized_mime = normalize_mime(mime)
    if not normalized_mime:
        raise ValueError("mime is required to build data URL.")
    return f"data:{normalized_mime};base64,{base64_payload}"


def parse_raw_image_value(raw: str) -> tuple[ParsedImageValueKind, str]:
    """
    统一解析原始图片字符串并返回 `(kind, value)`。

    规则：
    - `http(s) URL` => `("http_url", normalized_url)`
    - `data URL` => `("data_url", normalized_data_url)`
    - 其他输入按 base64 处理并校验 => `("base64", normalized_base64)`
    """
    normalized = normalize_text(raw)
    if not normalized:
        raise ValueError("raw image value must not be empty.")
    if is_http_url(normalized):
        return "http_url", normalized
    if is_data_url(normalized):
        return "data_url", normalized
    return "base64", normalize_base64_payload(normalized)
