from __future__ import annotations

import base64
import binascii
from typing import NamedTuple
from urllib.parse import unquote_to_bytes

from .normalize import (
    normalize_base64_payload,
    normalize_mime,
)


def decode_base64_payload(value: str) -> bytes:
    """base64 => bytes 同时校验 value 是否有效"""
    normalized = normalize_base64_payload(value)
    try:
        return base64.b64decode(normalized, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("base64 payload is invalid.") from exc


def encode_base64_payload(data: bytes) -> str:
    """bytes => base64"""
    return base64.b64encode(data).decode("ascii")


class DataUrlHeader(NamedTuple):
    mime: str
    is_base64: bool
    payload: str


def parse_data_url_header(data_url: str) -> DataUrlHeader:
    """解析 data url 头部"""
    normalized_data_url = data_url.strip()
    if not normalized_data_url.startswith("data:"):
        raise ValueError("data_url must start with 'data:'.")

    # 按第一个逗号切分：data:[meta],[payload]
    # meta: image/png; charset=utf-8; base64
    header_and_data = normalized_data_url.removeprefix("data:")
    try:
        meta, payload = header_and_data.split(",", 1)
    except ValueError as exc:
        raise ValueError("data_url is invalid.") from exc

    # 将 meta 按分号拆分为片段，去除首尾空白并过滤空片段
    tokens = [segment.strip() for segment in meta.split(";") if segment.strip()]
    is_base64 = any(token.lower() == "base64" for token in tokens)

    # 约定 MIME 在第一段，例如 image/png；若缺失则返回空字符串。
    mime = ""
    if tokens:
        first = tokens[0]
        if "/" in first and "=" not in first and first.lower() != "base64":
            mime = normalize_mime(first)

    return DataUrlHeader(mime=mime, is_base64=is_base64, payload=payload)


def transfer_data_url_to_bytes(data_url: str) -> tuple[bytes, str]:
    header = parse_data_url_header(data_url)
    if header.is_base64:
        content = decode_base64_payload(header.payload)
    else:
        content = unquote_to_bytes(header.payload)
    return content, header.mime


def transfer_data_url_to_base64(data_url: str) -> str:
    header = parse_data_url_header(data_url)
    if header.is_base64:
        return normalize_base64_payload(header.payload)
    # 非 base64 编码，需重新解码和编码
    data, _ = transfer_data_url_to_bytes(data_url)
    return encode_base64_payload(data)


def transfer_base64_to_data_url(mime: str, base64_payload: str) -> str:
    normalized_mime = normalize_mime(mime)
    if not normalized_mime:
        raise ValueError("mime is required to build data URL.")
    normalized_base64 = normalize_base64_payload(base64_payload)
    return f"data:{normalized_mime};base64,{normalized_base64}"
