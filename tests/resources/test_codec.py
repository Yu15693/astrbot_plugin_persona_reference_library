from __future__ import annotations

import pytest
from src.resources.codec import (
    decode_base64_payload,
    parse_data_url_header,
    transfer_data_url_to_base64,
    transfer_data_url_to_bytes,
)


def test_parse_data_url_header_base64() -> None:
    """验证：可正确解析带 base64 标记的 data URL 头信息。"""
    header = parse_data_url_header("data:image/png;charset=utf-8;base64,Zm9v")

    assert header.mime == "image/png"
    assert header.is_base64 is True
    assert header.payload == "Zm9v"


def test_transfer_data_url_to_bytes_and_base64_for_plain_text() -> None:
    """验证：非 base64 的 data URL 可正确解码为 bytes 并再编码为 base64。"""
    data_url = "data:text/plain,hello%20world"
    data, mime = transfer_data_url_to_bytes(data_url)
    encoded = transfer_data_url_to_base64(data_url)

    assert data == b"hello world"
    assert mime == "text/plain"
    assert encoded == "aGVsbG8gd29ybGQ="


def test_decode_base64_payload_raises_for_invalid_input() -> None:
    """验证：非法 base64 输入会抛出 ValueError。"""
    with pytest.raises(ValueError, match="base64 payload is invalid"):
        decode_base64_payload("%%%")
