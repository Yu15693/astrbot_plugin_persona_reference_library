from __future__ import annotations


def is_http_url(value: str) -> bool:
    """判断是否为 http(s) URL。"""
    return value.startswith("http://") or value.startswith("https://")


def is_data_url(value: str) -> bool:
    """判断是否为 data URL。"""
    return value.startswith("data:")
