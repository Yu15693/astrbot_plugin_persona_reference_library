from __future__ import annotations


def normalize_text(value: str) -> str:
    """规范化通用文本：去除首尾空白。"""
    return value.strip()


def normalize_mime(value: str) -> str:
    """规范化 MIME：去除首尾空白并转小写。"""
    return normalize_text(value).lower()


def compact_whitespace(value: str) -> str:
    """移除字符串中的所有空白字符。"""
    return "".join(value.split())
