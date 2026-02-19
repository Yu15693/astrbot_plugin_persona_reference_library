from __future__ import annotations

from uuid6 import uuid7


def generate_id() -> str:
    """生成按时间有序的唯一 ID。"""
    return str(uuid7())
