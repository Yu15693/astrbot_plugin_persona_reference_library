from __future__ import annotations

from typing import Any


def get_dict_value(data: Any, *keys: str) -> Any:
    """安全读取嵌套字典字段，路径不存在时返回 None。"""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current
