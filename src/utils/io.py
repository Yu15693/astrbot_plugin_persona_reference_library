from __future__ import annotations

from pathlib import Path


def save_file(
    path: Path,
    content: bytes | bytearray | memoryview | str,
    encoding: str = "utf-8",
) -> Path:
    """将内容保存到指定路径，自动创建父目录并返回目标路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding=encoding)
    else:
        path.write_bytes(bytes(content))
    return path
