from __future__ import annotations

import base64
import binascii
from pathlib import Path

import aiohttp

CONTENT_TYPE_SUFFIX_MAP: dict[str, str] = {
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/xml": ".xml",
    "application/zip": ".zip",
    "application/octet-stream": ".bin",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/plain": ".txt",
    "text/xml": ".xml",
}


def _normalize_content_type(content_type: str) -> str:
    """规范化 content-type，去掉参数并转为小写。"""
    return content_type.split(";", 1)[0].strip().lower()


def suffix_from_content_type(content_type: str) -> str:
    """根据 content-type 推导文件后缀，未知类型回退为 .bin。"""
    normalized = _normalize_content_type(content_type)
    if not normalized:
        return ".bin"

    if normalized in CONTENT_TYPE_SUFFIX_MAP:
        return CONTENT_TYPE_SUFFIX_MAP[normalized]
    if normalized.startswith("image/"):
        return f".{normalized.split('/', 1)[1]}"
    if normalized.startswith("text/"):
        return ".txt"
    return ".bin"


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    """解析 data URL，返回二进制内容与推导出的后缀。"""
    header, encoded = data_url.split(",", 1)
    content_type = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    suffix = suffix_from_content_type(content_type)
    try:
        return base64.b64decode(encoded, validate=True), suffix
    except (ValueError, binascii.Error):
        return encoded.encode("utf-8"), ".txt"


async def download_http_resource(url: str, timeout_sec: int = 60) -> tuple[bytes, str]:
    """异步下载 HTTP 资源并返回内容与响应头推导出的后缀。"""
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            content = await response.read()
            suffix = suffix_from_content_type(response.headers.get("Content-Type", ""))
    return content, suffix


def save_file(path: Path, content: bytes | str, encoding: str = "utf-8") -> Path:
    """将内容保存到指定路径，自动创建父目录并返回目标路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding=encoding)
    return path
