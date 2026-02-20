from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlparse

import aiohttp

from ..utils.io import infer_image_mime
from .normalize import compact_whitespace, normalize_mime, normalize_text

_SUFFIX_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def infer_http_url_mime(url: str, default_mime: str = "") -> str:
    """根据 HTTP URL 的扩展名推断 MIME。"""
    parsed = urlparse(normalize_text(url))
    suffix = PurePosixPath(parsed.path).suffix.lower()
    return _SUFFIX_TO_MIME.get(suffix, default_mime)


def extract_data_url_mime(data_url: str, default_mime: str = "") -> str:
    """从 data URL 头部提取 MIME。"""
    normalized = normalize_text(data_url)
    if not normalized.startswith("data:"):
        return default_mime
    header = normalized.split(",", 1)[0]
    mime = normalize_mime(header[5:].split(";", 1)[0])
    return mime or default_mime


def infer_base64_mime(base64_payload: str, default_mime: str = "image/png") -> str:
    """根据 base64 内容推断 MIME。"""
    import base64
    import binascii

    normalized = compact_whitespace(normalize_text(base64_payload))
    try:
        image_bytes = base64.b64decode(normalized, validate=True)
    except (ValueError, binascii.Error):
        return default_mime
    return infer_image_mime(image_bytes, default_mime=default_mime)


async def infer_http_mime_with_head(
    url: str,
    *,
    timeout_sec: int = 5,
    default_mime: str = "",
) -> str:
    """通过 HTTP HEAD 获取 Content-Type 并推断 MIME。"""
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url) as response:
                content_type = response.headers.get("Content-Type", "")
    except (aiohttp.ClientError, TimeoutError):
        return default_mime

    mime = normalize_mime(content_type.split(";", 1)[0])
    if mime.startswith("image/"):
        return mime
    return default_mime
