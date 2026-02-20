from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlparse

import aiohttp

from ..utils.io import CONTENT_TYPE_SUFFIX_MAP
from .codec import decode_base64_payload
from .io import infer_image_mime
from .normalize import normalize_mime, normalize_text


def _build_image_suffix_to_mime() -> dict[str, str]:
    """从 CONTENT_TYPE_SUFFIX_MAP 中筛选 image 相关 type。"""
    suffix_to_mime: dict[str, str] = {}
    for mime, suffix in CONTENT_TYPE_SUFFIX_MAP.items():
        # 仅筛选 image/*；若多个 MIME 对应同一后缀，保留先出现的规范项。
        if not mime.startswith("image/"):
            continue
        if suffix in suffix_to_mime:
            continue
        suffix_to_mime[suffix] = mime
    return suffix_to_mime


_IMAGE_SUFFIX_TO_MIME = _build_image_suffix_to_mime()
IMAGE_SUFFIXES = set(_IMAGE_SUFFIX_TO_MIME.keys())


def infer_http_url_mime(url: str, default_mime: str = "") -> str:
    """根据 HTTP URL 的扩展名推断 MIME。"""
    parsed = urlparse(normalize_text(url))
    suffix = PurePosixPath(parsed.path).suffix.lower()
    return _IMAGE_SUFFIX_TO_MIME.get(suffix, default_mime)


def infer_base64_mime(base64_payload: str, default_mime: str = "image/png") -> str:
    """根据 base64 内容推断 MIME。"""
    try:
        image_bytes = decode_base64_payload(base64_payload)
    except ValueError:
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
