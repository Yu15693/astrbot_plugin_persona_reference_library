from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote_to_bytes

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


def data_url_to_base64(data_url: str) -> str:
    """将 data URL 转换为纯 base64 字符串。"""
    normalized = data_url.strip()
    if not normalized.startswith("data:"):
        raise ValueError("data_url must start with 'data:'.")
    if "," not in normalized:
        raise ValueError("data_url must contain ',' separator.")

    header, payload = normalized.split(",", 1)
    if ";base64" in header.lower():
        payload = "".join(payload.split())
        try:
            raw_bytes = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("data_url contains invalid base64 payload.") from exc
        return base64.b64encode(raw_bytes).decode("ascii")

    raw_bytes = unquote_to_bytes(payload)
    return base64.b64encode(raw_bytes).decode("ascii")


def infer_image_mime(image_bytes: bytes, default_mime: str = "image/png") -> str:
    """根据图片字节头推断 MIME，无法识别时回退默认值。"""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith(b"II*\x00") or image_bytes.startswith(b"MM\x00*"):
        return "image/tiff"

    try:
        from PIL import Image, UnidentifiedImageError
    except Exception:
        return default_mime

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            format_name = (image.format or "").upper()
            mime = Image.MIME.get(format_name)
            if isinstance(mime, str) and mime:
                return mime
    except (UnidentifiedImageError, OSError, ValueError):
        return default_mime

    return default_mime


def base64_image_to_data_url(base64_value: str, default_mime: str = "image/png") -> str:
    """将 base64 图片字符串转换为 data URL。"""
    normalized = base64_value.strip()
    if not normalized:
        raise ValueError("base64_value is empty.")
    if normalized.startswith("data:"):
        return normalized
    if normalized.startswith("base64://"):
        normalized = normalized.removeprefix("base64://")

    # 去掉可能存在的空白，兼容跨行/带空格的输入。
    normalized = "".join(normalized.split())
    try:
        image_bytes = base64.b64decode(normalized, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("base64_value is not valid base64 content.") from exc

    mime = infer_image_mime(image_bytes, default_mime=default_mime)
    return f"data:{mime};base64,{normalized}"


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


def compress_image_bytes_to_jpg(
    image_bytes: bytes,
    quality: int = 85,
) -> bytes:
    """将图像二进制压缩并转换为 JPG（二进制）。"""
    # 采用实用的 JPEG 质量范围，避免出现极端体积。
    if quality < 1 or quality > 95:
        raise ValueError("quality must be in [1, 95].")

    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        # JPEG 不支持透明通道，这里将透明像素合成到白色背景。
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            alpha = image.convert("RGBA")
            background = Image.new("RGB", alpha.size, (255, 255, 255))
            background.paste(alpha, mask=alpha.split()[-1])
            rgb_image = background
        else:
            rgb_image = image.convert("RGB")

        output = BytesIO()
        # 启用 optimize，在保持可接受画质的同时尽量减小体积。
        rgb_image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()
