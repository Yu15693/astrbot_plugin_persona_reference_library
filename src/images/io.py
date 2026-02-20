from __future__ import annotations

import base64
from io import BytesIO

from ..utils.url import is_data_url, is_http_url
from .codec import build_data_url, data_url_to_bytes, decode_base64_payload
from .normalize import normalize_text


def infer_image_mime(image_bytes: bytes, default_mime: str = "image/png") -> str:
    """
    根据图片字节推断 MIME。

    先使用常见格式的魔数快速识别；若未命中，再交给 Pillow 实际解析图片。
    因此只要能识别成功，返回值通常就是图片的真实 MIME；
    仅在无法识别或解析失败时回退 `default_mime`，不会抛出异常。
    """
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
    """
    将图片输入统一转换为 data URL。

    支持输入：
    - 纯 base64（可带 `base64://` 前缀）
    - 已是 data URL（会先校验必须是合法 base64 data URL，再原样返回）

    不支持输入：
    - http(s) URL（会直接抛出 ValueError）

    对 base64 输入会先做合法性校验，再推断 MIME 并组装为规范 data URL。
    """
    normalized = normalize_text(base64_value)
    if not normalized:
        raise ValueError("base64_value is empty.")
    if is_data_url(normalized):
        try:
            data_url_to_bytes(normalized)
        except ValueError as exc:
            raise ValueError("base64_value is not valid base64 content.") from exc
        return normalized
    if is_http_url(normalized):
        raise ValueError("http_url can not be converted to data URL directly.")

    try:
        image_bytes = decode_base64_payload(normalized)
    except ValueError as exc:
        raise ValueError("base64_value is not valid base64 content.") from exc
    normalized_base64 = base64.b64encode(image_bytes).decode("ascii")
    mime = infer_image_mime(image_bytes, default_mime=default_mime)
    return build_data_url(mime, normalized_base64)


def compress_image_bytes_to_jpg(
    image_bytes: bytes,
    quality: int = 85,
) -> bytes:
    """
    将图片字节压缩并转换为 JPEG 字节。

    - `quality` 取值范围为 `[1, 95]`，越界会抛出 ValueError。
    - 自动处理透明通道：对带 alpha 的图片先合成白底再转 JPEG。
    """
    if quality < 1 or quality > 95:
        raise ValueError("quality must be in [1, 95].")

    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
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
        rgb_image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()
