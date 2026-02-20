from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import download_http_resource, save_file, suffix_from_content_type
from .codec import data_url_to_bytes
from .io import compress_image_bytes_to_jpg, infer_image_mime
from .mime import IMAGE_SUFFIXES
from .plugin_image import PluginImage


@dataclass(slots=True)
class SavePluginImageResult:
    image: PluginImage
    path: Path
    compressed: bool = False

    def to_metadata_dict(self) -> dict[str, object]:
        """转换为可安全写入 JSON 的摘要信息。"""
        return {
            "kind": self.image.kind,
            "mime": self.image.mime,
            "filename": self.path.name,
            "compressed": self.compressed,
        }


async def _load_image_content(
    image: PluginImage,
    *,
    http_timeout_sec: int,
) -> bytes:
    """加载图片原始字节。"""
    if image.kind == "http_url":
        content, _ = await download_http_resource(
            image.value,
            timeout_sec=http_timeout_sec,
        )
        return content
    if image.kind == "data_url":
        return data_url_to_bytes(image.value)
    if image.kind == "base64":
        return base64.b64decode(image.value, validate=True)

    raise ValueError(f"unsupported image kind: {image.kind}")


def _resolve_output_content(
    content: bytes,
    *,
    enable_compress: bool,
    jpeg_quality: int,
) -> tuple[bytes, str, bool]:
    """解析输出内容、后缀与压缩标记。"""
    resolved_mime = infer_image_mime(content, default_mime="")
    if not resolved_mime.startswith("image/"):
        raise ValueError("image mime type is not valid.")

    suffix = suffix_from_content_type(resolved_mime)
    if enable_compress and suffix in IMAGE_SUFFIXES:
        compressed = compress_image_bytes_to_jpg(content, quality=jpeg_quality)
        return compressed, ".jpg", True

    return content, suffix, False


async def save_plugin_image(
    *,
    image: PluginImage,
    target_dir: Path,
    filename_stem: str,
    enable_compress: bool = False,
    jpeg_quality: int = 85,
    http_timeout_sec: int = 60,
) -> SavePluginImageResult:
    """保存单个 PluginImage 到目标目录，失败时抛异常。"""

    # 加载图片
    content = await _load_image_content(
        image,
        http_timeout_sec=http_timeout_sec,
    )
    # 获取后缀，可选压缩
    content, suffix, compressed = _resolve_output_content(
        content,
        enable_compress=enable_compress,
        jpeg_quality=jpeg_quality,
    )

    output_path = target_dir / f"{filename_stem}{suffix}"
    save_file(output_path, content)
    return SavePluginImageResult(
        image=image,
        path=output_path,
        compressed=compressed,
    )
