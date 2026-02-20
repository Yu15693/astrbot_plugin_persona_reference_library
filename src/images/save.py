from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import (
    compress_image_bytes_to_jpg,
    download_http_resource,
    infer_image_mime,
    save_file,
    suffix_from_content_type,
)
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


def _maybe_compress_image_bytes(
    content: bytes,
    suffix: str,
    *,
    enable_compress: bool,
    jpeg_quality: int,
) -> tuple[bytes, str, bool]:
    if not enable_compress:
        return content, suffix, False
    normalized_suffix = suffix.lower()
    if normalized_suffix not in IMAGE_SUFFIXES:
        return content, suffix, False
    compressed = compress_image_bytes_to_jpg(content, quality=jpeg_quality)
    return compressed, ".jpg", True


def _resolve_image_mime_for_save(content: bytes) -> str:
    """解析保存阶段使用的图片 MIME，非图片直接报错。"""
    inferred_mime = infer_image_mime(content, default_mime="")
    if inferred_mime.startswith("image/"):
        return inferred_mime
    raise ValueError("image mime type is not valid.")


async def _load_image_content(
    image: PluginImage,
    *,
    http_timeout_sec: int,
) -> bytes:
    """加载图片原始字节；失败时直接抛异常。"""
    if image.kind == "data_url":
        return base64.b64decode(image.to_base64(), validate=True)

    if image.kind == "base64":
        return base64.b64decode(image.value, validate=True)

    if image.kind == "http_url":
        content, _ = await download_http_resource(
            image.value,
            timeout_sec=http_timeout_sec,
        )
        return content

    raise ValueError(f"unsupported image kind: {image.kind}")


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
    content = await _load_image_content(
        image,
        http_timeout_sec=http_timeout_sec,
    )
    resolved_mime = _resolve_image_mime_for_save(content)
    suffix = suffix_from_content_type(resolved_mime)
    content, suffix, compressed = _maybe_compress_image_bytes(
        content,
        suffix,
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
