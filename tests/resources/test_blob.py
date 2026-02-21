from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from src.resources import ImageBlob, ResourceBlob


def _build_png_bytes() -> bytes:
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_resource_blob_sniff_and_encode() -> None:
    """验证：ResourceBlob 可嗅探 MIME/后缀并输出 data URL。"""
    blob = ResourceBlob(data=_build_png_bytes())

    assert blob.mime == "image/png"
    assert blob.extension == "png"
    assert blob.to_data_url().startswith("data:image/png;base64,")


def test_resource_blob_save(tmp_path) -> None:
    """验证：save 会写入目标文件并返回目标路径。"""
    blob = ResourceBlob(data=b"abc", default_mime="application/octet-stream")
    output_path = tmp_path / "a.bin"

    saved_path = blob.save(output_path)

    assert saved_path == output_path
    assert output_path.read_bytes() == b"abc"


def test_resource_blob_transfer_to_image_blob_raises_for_non_image() -> None:
    """验证：非图片资源转换为 ImageBlob 时会抛出异常。"""
    blob = ResourceBlob(data=b"hello", default_mime="text/plain")

    with pytest.raises(ValueError, match="image mime is not valid"):
        blob.transfer_to_image_blob()


def test_image_blob_compress_to_jpg_returns_new_blob() -> None:
    """验证：compress_to_jpg 返回新对象且输出为 JPEG 类型。"""
    source = ImageBlob(data=_build_png_bytes(), default_mime="image/png")

    compressed = source.compress_to_jpg(quality=80)

    assert compressed is not source
    assert compressed.mime == "image/jpeg"
    assert compressed.extension == "jpg"
    assert compressed.data


def test_image_blob_compress_to_jpg_quality_validation() -> None:
    """验证：compress_to_jpg 会校验 quality 参数边界。"""
    source = ImageBlob(data=_build_png_bytes(), default_mime="image/png")

    with pytest.raises(ValueError, match="quality must be in \\[1, 95\\]"):
        source.compress_to_jpg(quality=0)
