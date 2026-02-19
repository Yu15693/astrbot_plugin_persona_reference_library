from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from src.utils.io import compress_image_bytes_to_jpg


def _build_png_bytes() -> bytes:
    image = Image.new("RGBA", (128, 128), (255, 0, 0, 128))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_compress_image_bytes_to_jpg_success() -> None:
    """验证：可将 PNG bytes 转换并压缩为 JPG bytes。"""
    png_bytes = _build_png_bytes()

    jpg_bytes = compress_image_bytes_to_jpg(png_bytes, quality=80)

    assert jpg_bytes[:2] == b"\xff\xd8"
    with Image.open(BytesIO(jpg_bytes)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (128, 128)


def test_compress_image_bytes_to_jpg_invalid_quality() -> None:
    """验证：quality 越界时抛出 ValueError。"""
    png_bytes = _build_png_bytes()

    with pytest.raises(ValueError, match="quality must be in \\[1, 95\\]"):
        compress_image_bytes_to_jpg(png_bytes, quality=0)
