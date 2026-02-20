from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image
from src.images.codec import data_url_to_base64
from src.images.io import base64_image_to_data_url, compress_image_bytes_to_jpg


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


def test_base64_image_to_data_url_infers_png_mime() -> None:
    """验证：base64 图片可被转换为 data URL，且能推断 PNG 类型。"""
    png_bytes = _build_png_bytes()
    encoded = base64.b64encode(png_bytes).decode("ascii")

    data_url = base64_image_to_data_url(encoded)

    assert data_url.startswith("data:image/png;base64,")


def test_base64_image_to_data_url_supports_base64_scheme() -> None:
    """验证：支持 base64:// 前缀输入。"""
    png_bytes = _build_png_bytes()
    encoded = base64.b64encode(png_bytes).decode("ascii")

    data_url = base64_image_to_data_url(f"base64://{encoded}")

    assert data_url.startswith("data:image/png;base64,")


def test_base64_image_to_data_url_invalid_input() -> None:
    """验证：非法 base64 输入会抛出 ValueError。"""
    with pytest.raises(ValueError, match="not valid base64"):
        base64_image_to_data_url("not-base64@@@")


def test_base64_image_to_data_url_rejects_non_base64_data_url() -> None:
    """验证：非 base64 data URL 输入会抛出 ValueError。"""
    with pytest.raises(ValueError, match="not valid base64"):
        base64_image_to_data_url("data:text/plain,hello%20world")


def test_data_url_to_base64_for_base64_payload() -> None:
    """验证：可从 base64 data URL 提取并规范化 base64 字符串。"""
    png_bytes = _build_png_bytes()
    encoded = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"

    result = data_url_to_base64(data_url)

    assert result == encoded


def test_data_url_to_base64_for_non_base64_payload() -> None:
    """验证：非 base64 data URL 会抛出 ValueError。"""
    data_url = "data:text/plain,hello%20world"

    with pytest.raises(ValueError, match="must contain ';base64'"):
        data_url_to_base64(data_url)


def test_data_url_to_base64_invalid_input() -> None:
    """验证：非法 data URL 会抛出 ValueError。"""
    with pytest.raises(ValueError, match="must start with 'data:'"):
        data_url_to_base64("https://example.com/a.png")
