from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from src.images import PluginImage, save_plugin_image


def _build_png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_save_plugin_image_data_url(tmp_path: Path) -> None:
    """验证：data_url 图片可保存为文件。"""
    import base64

    png_bytes = _build_png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
    data_image = PluginImage.from_data_url(data_url)

    result = await save_plugin_image(
        image=data_image,
        target_dir=tmp_path,
        filename_stem="1",
        enable_compress=False,
    )

    assert result.path is not None
    assert result.path.name.endswith(".png")
    assert result.path.exists()


@pytest.mark.asyncio
async def test_save_plugin_image_base64_with_compress(tmp_path: Path) -> None:
    """验证：base64 图片在开启压缩时保存为 jpg。"""
    import base64

    png_bytes = _build_png_bytes()
    image = PluginImage.from_base64(base64.b64encode(png_bytes).decode("ascii"))

    result = await save_plugin_image(
        image=image,
        target_dir=tmp_path,
        filename_stem="2",
        enable_compress=True,
        jpeg_quality=80,
    )

    assert result.path is not None
    assert result.path.name.endswith(".jpg")
    assert result.compressed is True
    assert result.path.exists()


@pytest.mark.asyncio
async def test_save_plugin_image_http_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证：http_url 图片可通过下载后保存。"""
    png_bytes = _build_png_bytes()

    async def _fake_download_http_resource(
        url: str,
        timeout_sec: int = 60,
    ) -> tuple[bytes, str]:
        assert url == "https://example.com/demo.png"
        assert timeout_sec == 60
        return png_bytes, ".png"

    monkeypatch.setattr("src.images.save.download_http_resource", _fake_download_http_resource)
    image = PluginImage.from_http_url("https://example.com/demo.png")

    result = await save_plugin_image(
        image=image,
        target_dir=tmp_path,
        filename_stem="3",
        enable_compress=False,
    )

    assert result.path is not None
    assert result.path.name.endswith(".png")
    assert result.path.exists()


@pytest.mark.asyncio
async def test_save_plugin_image_base64_non_image_rejected(tmp_path: Path) -> None:
    """验证：非图片 base64 内容不会保存，并抛出异常。"""
    import base64

    text_base64 = base64.b64encode(b"hello world").decode("ascii")
    image = PluginImage.from_base64(text_base64)

    with pytest.raises(ValueError, match="image mime type is not valid"):
        await save_plugin_image(
            image=image,
            target_dir=tmp_path,
            filename_stem="4",
            enable_compress=False,
        )


@pytest.mark.asyncio
async def test_save_plugin_image_http_url_non_image_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证：HTTP 下载到非图片内容时不会保存。"""

    async def _fake_download_http_resource(
        url: str,
        timeout_sec: int = 60,
    ) -> tuple[bytes, str]:
        assert url == "https://example.com/not-image"
        assert timeout_sec == 60
        return b"<html>not image</html>", ".html"

    monkeypatch.setattr("src.images.save.download_http_resource", _fake_download_http_resource)
    image = PluginImage.from_http_url("https://example.com/not-image")

    with pytest.raises(ValueError, match="image mime type is not valid"):
        await save_plugin_image(
            image=image,
            target_dir=tmp_path,
            filename_stem="5",
            enable_compress=False,
        )
