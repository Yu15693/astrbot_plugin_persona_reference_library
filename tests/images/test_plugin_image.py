from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image
from src.images import PluginImage

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Z6N0A"
    "AAAASUVORK5CYII="
)


def _build_jpeg_base64() -> str:
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    output = BytesIO()
    image.save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_plugin_image_from_http_url() -> None:
    """验证：可从 http_url 创建对象并按扩展名推断 MIME。"""
    image = PluginImage.from_raw("https://example.com/a.png")

    assert image.kind == "http_url"
    assert image.value == "https://example.com/a.png"
    assert image.mime == "image/png"


def test_plugin_image_from_data_url() -> None:
    """验证：可从 data_url 创建对象并提取 MIME。"""
    data_url = f"data:image/png;base64,{_PNG_BASE64}"
    image = PluginImage.from_raw(data_url)

    assert image.kind == "data_url"
    assert image.mime == "image/png"
    assert image.to_base64() == _PNG_BASE64


def test_plugin_image_from_base64_scheme() -> None:
    """验证：支持 base64:// 输入并归一化为 base64 kind。"""
    image = PluginImage.from_raw(f"base64://{_PNG_BASE64}")

    assert image.kind == "base64"
    assert image.value == _PNG_BASE64
    assert image.mime == "image/png"


def test_plugin_image_from_plain_base64() -> None:
    """验证：支持纯 base64 输入。"""
    image = PluginImage.from_raw(_PNG_BASE64)

    assert image.kind == "base64"
    assert image.value == _PNG_BASE64
    assert image.mime == "image/png"
    assert image.to_data_url().startswith("data:image/png;base64,")
    assert base64.b64decode(image.to_base64(), validate=True).startswith(b"\x89PNG\r\n\x1a\n")


def test_plugin_image_base64_to_data_url() -> None:
    """验证：base64 kind 可转 data_url。"""
    image = PluginImage.from_base64(_PNG_BASE64)

    assert image.to_data_url().startswith("data:image/png;base64,")
    assert image.to_data_url().endswith(_PNG_BASE64)


def test_plugin_image_from_base64_infers_jpeg_mime() -> None:
    """验证：from_base64 会按内容推断 MIME，而非固定为 image/png。"""
    jpeg_base64 = _build_jpeg_base64()
    image = PluginImage.from_base64(jpeg_base64)

    assert image.kind == "base64"
    assert image.mime == "image/jpeg"
    assert image.to_data_url().startswith("data:image/jpeg;base64,")
    assert image.default_mime == "image/png"


def test_plugin_image_from_base64_fallback_to_default_mime() -> None:
    """验证：base64 非图片内容时，mime 回退到 default_mime。"""
    text_base64 = base64.b64encode(b"hello").decode("ascii")
    image = PluginImage.from_base64(text_base64, default_mime="image/webp")

    assert image.kind == "base64"
    assert image.mime == "image/webp"
    assert image.default_mime == "image/webp"
    assert image.to_data_url().startswith("data:image/webp;base64,")


def test_plugin_image_http_to_data_url_raises() -> None:
    """验证：http_url 不支持直接转 data_url。"""
    image = PluginImage.from_http_url("https://example.com/a.jpg")

    with pytest.raises(ValueError, match="http_url image can not be converted"):
        image.to_data_url()


def test_plugin_image_local_path_falls_back_to_base64_validation() -> None:
    """验证：本地路径输入会走 base64 校验并报格式错误。"""
    with pytest.raises(ValueError, match="base64 payload is invalid"):
        PluginImage.from_raw("/tmp/a.png")


@pytest.mark.asyncio
async def test_plugin_image_enrich_mime_with_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证：http_url 可按需通过 HEAD 更新 MIME。"""

    async def _fake_head(
        url: str,
        *,
        timeout_sec: int = 5,
        default_mime: str = "",
    ) -> str:
        assert url == "https://example.com/download"
        assert timeout_sec == 3
        assert default_mime == ""
        return "image/webp"

    monkeypatch.setattr("src.images.plugin_image.infer_http_mime_with_head", _fake_head)
    image = PluginImage.from_http_url("https://example.com/download")
    assert image.mime == ""

    await image.enrich_mime_with_head(timeout_sec=3)
    assert image.mime == "image/webp"


@pytest.mark.asyncio
async def test_plugin_image_enrich_mime_with_head_ignores_non_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：HEAD 返回非图片类型时不覆盖 MIME。"""

    async def _fake_head(
        url: str,
        *,
        timeout_sec: int = 5,
        default_mime: str = "",
    ) -> str:
        assert url == "https://example.com/no-ext"
        assert timeout_sec == 2
        return default_mime

    monkeypatch.setattr("src.images.plugin_image.infer_http_mime_with_head", _fake_head)
    image = PluginImage.from_http_url("https://example.com/no-ext")
    image.mime = "image/png"

    await image.enrich_mime_with_head(timeout_sec=2)
    assert image.mime == "image/png"
