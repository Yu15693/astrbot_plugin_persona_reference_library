from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from src.resources import ResourceSpec


def _build_png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_resource_spec_base64_normalization() -> None:
    """验证：base64 输入会移除前缀与空白并保留规范化 MIME。"""
    spec = ResourceSpec.from_base64("  base64:// Zm9vIA==  ", mime="image/png")

    assert spec.kind == "base64"
    assert spec.raw == "Zm9vIA=="
    assert spec.mime == "image/png"


def test_resource_spec_kind_value_mismatch_raises() -> None:
    """验证：kind 与 raw 内容不匹配时会抛出明确异常。"""
    with pytest.raises(ValueError, match="http_url resource must be a valid http"):
        ResourceSpec.from_http_url("data:image/png;base64,Zm9v")

    with pytest.raises(ValueError, match="data_url resource must start with 'data:'"):
        ResourceSpec.from_data_url("https://example.com/a.png")


@pytest.mark.asyncio
async def test_resource_spec_to_data_url_from_base64_uses_default_mime() -> None:
    """验证：base64 资源在未声明 MIME 时会使用 default_mime 组装 data URL。"""
    spec = ResourceSpec.from_base64("ZmFrZQ==")

    data_url = await spec.to_data_url(default_mime="image/png")

    assert data_url == "data:image/png;base64,ZmFrZQ=="


@pytest.mark.asyncio
async def test_resource_spec_to_base64_from_plain_data_url() -> None:
    """验证：非 base64 的 data URL 能转换为标准 base64 字符串。"""
    spec = ResourceSpec.from_data_url("data:text/plain,hello%20world")

    encoded = await spec.to_base64()

    assert encoded == "aGVsbG8gd29ybGQ="


@pytest.mark.asyncio
async def test_resource_spec_convert_to_resource_blob_enforces_max_bytes() -> None:
    """验证：convert_to_resource_blob 会严格执行 max_bytes 限制。"""
    spec = ResourceSpec.from_base64("aGVsbG8=")

    with pytest.raises(ValueError, match="exceeds max_bytes"):
        await spec.convert_to_resource_blob(max_bytes=4)


@pytest.mark.asyncio
async def test_resource_spec_convert_to_image_blob_rejects_non_image() -> None:
    """验证：非图片内容转换为 ImageBlob 时会被拒绝。"""
    spec = ResourceSpec.from_base64("aGVsbG8=", mime="text/plain")

    with pytest.raises(ValueError, match="image mime is not valid"):
        await spec.convert_to_image_blob()


@pytest.mark.asyncio
async def test_resource_spec_convert_http_url_with_mocked_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：http_url 资源可通过 loader 下载并嗅探出图片类型。"""
    async def fake_get_bytes(*, url: str, timeout_sec: int):
        assert url == "https://example.com/demo.png"
        assert timeout_sec == 60
        return {
            "data": _build_png_bytes(),
            "mime": "",
            "elapsed_ms": 1,
        }

    monkeypatch.setattr("src.resources.spec.get_bytes", fake_get_bytes)
    spec = ResourceSpec.from_http_url("https://example.com/demo.png")

    blob = await spec.convert_to_resource_blob()

    assert blob.mime == "image/png"
    assert blob.extension == "png"
