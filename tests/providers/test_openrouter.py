from __future__ import annotations

from typing import Any

import pytest

from src.providers.openrouter import OpenRouterAdapter
from src.providers.schema import ImageGenerateInput
from src.utils.errors import PluginErrorCode, PluginException


def _make_adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        timeout_sec=30,
        image_model="test-image-model",
        tool_model="test-tool-model",
    )


@pytest.mark.asyncio
async def test_openrouter_image_generate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证：上游返回合法图片 URL 时可正确归一化输出。"""
    adapter = _make_adapter()

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "https://example.com/image-1.png"}},
                            {
                                "image_url": {
                                    "url": "data:image/png;base64,ZmFrZS1pbWFnZQ=="
                                }
                            },
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(adapter, "_request_chat_completions", fake_request)

    result = await adapter.image_generate(
        ImageGenerateInput(
            prompt="A cat on the moon",
            aspect_ratio="1:1",
            image_size="1K",
            count=2,
        )
    )

    assert len(result.images) == 2
    assert result.images[0].kind == "http_url"
    assert result.images[1].kind == "data_url"
    assert result.warnings == []


@pytest.mark.asyncio
async def test_openrouter_image_generate_reference_validation_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：非法参考图会被忽略并产生 warning，合法参考图会保留。"""
    adapter = _make_adapter()
    captured_payload: dict[str, Any] = {}

    async def fake_request(payload: dict[str, Any]) -> dict[str, Any]:
        captured_payload["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "https://example.com/image-1.png"}}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(adapter, "_request_chat_completions", fake_request)

    result = await adapter.image_generate(
        ImageGenerateInput(
            prompt="A portrait in watercolor",
            aspect_ratio="3:4",
            image_size="2K",
            count=1,
            reference_images=[
                "",
                "ftp://invalid.example/ref.png",
                "https://example.com/ref.png",
                "data:image/png;base64,ZmFrZS1yZWY=",
            ],
        )
    )

    content = captured_payload["payload"]["messages"][0]["content"]
    image_inputs = [item for item in content if item.get("type") == "image_url"]

    assert captured_payload["payload"]["n"] == 1
    assert len(image_inputs) == 2
    assert any("is empty and ignored" in warning for warning in result.warnings)
    assert any("not a valid http(s) URL or data URL" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_openrouter_image_generate_count_mismatch_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：返回数量与请求数量不一致时会产生 warning。"""
    adapter = _make_adapter()

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "https://example.com/image-1.png"}},
                            {"image_url": {"url": "https://example.com/image-2.png"}},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(adapter, "_request_chat_completions", fake_request)

    result = await adapter.image_generate(
        ImageGenerateInput(
            prompt="A city skyline",
            aspect_ratio="16:9",
            image_size="1K",
            count=1,
        )
    )

    assert len(result.images) == 2
    assert any("different from requested" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_openrouter_image_generate_no_images_raises_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：上游无有效图片时抛出 UPSTREAM_ERROR。"""
    adapter = _make_adapter()

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]}

    monkeypatch.setattr(adapter, "_request_chat_completions", fake_request)

    with pytest.raises(PluginException) as exc_info:
        await adapter.image_generate(
            ImageGenerateInput(
                prompt="A mountain landscape",
                aspect_ratio="1:1",
                image_size="1K",
                count=1,
            )
        )

    assert exc_info.value.code == PluginErrorCode.UPSTREAM_ERROR
