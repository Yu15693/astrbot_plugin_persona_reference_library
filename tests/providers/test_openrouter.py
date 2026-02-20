from __future__ import annotations

from typing import Any

import pytest
from src.images import PluginImage
from src.providers.openrouter import OPENROUTER_DEFAULT_BASE_URL, OpenRouterAdapter
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


def test_openrouter_image_generate_modalities_default() -> None:
    """验证：默认模型使用 image+text 双模态。"""
    adapter = _make_adapter()

    payload, _ = adapter._build_image_generate_payload(
        ImageGenerateInput(
            prompt="A cat on the moon",
            aspect_ratio="1:1",
            image_size="1K",
            count=1,
        ),
        image_model=adapter.image_model,
    )

    assert payload["modalities"] == ["image", "text"]


def test_openrouter_image_generate_modalities_seedream_image_only() -> None:
    """验证：seedream 系列模型只使用 image 单模态。"""
    adapter = OpenRouterAdapter(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        timeout_sec=30,
        image_model="bytedance-seed/seedream-4.5",
        tool_model="test-tool-model",
    )

    payload, _ = adapter._build_image_generate_payload(
        ImageGenerateInput(
            prompt="A cat on the moon",
            aspect_ratio="1:1",
            image_size="1K",
            count=1,
        ),
        image_model=adapter.image_model,
    )

    assert payload["modalities"] == ["image"]


def test_openrouter_image_generate_without_image_config_fields() -> None:
    """验证：未指定比例和分辨率时，不传 image_config 字段。"""
    adapter = _make_adapter()

    payload, _ = adapter._build_image_generate_payload(
        ImageGenerateInput(
            prompt="A cat on the moon",
            count=1,
        ),
        image_model=adapter.image_model,
    )

    assert "image_config" not in payload


@pytest.mark.asyncio
async def test_openrouter_request_uses_default_base_url_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：base_url 留空时，自动回退到 OpenRouter 默认地址。"""
    adapter = OpenRouterAdapter(
        base_url="",
        api_key="test-key",
        timeout_sec=30,
        image_model="test-image-model",
        tool_model="test-tool-model",
    )
    assert adapter.base_url == OPENROUTER_DEFAULT_BASE_URL
    captured: dict[str, Any] = {}

    async def fake_post_json(*, url: str, **_: Any) -> dict[str, Any]:
        captured["url"] = url
        return {"data": {"choices": []}, "elapsed_ms": 1}

    monkeypatch.setattr("src.providers.openrouter.post_json", fake_post_json)

    await adapter._request_chat_completions({"messages": []})

    assert captured["url"] == f"{OPENROUTER_DEFAULT_BASE_URL}/chat/completions"


@pytest.mark.asyncio
async def test_openrouter_image_generate_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：上游返回合法图片 URL 时可正确归一化输出。"""
    adapter = _make_adapter()

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": {
                "choices": [
                    {
                        "message": {
                            "images": [
                                {
                                    "image_url": {
                                        "url": "https://example.com/image-1.png"
                                    }
                                },
                                {
                                    "image_url": {
                                        "url": "data:image/png;base64,ZmFrZS1pbWFnZQ=="
                                    }
                                },
                            ]
                        }
                    }
                ]
            },
            "elapsed_ms": 321,
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
    assert result.metadata.provider == "openrouter"
    assert result.metadata.model == "test-image-model"
    assert result.metadata.elapsed_ms == 321


@pytest.mark.asyncio
async def test_openrouter_image_generate_reference_validation_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：参考图列表中的 PluginImage 会被正确带入请求。"""
    adapter = _make_adapter()
    captured_payload: dict[str, Any] = {}

    async def fake_request(payload: dict[str, Any]) -> dict[str, Any]:
        captured_payload["payload"] = payload
        return {
            "data": {
                "choices": [
                    {
                        "message": {
                            "images": [
                                {
                                    "image_url": {
                                        "url": "https://example.com/image-1.png"
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "elapsed_ms": 56,
        }

    monkeypatch.setattr(adapter, "_request_chat_completions", fake_request)

    result = await adapter.image_generate(
        ImageGenerateInput(
            prompt="A portrait in watercolor",
            aspect_ratio="3:4",
            image_size="2K",
            count=1,
            reference_images=[
                PluginImage.from_http_url("https://example.com/ref.png"),
                PluginImage.from_data_url("data:image/png;base64,ZmFrZS1yZWY="),
            ],
        )
    )

    content = captured_payload["payload"]["messages"][0]["content"]
    image_inputs = [item for item in content if item.get("type") == "image_url"]

    assert captured_payload["payload"]["n"] == 1
    assert len(image_inputs) == 2
    assert result.warnings == []
    assert result.metadata.provider == "openrouter"
    assert result.metadata.model == "test-image-model"
    assert result.metadata.elapsed_ms == 56


@pytest.mark.asyncio
async def test_openrouter_image_generate_count_mismatch_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：返回数量与请求数量不一致时会产生 warning。"""
    adapter = _make_adapter()

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": {
                "choices": [
                    {
                        "message": {
                            "images": [
                                {
                                    "image_url": {
                                        "url": "https://example.com/image-1.png"
                                    }
                                },
                                {
                                    "image_url": {
                                        "url": "https://example.com/image-2.png"
                                    }
                                },
                            ]
                        }
                    }
                ]
            },
            "elapsed_ms": 78,
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
    assert result.metadata.provider == "openrouter"
    assert result.metadata.model == "test-image-model"
    assert result.metadata.elapsed_ms == 78


@pytest.mark.asyncio
async def test_openrouter_image_generate_no_images_raises_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：上游无有效图片时抛出 UPSTREAM_ERROR。"""
    adapter = _make_adapter()

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": {
                "choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]
            },
            "elapsed_ms": 15,
        }

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
