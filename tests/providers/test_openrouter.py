from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from src.providers.openrouter import OPENROUTER_DEFAULT_BASE_URL, OpenRouterAdapter
from src.providers.schema import ImageGenerateInput
from src.providers.utils import ImageGenerateRenderResult
from src.resources import ResourceSpec
from src.utils.errors import PluginErrorCode, PluginException

from astrbot.api.event import MessageEventResult


def _make_adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        timeout_sec=30,
        image_model="test-image-model",
        tool_model="test-tool-model",
        save_image_format="png",
    )


@pytest.mark.asyncio
async def test_openrouter_image_generate_modalities_default() -> None:
    """验证：默认模型使用 image+text 双模态。"""
    adapter = _make_adapter()

    payload, _ = await adapter._build_image_generate_payload(
        ImageGenerateInput(
            prompt="A cat on the moon",
            aspect_ratio="1:1",
            image_size="1K",
            count=1,
        ),
        image_model=adapter.image_model,
    )

    assert payload["modalities"] == ["image", "text"]


@pytest.mark.asyncio
async def test_openrouter_image_generate_modalities_seedream_image_only() -> None:
    """验证：seedream 系列模型只使用 image 单模态。"""
    adapter = OpenRouterAdapter(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        timeout_sec=30,
        image_model="bytedance-seed/seedream-4.5",
        tool_model="test-tool-model",
        save_image_format="png",
    )

    payload, _ = await adapter._build_image_generate_payload(
        ImageGenerateInput(
            prompt="A cat on the moon",
            aspect_ratio="1:1",
            image_size="1K",
            count=1,
        ),
        image_model=adapter.image_model,
    )

    assert payload["modalities"] == ["image"]


@pytest.mark.asyncio
async def test_openrouter_image_generate_without_image_config_fields() -> None:
    """验证：未指定比例和分辨率时，不传 image_config 字段。"""
    adapter = _make_adapter()

    payload, _ = await adapter._build_image_generate_payload(
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
        save_image_format="png",
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
async def test_openrouter_image_generate_reference_images_to_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：参考图会按协议拼入 image_url 输入。"""
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
                ResourceSpec.from_http_url("https://example.com/ref.png"),
                ResourceSpec.from_data_url("data:image/png;base64,ZmFrZS1yZWY="),
                ResourceSpec.from_base64("ZmFrZS1yZWY=", mime="image/png"),
            ],
        )
    )

    content = captured_payload["payload"]["messages"][0]["content"]
    image_inputs = [item for item in content if item.get("type") == "image_url"]

    assert captured_payload["payload"]["n"] == 1
    assert len(image_inputs) == 3
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


@pytest.mark.asyncio
async def test_openrouter_image_generate_jpg_postprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：save_image_format=jpg 时对输出进行 jpg 压缩。"""
    adapter = OpenRouterAdapter(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        timeout_sec=30,
        image_model="test-image-model",
        tool_model="test-tool-model",
        save_image_format="jpg",
    )

    async def fake_request(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": {
                "choices": [
                    {
                        "message": {
                            "images": [
                                {
                                    "image_url": {
                                        "url": (
                                            "data:image/png;base64,"
                                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nmf8AAAAASUVORK5CYII="
                                        )
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "elapsed_ms": 123,
        }

    monkeypatch.setattr(adapter, "_request_chat_completions", fake_request)

    result = await adapter.image_generate(
        ImageGenerateInput(
            prompt="A tiny red dot",
            count=1,
        )
    )

    assert len(result.images) == 1
    assert result.images[0].kind == "base64"
    assert result.images[0].mime == "image/jpeg"
    assert result.warnings == []


@pytest.mark.asyncio
async def test_image_generate_tool_returns_single_detail_text_without_sendable_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：无可发送图片时仅返回合并后的 detail_text。"""
    adapter = _make_adapter()

    def fake_render_result(*args, **kwargs):
        return ImageGenerateRenderResult(
            send_results=[],
            sent_count=0,
            detail_text="生图完成\n提示：生图完成，但没有可发送的图片。",
        )

    monkeypatch.setattr(
        "src.providers.openrouter.build_image_generate_render_result",
        fake_render_result,
    )

    async def fake_success(_: ImageGenerateInput):
        class _Output:
            pass

        return _Output()

    monkeypatch.setattr(adapter, "image_generate", fake_success)
    tool = adapter.get_image_generate_tool(show_image_generate_details=False)
    handler = tool.handler
    assert handler is not None

    class _DummyEvent:
        pass

    output = handler(
        _DummyEvent(),
        prompt="A cat on the moon",
    )
    results: list[Any]
    if isinstance(output, AsyncGenerator):
        results = [item async for item in output]
    else:
        results = [await output]

    assert results == ["生图完成\n提示：生图完成，但没有可发送的图片。"]


@pytest.mark.asyncio
async def test_image_generate_tool_sends_user_messages_via_event_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证：工具通过 event.send 给用户发消息，yield 仅用于给模型返回文本。"""
    adapter = _make_adapter()

    marker_result = MessageEventResult().message("image-send")

    def fake_render_result(*args, **kwargs):
        return ImageGenerateRenderResult(
            send_results=[marker_result],
            sent_count=1,
            detail_text="生图完成",
        )

    monkeypatch.setattr(
        "src.providers.openrouter.build_image_generate_render_result",
        fake_render_result,
    )

    async def fake_success(_: ImageGenerateInput):
        class _Output:
            pass

        return _Output()

    monkeypatch.setattr(adapter, "image_generate", fake_success)
    tool = adapter.get_image_generate_tool(show_image_generate_details=True)
    handler = tool.handler
    assert handler is not None

    class _DummyEvent:
        def __init__(self):
            self.sent: list[MessageEventResult] = []

        def plain_result(self, text: str) -> MessageEventResult:
            return MessageEventResult().message(text)

        async def send(self, message: MessageEventResult) -> None:
            self.sent.append(message)

    event = _DummyEvent()
    output = handler(event, prompt="A cat on the moon")
    results: list[Any]
    if isinstance(output, AsyncGenerator):
        results = [item async for item in output]
    else:
        results = [await output]

    assert results == ["生图完成"]
    assert len(event.sent) == 2
    assert event.sent[0].get_plain_text() == "生图完成"
    assert event.sent[1] is marker_result
