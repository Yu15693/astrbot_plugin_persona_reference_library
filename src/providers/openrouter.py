"""OpenRouter 供应商适配器"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..images import PluginImage
from ..utils.dicts import get_dict_value
from ..utils.errors import PluginErrorCode, PluginException
from ..utils.http import PostJsonSuccessResponse, post_json
from .base import ProviderAdapter
from .schema import (
    ImageGenerateInput,
    ImageGenerateOutput,
    InferenceMetadata,
)

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
IMAGE_ONLY_MODALITY_MODEL_KEYWORDS = ("seedream-4.5",)


@dataclass(slots=True)
class OpenRouterAdapter(ProviderAdapter):
    base_url: str
    api_key: str
    timeout_sec: int
    image_model: str
    tool_model: str
    provider: str = "openrouter"

    def __post_init__(self) -> None:
        normalized = self.base_url.strip()
        self.base_url = normalized or OPENROUTER_DEFAULT_BASE_URL

    async def _request_chat_completions(
        self, payload: dict[str, Any]
    ) -> PostJsonSuccessResponse:
        """统一请求 OpenRouter chat/completions 并返回响应封装对象。"""
        if not self.api_key.strip():
            raise PluginException(
                code=PluginErrorCode.PERMISSION_DENIED,
                message="OpenRouter API key is not configured.",
                retryable=False,
                detail={
                    "provider": self.provider,
                    "base_url": self.base_url,
                },
            )
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        response = await post_json(
            url=url,
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout_sec=self.timeout_sec,
            source="OpenRouter",
        )
        return response

    def _build_image_generate_payload(
        self,
        payload: ImageGenerateInput,
        *,
        image_model: str,
    ) -> tuple[dict[str, Any], list[str]]:
        """构造 OpenRouter 生图请求体。"""
        warnings: list[str] = []
        content: list[dict[str, Any]] = [{"type": "text", "text": payload.prompt}]
        for index, reference_image in enumerate(payload.reference_images):
            if reference_image.kind == "http_url":
                normalized = reference_image.value
            elif reference_image.kind in {"data_url", "base64"}:
                normalized = reference_image.to_data_url()
            else:
                warnings.append(
                    f"reference_images[{index}] has unsupported kind '{reference_image.kind}' and ignored."
                )
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": normalized},
                }
            )

        image_config = {
            key: value
            for key, value in {
                "aspect_ratio": payload.aspect_ratio.strip(),
                "image_size": payload.image_size.strip(),
            }.items()
            if value
        }

        request_payload: dict[str, Any] = {
            "model": image_model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "n": payload.count,
            "modalities": _build_image_modalities_for_model(image_model),
            **({"image_config": image_config} if image_config else {}),
        }
        return request_payload, warnings

    async def image_generate(self, payload: ImageGenerateInput) -> ImageGenerateOutput:
        """执行图像生成请求并返回统一 ImageGenerateOutput。"""
        provider = self.provider
        image_model = self.image_model
        request_payload, warnings = self._build_image_generate_payload(
            payload,
            image_model=image_model,
        )

        response = await self._request_chat_completions(request_payload)
        data = response["data"]
        elapsed_ms = response["elapsed_ms"]

        images = _extract_openrouter_images(data)
        if not images:
            raise PluginException(
                code=PluginErrorCode.UPSTREAM_ERROR,
                message="OpenRouter returned no valid image content.",
                retryable=True,
                detail={
                    "provider": provider,
                    "response_keys": sorted(data.keys()),
                    "elapsed_ms": elapsed_ms,
                },
            )

        if len(images) != payload.count:
            warnings.append(
                f"Upstream returned {len(images)} images, different from requested {payload.count}."
            )

        return ImageGenerateOutput(
            images=images,
            metadata=InferenceMetadata(
                provider=provider,
                model=image_model,
                elapsed_ms=elapsed_ms,
            ),
            warnings=warnings,
        )


def _build_image_modalities_for_model(image_model: str) -> list[str]:
    model_name = image_model.strip().lower()
    if any(keyword in model_name for keyword in IMAGE_ONLY_MODALITY_MODEL_KEYWORDS):
        return ["image"]
    return ["image", "text"]


def _extract_openrouter_images(
    data: dict[str, Any],
) -> list[PluginImage]:
    """提取 OpenRouter 返回中的图片内容（http(s) URL 或 data URL）。"""
    output: list[PluginImage] = []
    choices = data.get("choices")
    if not isinstance(choices, list):
        return output

    for choice in choices:
        # 结构1：choices[].message.images[].image_url.url
        message_images = get_dict_value(choice, "message", "images")
        if isinstance(message_images, list):
            for item in message_images:
                raw_url = get_dict_value(item, "image_url", "url")
                if isinstance(raw_url, str):
                    try:
                        output.append(PluginImage.from_raw(raw_url))
                    except ValueError:
                        continue

    return output
