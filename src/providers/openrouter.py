"""OpenRouter 供应商适配器"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent

from ..resources import ResourceSpec
from ..utils.dicts import get_dict_value
from ..utils.errors import PluginErrorCode, PluginException
from ..utils.http import PostJsonSuccessResponse, post_json
from ..utils.log import logger
from .base import ProviderAdapter
from .schema import (
    ImageGenerateInput,
    ImageGenerateOutput,
    InferenceMetadata,
)
from .utils import build_image_generate_render_result, parse_reference_images

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
IMAGE_ONLY_MODALITY_MODEL_KEYWORDS = ("seedream-4.5",)

OPENROUTER_SUPPORTED_ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]
OPENROUTER_SUPPORTED_IMAGE_SIZES = ["1K", "2K", "4K"]

OPENROUTER_IMAGE_GENERATE_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "生图提示词（非空白）。",
            "minLength": 1,
        },
        "aspect_ratio": {
            "type": "string",
            "description": "图片比例（可选）。可与 image_size 同时传入。",
            "enum": OPENROUTER_SUPPORTED_ASPECT_RATIOS,
        },
        "image_size": {
            "type": "string",
            "description": "图片尺寸档位（可选）。可与 aspect_ratio 同时传入。",
            "enum": OPENROUTER_SUPPORTED_IMAGE_SIZES,
        },
        "n": {
            "type": "integer",
            "description": "图片数量（可选，不传默认 1）。",
            "minimum": 1,
            "maximum": 4,
            "default": 1,
        },
        "reference_images": {
            "type": "array",
            "description": "参考图列表（可选），仅支持 http(s) 图片 URL。",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "minLength": 1,
                "pattern": "^https?://.+$",
            },
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class OpenRouterAdapter(ProviderAdapter):
    base_url: str
    api_key: str
    timeout_sec: int
    image_model: str
    tool_model: str
    save_image_format: str = "jpg"
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

    async def _build_image_generate_payload(
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
                normalized = reference_image.raw
            elif reference_image.kind in {"data_url", "base64"}:
                normalized = await reference_image.to_data_url(default_mime="image/png")
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
        request_payload, warnings = await self._build_image_generate_payload(
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

        images = await self._process_output_images(images, warnings)

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

    async def _process_output_images(
        self,
        images: list[ResourceSpec],
        warnings: list[str],
    ) -> list[ResourceSpec]:
        if self.save_image_format != "jpg":
            return images

        converted_images: list[ResourceSpec] = []
        for index, image in enumerate(images):
            try:
                image_blob = await image.convert_to_image_blob(
                    timeout_sec=self.timeout_sec
                )
                jpg_blob = image_blob.compress_to_jpg()
                converted_images.append(
                    ResourceSpec.from_base64(
                        jpg_blob.to_base64(),
                        mime="image/jpeg",
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"image[{index}] convert to jpg failed, fallback to original ({exc!s})."
                )
                converted_images.append(image)
        return converted_images

    def get_image_generate_tool(
        self,
        *,
        show_image_generate_details: bool = True,
    ) -> FunctionTool:
        async def handler(
            event: AstrMessageEvent,
            prompt: str,
            aspect_ratio: str = "",
            image_size: str = "",
            n: int = 1,
            reference_images: list[str] | None = None,
        ):
            parsed_reference_images = parse_reference_images(reference_images)
            payload = ImageGenerateInput(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                count=n,
                reference_images=parsed_reference_images,
            )

            try:
                output = await self.image_generate(payload)
            except Exception as exc:
                logger.exception("prl_image_generate failed: %s", exc)
                yield "生图失败：上游请求失败。"
                return

            render_result = build_image_generate_render_result(
                event,
                output,
                requested_count=payload.count,
            )

            # 这里不要 yield MessageEventResult。
            # tool 执行器会把 MessageEventResult 转成 `None`，
            # runner 随后会记录“tool has no return value”，并可能提前结束循环。
            # 因此面向用户的消息改为 event.send 直接发送，
            # 只把给模型的文本结果通过 `yield str` 返回。
            if show_image_generate_details:
                await event.send(event.plain_result(render_result.detail_text))

            for result in render_result.send_results:
                await event.send(result)

            yield render_result.detail_text

        return FunctionTool(
            self.image_generate_tool_name,
            "根据提示词生成图片。可选传入比例、尺寸、数量和参考图（参考图仅支持 http(s) 图片 URL，不支持 data URL/base64）。",
            OPENROUTER_IMAGE_GENERATE_TOOL_PARAMETERS,
            handler,  # type: ignore 类型符合要求
        )


def _build_image_modalities_for_model(image_model: str) -> list[str]:
    model_name = image_model.strip().lower()
    if any(keyword in model_name for keyword in IMAGE_ONLY_MODALITY_MODEL_KEYWORDS):
        return ["image"]
    return ["image", "text"]


def _extract_openrouter_images(
    data: dict[str, Any],
) -> list[ResourceSpec]:
    """提取 OpenRouter 返回中的图片内容（http(s) URL 或 data URL）。"""
    output: list[ResourceSpec] = []
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
                    normalized = raw_url.strip()
                    if not normalized:
                        continue
                    try:
                        if normalized.startswith(("http://", "https://")):
                            output.append(ResourceSpec.from_http_url(normalized))
                        elif normalized.startswith("data:"):
                            output.append(ResourceSpec.from_data_url(normalized))
                    except ValueError:
                        continue

    return output
