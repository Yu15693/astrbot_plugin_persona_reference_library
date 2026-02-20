from __future__ import annotations

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image

from ..images import PluginImage
from ..utils.log import logger


async def extract_images_from_event(event: AstrMessageEvent) -> list[PluginImage]:
    """从消息事件提取图片并统一为 `PluginImage`。"""

    images: list[PluginImage] = []
    for component in event.get_messages():
        if not isinstance(component, Image):
            continue
        raw = (component.url or component.file or "").strip()
        if raw:
            try:
                images.append(PluginImage.from_raw(raw))
                continue
            except ValueError as exc:
                logger.debug(
                    "image.parse_raw_fallback_to_base64",
                    {
                        "component": repr(component),
                        "reason": str(exc),
                    },
                )
        try:
            encoded = await component.convert_to_base64()
        except Exception:
            logger.exception("Failed to convert input image to base64.")
            continue
        try:
            images.append(PluginImage.from_base64(encoded))
        except ValueError:
            logger.exception(
                "Failed to convert normalized base64 image to PluginImage."
            )
    return images


def build_image_send_result(event: AstrMessageEvent, image: PluginImage):
    """将适配器输出图片转换为可发送结果。"""
    if image.kind == "http_url":
        return event.image_result(image.value)
    if image.kind in {"data_url", "base64"}:
        try:
            encoded = image.to_base64()
        except ValueError:
            logger.warning(
                "image.output_convert_base64_failed",
                {
                    "kind": image.kind,
                },
            )
            return None
        return event.make_result().base64_image(encoded)

    logger.warning(
        "image.unsupported_output_kind",
        {"kind": image.kind},
    )
    return None
