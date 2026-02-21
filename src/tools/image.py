from __future__ import annotations

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image

from ..resources import ResourceSpec
from ..resources.codec import transfer_data_url_to_base64
from ..utils.log import logger


def _parse_image_component_raw(
    *,
    file_raw: str,
    url_raw: str,
) -> ResourceSpec | None:
    normalized_file = file_raw.strip()
    if normalized_file:
        if normalized_file.startswith(("http://", "https://")):
            return ResourceSpec.from_http_url(normalized_file)
        if normalized_file.startswith("data:"):
            return ResourceSpec.from_data_url(normalized_file)
        if normalized_file.startswith("base64://"):
            return ResourceSpec.from_base64(normalized_file, mime="image/png")

    normalized_url = url_raw.strip()
    if normalized_url:
        if normalized_url.startswith(("http://", "https://")):
            return ResourceSpec.from_http_url(normalized_url)
        if normalized_url.startswith("data:"):
            return ResourceSpec.from_data_url(normalized_url)

    return None


async def extract_images_from_event(event: AstrMessageEvent) -> list[ResourceSpec]:
    """从消息事件提取图片并统一为 `ResourceSpec`。"""

    images: list[ResourceSpec] = []
    for component in event.get_messages():
        if not isinstance(component, Image):
            continue
        try:
            parsed = _parse_image_component_raw(
                url_raw=component.url or "",
                file_raw=component.file or "",
            )
            if parsed is not None:
                images.append(parsed)
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
            images.append(ResourceSpec.from_base64(encoded, mime="image/png"))
        except ValueError:
            logger.exception(
                "Failed to convert normalized base64 image to ResourceSpec."
            )
    return images


def build_image_send_result(event: AstrMessageEvent, image: ResourceSpec):
    """将适配器输出图片转换为可发送结果。"""
    if image.kind == "http_url":
        return event.image_result(image.raw)
    if image.kind == "data_url":
        try:
            encoded = transfer_data_url_to_base64(image.raw)
        except ValueError:
            logger.warning(
                "image.output_convert_base64_failed",
                {
                    "kind": image.kind,
                },
            )
            return None
        return event.make_result().base64_image(encoded)
    if image.kind == "base64":
        return event.make_result().base64_image(image.raw)

    logger.warning(
        "image.unsupported_output_kind",
        {"kind": image.kind},
    )
    return None
