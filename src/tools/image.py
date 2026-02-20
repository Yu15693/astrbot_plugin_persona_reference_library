from __future__ import annotations

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image

from ..providers.schema import AdapterImage
from ..utils.io import base64_image_to_data_url, data_url_to_base64
from ..utils.log import logger
from ..utils.url import is_data_url, is_http_url


async def extract_images_from_event(event: AstrMessageEvent) -> list[AdapterImage]:
    """从消息事件提取图片并统一为 `AdapterImage`（`http_url`/`data_url`）。"""

    images: list[AdapterImage] = []
    for component in event.get_messages():
        if not isinstance(component, Image):
            continue
        raw = (component.url or component.file or "").strip()
        if is_http_url(raw):
            images.append(AdapterImage(kind="http_url", value=raw))
            continue
        if is_data_url(raw):
            images.append(AdapterImage(kind="data_url", value=raw))
            continue
        if raw.startswith("base64://"):
            try:
                images.append(
                    AdapterImage(kind="data_url", value=base64_image_to_data_url(raw))
                )
            except ValueError:
                logger.exception(
                    "Failed to convert input base64 image to data URL."
                )
            continue
        try:
            encoded = await component.convert_to_base64()
        except Exception:
            logger.exception("Failed to convert input image to base64.")
            continue
        try:
            images.append(
                AdapterImage(kind="data_url", value=base64_image_to_data_url(encoded))
            )
        except ValueError:
            logger.exception(
                "Failed to convert normalized base64 image to data URL."
            )
    return images


def build_image_send_result(event: AstrMessageEvent, image: AdapterImage):
    """将适配器输出图片转换为可发送结果。"""
    if image.kind == "http_url":
        return event.image_result(image.value)
    if image.kind == "data_url":
        try:
            encoded = data_url_to_base64(image.value)
        except ValueError:
            logger.exception("Failed to convert output data URL to base64.")
            return None
        return event.make_result().base64_image(encoded)

    logger.warning(
        "image.unsupported_output_kind",
        {"kind": image.kind},
    )
    return None
