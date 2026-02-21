from __future__ import annotations

from dataclasses import dataclass

from astrbot.api.event import AstrMessageEvent, MessageEventResult

from ..resources import ResourceSpec
from ..tools.image import build_image_send_result
from .schema import ImageGenerateOutput


def parse_reference_images(raw_value: list[str] | None) -> list[ResourceSpec]:
    """按既有 ResourceSpec 规则解析 reference_images。"""
    if raw_value is None:
        return []
    parsed: list[ResourceSpec] = []
    for item in raw_value:
        normalized = item.strip()
        if normalized.startswith(("http://", "https://")):
            parsed.append(ResourceSpec.from_http_url(normalized))
            continue
        if normalized.startswith("data:"):
            parsed.append(ResourceSpec.from_data_url(normalized))
            continue
        if normalized.startswith("base64://"):
            parsed.append(ResourceSpec.from_base64(normalized, mime="image/png"))
            continue
        # 兼容直接传原始 base64 字符串。
        parsed.append(ResourceSpec.from_base64(normalized, mime="image/png"))
    return parsed


@dataclass(slots=True)
class ImageGenerateRenderResult:
    send_results: list[MessageEventResult]
    sent_count: int
    detail_text: str


def build_image_generate_render_result(
    event: AstrMessageEvent,
    output: ImageGenerateOutput,
    *,
    requested_count: int,
) -> ImageGenerateRenderResult:
    """构建统一的生图输出结果，供 tool 与命令共用。"""
    send_results: list[MessageEventResult] = []
    for image in output.images:
        result = build_image_send_result(event, image)
        if result is not None:
            send_results.append(result)

    sent_count = len(send_results)
    detail_lines = [
        "生图完成",
        f"模型={output.metadata.model}  耗时={output.metadata.elapsed_ms} ms",
        f"请求数量={requested_count}  返回数量={len(output.images)}  发送数量={sent_count}",
    ]
    if output.warnings:
        detail_lines.append("警告：")
        detail_lines.extend(f"- {warning}" for warning in output.warnings)
    if sent_count == 0:
        detail_lines.append("提示：生图完成，但没有可发送的图片。")

    return ImageGenerateRenderResult(
        send_results=send_results,
        sent_count=sent_count,
        detail_text="\n".join(detail_lines),
    )
