from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class ProviderAdapterConfig:
    provider: str
    """供应商标识"""
    base_url: str
    """供应商 API 基础地址"""
    api_key: str
    """供应商 API 密钥"""
    timeout_sec: int
    """HTTP 请求超时时间（秒）"""
    image_model: str
    """图像生成模型名称"""
    tool_model: str
    """工具调用模型名称"""


@dataclass(slots=True)
class ImageGenerateInput:
    prompt: str
    """生图提示词"""
    aspect_ratio: str
    """目标宽高比"""
    image_size: str
    """目标分辨率"""
    count: int = 1
    """期望生成数量"""
    reference_images: list[str] = field(default_factory=list)
    """参考图输入列表，每项可为 http(s) URL 或 data URL。"""


@dataclass(slots=True)
class AdapterImage:
    kind: Literal["http_url", "data_url"]
    """图片承载形式：`http_url` 或 `data_url`。"""
    value: str
    """图片内容：当 `kind=http_url` 时为 http(s) URL，当 `kind=data_url` 时为 data URL。"""


@dataclass(slots=True)
class InferenceMetadata:
    provider: str
    """供应商标识。"""
    model: str
    """实际请求使用的模型名。"""
    elapsed_ms: int | None = None
    """从发起请求到收到响应的耗时（毫秒）。"""


@dataclass(slots=True)
class ImageGenerateOutput:
    images: list[AdapterImage]
    metadata: InferenceMetadata | None = None
    """推理元数据"""
    warnings: list[str] = field(default_factory=list)
