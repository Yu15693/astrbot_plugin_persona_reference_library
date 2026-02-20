from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..utils.url import is_data_url, is_http_url
from .codec import (
    build_data_url,
    data_url_to_base64,
    extract_data_url_mime,
    normalize_base64_payload,
    parse_raw_image_value,
)
from .mime import (
    infer_base64_mime,
    infer_http_mime_with_head,
    infer_http_url_mime,
)
from .normalize import normalize_mime, normalize_text

PluginImageKind = Literal["http_url", "data_url", "base64"]


@dataclass(slots=True)
class PluginImage:
    """
    插件内部统一图像对象。

    约定：
    - 创建成功后，`value` 一定是规范化后的值（去前后空白、base64 去掉 `base64://` 并标准化）。
    - `data_url` 与 `base64` 在构造期会完成格式校验；非法输入会直接抛出 ValueError。
    - `mime` 允许为空（典型场景是无法从 http URL 后缀推断时）。
    """

    kind: PluginImageKind
    value: str
    """
    图片承载值（按 kind 不同含义不同）：
    - http_url：完整的 http(s) URL。
    - data_url：完整的 data URL（如 data:image/png;base64,...）。
    - base64：纯 base64 负载字符串（不包含 base64:// 前缀，输入时会自动归一化）。
    """
    mime: str = ""
    """
    图片 MIME（可能为空）。
    - http_url：优先按 URL 扩展名推断；若无法推断则保持空字符串，可按需调用 enrich_mime_with_head 补全。
    - data_url：优先从 data URL 头部提取。
    - base64：优先按字节头推断，失败时使用默认值（通常为 image/png）。
    """
    default_mime: str = "image/png"
    """
    推断 MIME 失败时使用的回退值，不等同于最终 `mime`。
    """

    def __post_init__(self) -> None:
        """
        对输入做一次性规范化与校验。

        成功返回后对象满足：
        - `value` 与 `default_mime` 已规范化；
        - `kind` 对应的数据格式合法；
        - `mime` 若为空则按 kind 的规则自动推断或保持空字符串。
        """
        normalized_value = normalize_text(self.value)
        if not normalized_value:
            raise ValueError("image value must not be empty.")
        normalized_default_mime = normalize_mime(self.default_mime) or "image/png"
        normalized_mime = normalize_mime(self.mime)

        if self.kind == "http_url":
            if not is_http_url(normalized_value):
                raise ValueError("http_url image must be a valid http(s) URL.")
            if not normalized_mime:
                normalized_mime = infer_http_url_mime(normalized_value, "")

        elif self.kind == "data_url":
            if not is_data_url(normalized_value):
                raise ValueError("data_url image must be a valid data URL.")
            try:
                data_url_to_base64(normalized_value)
            except ValueError as exc:
                raise ValueError(
                    "data_url image must be a valid base64 data URL."
                ) from exc
            if not normalized_mime:
                normalized_mime = extract_data_url_mime(
                    normalized_value,
                    normalized_default_mime,
                )

        elif self.kind == "base64":
            normalized_value = normalize_base64_payload(normalized_value)
            if not normalized_mime:
                normalized_mime = infer_base64_mime(
                    normalized_value,
                    normalized_default_mime,
                )

        else:
            raise ValueError(f"unsupported image kind: {self.kind}")

        self.value = normalized_value
        self.mime = normalized_mime
        self.default_mime = normalized_default_mime

    @classmethod
    def from_http_url(cls, url: str) -> PluginImage:
        """从 HTTP URL 创建图像对象（构造期会校验 URL 格式）。"""
        return cls(kind="http_url", value=url)

    @classmethod
    def from_data_url(cls, data_url: str) -> PluginImage:
        """从 data URL 创建图像对象（要求是合法 base64 data URL）。"""
        return cls(kind="data_url", value=data_url)

    @classmethod
    def from_base64(
        cls,
        base64_value: str,
        *,
        default_mime: str = "image/png",
    ) -> PluginImage:
        """从 base64 字符串创建图像对象（构造期会校验并规范化）。"""
        return cls(
            kind="base64",
            value=base64_value,
            default_mime=default_mime,
        )

    @classmethod
    def from_raw(
        cls,
        raw: str,
        *,
        default_mime: str = "image/png",
    ) -> PluginImage:
        """从原始字符串创建图像对象，自动识别 http_url/data_url/base64。"""
        kind, value = parse_raw_image_value(raw)
        return cls(
            kind=kind,
            value=value,
            default_mime=default_mime,
        )

    def to_base64(self) -> str:
        """
        转换为纯 base64 字符串。

        规则：
        - kind=http_url：当前不支持直接下载并转换，抛出 ValueError。
        - kind=data_url：解析 data URL 负载后返回 base64。
        - kind=base64：直接返回归一化后的 base64 负载。
        """
        if self.kind == "http_url":
            raise ValueError("http_url image can not be converted to base64 directly.")
        if self.kind == "data_url":
            return data_url_to_base64(self.value)
        if self.kind == "base64":
            return self.value
        raise ValueError(f"unsupported image kind: {self.kind}")

    def to_data_url(self, *, default_mime: str = "") -> str:
        """
        转换为 data URL。

        规则：
        - kind=http_url：当前不支持直接下载并转换，抛出 ValueError。
        - kind=data_url：直接返回原值。
        - kind=base64：使用 mime（为空则回退 default_mime）组装 data URL。
        """
        if self.kind == "http_url":
            raise ValueError(
                "http_url image can not be converted to data URL directly."
            )
        if self.kind == "data_url":
            return self.value
        if self.kind == "base64":
            mime = self.mime or normalize_mime(default_mime) or self.default_mime
            return build_data_url(mime, self.value)
        raise ValueError(f"unsupported image kind: {self.kind}")

    async def enrich_mime_with_head(self, *, timeout_sec: int = 5) -> None:
        """
        按需通过 HTTP HEAD 更新 MIME（仅 http_url 生效）。

        该方法是 best-effort：请求失败或返回非 image 类型时不抛异常，保留原 `mime`。
        """
        if self.kind != "http_url":
            return
        resolved = await infer_http_mime_with_head(
            self.value,
            timeout_sec=timeout_sec,
            default_mime=self.mime,
        )
        if resolved:
            self.mime = resolved
