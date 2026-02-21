from __future__ import annotations

from dataclasses import dataclass

from ..utils.http import get_bytes
from .blob import ImageBlob, ResourceBlob
from .codec import (
    decode_base64_payload,
    transfer_base64_to_data_url,
    transfer_data_url_to_base64,
    transfer_data_url_to_bytes,
)
from .mime import (
    extract_mime_from_data_url,
    guess_mime_from_http_url,
)
from .normalize import (
    normalize_base64_payload,
    normalize_mime,
)
from .schema import ResourceKind


def _ensure_max_bytes(data: bytes, max_bytes: int | None) -> None:
    if max_bytes is None:
        return
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0.")
    if len(data) > max_bytes:
        raise ValueError(f"resource data exceeds max_bytes: {len(data)} > {max_bytes}.")


@dataclass(slots=True)
class ResourceSpec:
    kind: ResourceKind
    raw: str
    mime: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.raw, str):
            raise TypeError("raw must be str.")
        normalized_raw = self.raw.strip()
        if not normalized_raw:
            raise ValueError("raw resource value must not be empty.")

        if self.kind == "http_url":
            if not normalized_raw.startswith(("http://", "https://")):
                raise ValueError("http_url resource must be a valid http(s) URL.")
        elif self.kind == "data_url":
            if not normalized_raw.startswith("data:"):
                raise ValueError("data_url resource must start with 'data:'.")
        elif self.kind == "base64":
            normalized_raw = normalize_base64_payload(normalized_raw)
        else:
            raise ValueError(f"unsupported resource kind: {self.kind}")

        normalized_mime = normalize_mime(self.mime)
        if not normalized_mime:
            if self.kind == "http_url":
                normalized_mime = guess_mime_from_http_url(normalized_raw, "")
            elif self.kind == "data_url":
                normalized_mime = extract_mime_from_data_url(normalized_raw, "")

        self.raw = normalized_raw
        self.mime = normalized_mime

    @classmethod
    def from_http_url(cls, raw: str, *, mime: str = "") -> ResourceSpec:
        return cls(kind="http_url", raw=raw, mime=mime)

    @classmethod
    def from_data_url(cls, raw: str, *, mime: str = "") -> ResourceSpec:
        return cls(kind="data_url", raw=raw, mime=mime)

    @classmethod
    def from_base64(cls, raw: str, *, mime: str = "") -> ResourceSpec:
        return cls(kind="base64", raw=raw, mime=mime)

    async def to_base64(self) -> str:
        if self.kind == "http_url":
            blob = await self.convert_to_resource_blob()
            return blob.to_base64()
        if self.kind == "data_url":
            return transfer_data_url_to_base64(self.raw)
        return self.raw

    async def to_data_url(
        self,
        default_mime: str = "application/octet-stream",
    ) -> str:
        if self.kind == "http_url":
            blob = await self.convert_to_resource_blob()
            return blob.to_data_url()
        if self.kind == "data_url":
            return self.raw
        mime = self.mime or normalize_mime(default_mime)
        return transfer_base64_to_data_url(mime, self.raw)

    async def convert_to_resource_blob(
        self,
        *,
        max_bytes: int | None = None,
        default_mime: str = "application/octet-stream",
        default_extension: str = "bin",
        timeout_sec: int = 60,
    ) -> ResourceBlob:

        if self.kind == "http_url":
            res = await get_bytes(url=self.raw, timeout_sec=timeout_sec)
            data = res["data"]
            loaded_mime = res["mime"]
            declared_mime = loaded_mime or self.mime
        elif self.kind == "data_url":
            data, parsed_mime = transfer_data_url_to_bytes(self.raw)
            declared_mime = parsed_mime or self.mime
        elif self.kind == "base64":
            data = decode_base64_payload(self.raw)
            declared_mime = self.mime
        else:
            raise ValueError(f"unsupported resource kind: {self.kind}")

        _ensure_max_bytes(data, max_bytes)
        return ResourceBlob(
            data=data,
            default_mime=declared_mime
            or normalize_mime(default_mime)
            or "application/octet-stream",
            default_extension=default_extension,
        )

    async def convert_to_image_blob(
        self,
        *,
        max_bytes: int | None = None,
        default_mime: str = "application/octet-stream",
        default_extension: str = "bin",
        timeout_sec: int = 60,
    ) -> ImageBlob:
        blob = await self.convert_to_resource_blob(
            max_bytes=max_bytes,
            default_mime=default_mime,
            default_extension=default_extension,
            timeout_sec=timeout_sec,
        )
        return blob.transfer_to_image_blob()
