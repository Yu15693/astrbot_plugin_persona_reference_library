from __future__ import annotations

from io import BytesIO
from pathlib import Path

from .codec import encode_base64_payload, transfer_base64_to_data_url
from .mime import sniff_file_type


class ResourceBlob:
    data: bytes
    """文件字节数据"""
    mime: str
    """文件内容类型标识，嗅探结果兜底"""
    extension: str
    """文件扩展名（不带点），嗅探结果兜底"""

    def __init__(
        self,
        *,
        data: bytes,
        default_mime: str = "application/octet-stream",
        default_extension: str = "bin",
    ) -> None:
        self.data = data
        normalized_default_extension = (
            default_extension.strip().lower().removeprefix(".")
        )
        if not normalized_default_extension:
            normalized_default_extension = "bin"

        sniffed_mime, sniffed_extension = sniff_file_type(
            self.data, default_mime, normalized_default_extension
        )
        self.mime = sniffed_mime
        self.extension = sniffed_extension

    def to_base64(self) -> str:
        return encode_base64_payload(self.data)

    def to_data_url(self) -> str:
        return transfer_base64_to_data_url(self.mime, self.to_base64())

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.data)
        return output_path

    def transfer_to_image_blob(self) -> ImageBlob:
        return ImageBlob(
            data=self.data,
            default_mime=self.mime,
            default_extension=self.extension,
        )


class ImageBlob(ResourceBlob):
    def __init__(
        self,
        *,
        data: bytes,
        default_mime: str = "application/octet-stream",
        default_extension: str = "bin",
    ) -> None:
        super().__init__(
            data=data,
            default_mime=default_mime,
            default_extension=default_extension,
        )
        if not self.mime.startswith("image/"):
            raise ValueError("image mime is not valid.")

    def compress_to_jpg(self, quality: int = 85) -> ImageBlob:
        """无副作用，将原对象压缩为JPG，返回新对象"""

        if quality < 1 or quality > 95:
            raise ValueError("quality must be in [1, 95].")

        from PIL import Image

        with Image.open(BytesIO(self.data)) as image:
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                alpha = image.convert("RGBA")
                background = Image.new("RGB", alpha.size, (255, 255, 255))
                background.paste(alpha, mask=alpha.split()[-1])
                rgb_image = background
            else:
                rgb_image = image.convert("RGB")

            output = BytesIO()
            rgb_image.save(output, format="JPEG", quality=quality, optimize=True)
            return ImageBlob(
                data=output.getvalue(),
                default_mime="image/jpeg",
                default_extension="jpg",
            )
