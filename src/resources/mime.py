from __future__ import annotations

import mimetypes
from urllib.parse import urlparse

import filetype

from .codec import parse_data_url_header
from .normalize import normalize_mime


def guess_mime_from_http_url(url: str, default_mime: str = "") -> str:
    parsed = urlparse(url.strip())
    guessedMime, _ = mimetypes.guess_type(parsed.path)
    return guessedMime or default_mime


def extract_mime_from_data_url(data_url: str, default_mime: str = "") -> str:
    normalized_default_mime = normalize_mime(default_mime)
    try:
        header = parse_data_url_header(data_url)
    except Exception:
        return normalized_default_mime
    return header.mime or normalized_default_mime


def sniff_file_type(
    data: bytes,
    default_mime: str = "application/octet-stream",
    default_extension: str = "bin",
) -> tuple[str, str]:
    normalized_default = normalize_mime(default_mime) or "application/octet-stream"
    guessedType = filetype.guess(data)
    mime = getattr(guessedType, "mime", "") or normalized_default
    extension = getattr(guessedType, "extension", "") or default_extension
    return (mime, extension)
