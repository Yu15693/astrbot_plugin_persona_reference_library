from __future__ import annotations


def normalize_mime(value: str) -> str:
    return value.strip().lower()


def normalize_base64_payload(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("base64://"):
        normalized = normalized.removeprefix("base64://")
    normalized = "".join(normalized.split())
    if not normalized:
        raise ValueError("base64 payload is empty.")
    return normalized
