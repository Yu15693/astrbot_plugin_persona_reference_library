from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from .errors import PluginErrorCode, PluginException


def _mask_headers(headers: dict[str, str]) -> dict[str, str]:
    masked: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "cookie"}:
            masked[key] = "<redacted>"
            continue
        masked[key] = value
    return masked


def _summarize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _summarize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        limit = 5
        items = [_summarize_payload(item) for item in value[:limit]]
        if len(value) > limit:
            items.append(f"<+{len(value) - limit} items>")
        return items
    if isinstance(value, str):
        if value.startswith("data:"):
            return f"<data-url len={len(value)}>"
        if len(value) > 200:
            return f"{value[:200]}...(truncated)"
    return value


async def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: int = 30,
    source: str = "Upstream",
) -> dict[str, Any]:
    """发送 JSON POST 请求并返回 JSON 对象。

    约定：
    - 传输层错误映射为 `NETWORK_ERROR/TIMEOUT`
    - 非 2xx HTTP 响应映射为 `UPSTREAM_ERROR`
    - 成功响应必须是 JSON object（dict）
    """
    if timeout_sec <= 0:
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message="timeout_sec must be > 0.",
            retryable=False,
            detail={
                "source": source,
                "url": url,
                "timeout_sec": timeout_sec,
            },
        )

    request_detail = {
        "source": source,
        "url": url,
        "timeout_sec": timeout_sec,
        "headers": _mask_headers(headers),
        "payload": _summarize_payload(payload),
    }

    # 使用 total timeout，覆盖连接、读写和响应等待总耗时。
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                raw_text = await response.text()
                # HTTP 错误由状态码判断，保留响应片段用于问题定位。
                if response.status >= 400:
                    raise PluginException(
                        code=PluginErrorCode.UPSTREAM_ERROR,
                        message=f"{source} HTTP {response.status}: {raw_text[:400]}",
                        retryable=(response.status >= 500 or response.status == 429),
                        detail={
                            **request_detail,
                            "status_code": response.status,
                            "response_excerpt": raw_text[:400],
                        },
                    )
    except asyncio.TimeoutError as exc:
        raise PluginException(
            code=PluginErrorCode.TIMEOUT,
            message=f"{source} request timed out.",
            retryable=True,
            detail=request_detail,
        ) from exc
    except aiohttp.ClientError as exc:
        raise PluginException(
            code=PluginErrorCode.NETWORK_ERROR,
            message=f"{source} request failed.",
            retryable=True,
            detail={
                **request_detail,
                "client_error": str(exc),
                "client_error_type": type(exc).__name__,
            },
        ) from exc

    # 网络链路成功后再解析 JSON，便于区分“传输错误”与“响应格式错误”。
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message=f"{source} returned invalid JSON.",
            retryable=True,
            detail={**request_detail, "response_excerpt": raw_text[:400]},
        ) from exc

    if not isinstance(data, dict):
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message=f"{source} response must be a JSON object.",
            retryable=True,
            detail={**request_detail, "response_type": type(data).__name__},
        )
    return data
