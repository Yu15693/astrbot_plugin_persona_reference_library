from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, TypedDict

import aiohttp

from .errors import PluginErrorCode, PluginException
from .log import StructuredLogEmitter

logger = logging.getLogger(__name__)
structured_log = StructuredLogEmitter(logger=logger)


class PostJsonSuccessResponse(TypedDict):
    data: dict[str, Any]
    elapsed_ms: int


def _mask_headers(headers: dict[str, str]) -> dict[str, str]:
    secret_keys = {"authorization", "cookie", "set-cookie", "x-api-key"}
    masked: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in secret_keys:
            masked[key] = "<redacted>"
            continue
        masked[key] = value
    return masked


async def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: int = 30,
    source: str = "Upstream",
) -> PostJsonSuccessResponse:
    """发送 JSON POST 请求并返回结果对象。

    约定：
    - 传输层错误映射为 `NETWORK_ERROR/TIMEOUT`
    - 非 2xx HTTP 响应映射为 `UPSTREAM_ERROR`
    - 成功响应必须是 JSON object（dict）
    - 成功返回结构：`{"data": <json_object>, "elapsed_ms": <int>}`
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

    masked_headers = _mask_headers(headers)
    started_at = time.perf_counter()
    request_error_detail = {
        "source": source,
        "url": url,
        "timeout_sec": timeout_sec,
        "headers": masked_headers,
        "payload": payload,
    }
    structured_log.debug("http.request", request_error_detail)

    # 使用 total timeout，覆盖连接、读写和响应等待总耗时。
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                raw_text = await response.text()
                masked_response_headers = _mask_headers(dict(response.headers))
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                structured_log.debug(
                    "http.response",
                    {
                        "elapsed_ms": elapsed_ms,
                        "status_code": response.status,
                        "headers": masked_response_headers,
                        "body": raw_text,
                    },
                )
                # HTTP 错误由状态码判断，保留响应片段用于问题定位。
                if response.status >= 400:
                    raise PluginException(
                        code=PluginErrorCode.UPSTREAM_ERROR,
                        message=f"{source} HTTP {response.status}",
                        retryable=(response.status >= 500 or response.status == 429),
                        detail={
                            **request_error_detail,
                            "elapsed_ms": elapsed_ms,
                            "status_code": response.status,
                            "headers": masked_response_headers,
                            "body": raw_text,
                        },
                    )

    except asyncio.TimeoutError as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        raise PluginException(
            code=PluginErrorCode.TIMEOUT,
            message=f"{source} request timed out.",
            retryable=True,
            detail={**request_error_detail, "elapsed_ms": elapsed_ms},
        ) from exc
    except aiohttp.ClientError as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        raise PluginException(
            code=PluginErrorCode.NETWORK_ERROR,
            message=f"{source} request failed.",
            retryable=True,
            detail={
                **request_error_detail,
                "elapsed_ms": elapsed_ms,
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
            detail={
                **request_error_detail,
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                "body": raw_text,
            },
        ) from exc

    if not isinstance(data, dict):
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message=f"{source} response must be a JSON object.",
            retryable=True,
            detail={
                **request_error_detail,
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                "response_type": type(data).__name__,
            },
        )

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    result = {
        "data": data,
        "elapsed_ms": elapsed_ms,
    }

    structured_log.debug(
        "http.response",
        result,
    )

    return result
