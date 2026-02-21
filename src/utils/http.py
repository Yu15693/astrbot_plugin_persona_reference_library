from __future__ import annotations

import asyncio
import json
import time
from typing import Any, TypedDict

import aiohttp

from .errors import PluginErrorCode, PluginException
from .log import logger


class HttpResponse(TypedDict):
    status_code: int
    headers: dict[str, str]
    body: bytes
    elapsed_ms: int


class PostJsonSuccessResponse(TypedDict):
    data: dict[str, Any]
    elapsed_ms: int


class GetBytesSuccessResponse(TypedDict):
    data: bytes
    mime: str
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


async def request(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_sec: int = 60,
    source: str = "Upstream",
) -> HttpResponse:
    if timeout_sec <= 0:
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message="timeout_sec must be > 0.",
            retryable=False,
            detail={
                "source": source,
                "method": method,
                "url": url,
                "timeout_sec": timeout_sec,
            },
        )

    normalized_method = method.strip().upper()
    if not normalized_method:
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message="method must not be empty.",
            retryable=False,
            detail={
                "source": source,
                "url": url,
                "timeout_sec": timeout_sec,
            },
        )

    normalized_headers = headers or {}
    masked_headers = _mask_headers(normalized_headers)
    started_at = time.perf_counter()
    request_error_detail = {
        "source": source,
        "method": normalized_method,
        "url": url,
        "timeout_sec": timeout_sec,
        "headers": masked_headers,
        "payload": payload,
    }
    logger.debug("http.request", request_error_detail)

    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request_kwargs: dict[str, Any] = {"headers": normalized_headers}
            if payload is not None:
                request_kwargs["json"] = payload

            async with session.request(
                normalized_method,
                url,
                **request_kwargs,
            ) as response:
                body = await response.read()
                response_headers = dict(response.headers)

                masked_response_headers = _mask_headers(response_headers)
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                logger.debug(
                    "http.response",
                    {
                        "elapsed_ms": elapsed_ms,
                        "status_code": response.status,
                        "headers": masked_response_headers,
                    },
                )

                # HTTP 错误由状态码判断，保留响应片段用于问题定位
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
                            "body": body.decode("utf-8", errors="replace"),
                        },
                    )

                return {
                    "status_code": response.status,
                    "headers": response_headers,
                    "body": body,
                    "elapsed_ms": elapsed_ms,
                }

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


async def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: int = 60,
    source: str = "Upstream",
) -> PostJsonSuccessResponse:
    response = await request(
        method="POST",
        url=url,
        headers=headers,
        payload=payload,
        timeout_sec=timeout_sec,
        source=source,
    )

    request_error_detail = {
        "source": source,
        "method": "POST",
        "url": url,
        "timeout_sec": timeout_sec,
        "headers": _mask_headers(headers),
        "payload": payload,
    }

    raw_text = response["body"].decode("utf-8", errors="replace")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PluginException(
            code=PluginErrorCode.UPSTREAM_ERROR,
            message=f"{source} returned invalid JSON.",
            retryable=True,
            detail={
                **request_error_detail,
                "elapsed_ms": response["elapsed_ms"],
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
                "elapsed_ms": response["elapsed_ms"],
                "response_type": type(data).__name__,
            },
        )

    logger.debug("http.response.data", data)

    return {
        "data": data,
        "elapsed_ms": response["elapsed_ms"],
    }


async def get_bytes(
    *,
    url: str,
    headers: dict[str, str] | None = None,
    timeout_sec: int = 60,
    source: str = "Upstream",
) -> GetBytesSuccessResponse:
    response = await request(
        method="GET",
        url=url,
        headers=headers,
        timeout_sec=timeout_sec,
        source=source,
    )
    content_type = response["headers"].get("Content-Type", "")
    mime = content_type.split(";", 1)[0].strip().lower()
    return {
        "data": response["body"],
        "mime": mime,
        "elapsed_ms": response["elapsed_ms"],
    }
