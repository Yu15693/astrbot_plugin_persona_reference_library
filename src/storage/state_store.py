from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ..utils.log import StructuredLogEmitter
from .keys import CURRENT_IMAGE_MODEL_KEY, PLUGIN_STATE_KEY
from .schema import PluginState

structured_log = StructuredLogEmitter(logger=logging.getLogger(__name__))

# KV 访问函数签名（通过函数注入而非硬编码依赖）：
# - kv_get(key, default) -> 已存值或 default
# - kv_put(key, value) -> 持久化写入
#
# 这样做的目的：
# 1. 与宿主具体 KV 实现解耦（只依赖调用约定）。
# 2. 便于单元测试（可注入 fake/get put）。
# 3. 后续可平滑切换存储后端（只要适配同一签名）。
KVGet = Callable[[str, object], Awaitable[object | None]]
KVPut = Callable[[str, object], Awaitable[None]]
StateValueValidator = Callable[[str], str]


def normalize_image_models(raw_models: object) -> list[str]:
    """规范化模型列表：仅保留去空白后的非空字符串。"""
    if not isinstance(raw_models, list):
        return []
    normalized: list[str] = []
    for item in raw_models:
        candidate = str(item).strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def resolve_current_image_model(image_models: list[str], candidate: object) -> str:
    """根据候选值与模型列表解析当前模型。"""
    if isinstance(candidate, str) and candidate in image_models:
        return candidate
    if image_models:
        return image_models[0]
    return ""


def _normalize_plugin_state(raw_state: object) -> PluginState:
    """将外部输入归一化为 `dict[str, str]` 状态。"""
    if not isinstance(raw_state, dict):
        return {}
    normalized: PluginState = {}
    for key, value in raw_state.items():
        if isinstance(key, str) and isinstance(value, str):
            normalized[key] = value
    return normalized


class PluginStateStore:
    """插件运行时状态管理器（单向：config -> state，state -> KV）。"""

    def __init__(
        self,
        *,
        config: dict[str, object],
        kv_get: KVGet,
        kv_put: KVPut,
    ) -> None:
        # 通过依赖注入接收配置与 KV 访问函数，避免直接耦合宿主实现细节。
        self._config = config
        self._kv_get = kv_get
        self._kv_put = kv_put
        self._state: PluginState = {}
        self._lock = asyncio.Lock()
        self._validators: dict[str, StateValueValidator] = {
            CURRENT_IMAGE_MODEL_KEY: self._validate_current_image_model,
        }

    async def initialize(self) -> PluginState:
        """初始化顺序：先规范化 config，再初始化 state，最后全量写入 KV。"""
        async with self._lock:
            image_models = normalize_image_models(self._config.get("image_models"))
            self._config["image_models"] = list(image_models)

            if not image_models:
                structured_log.warning(
                    "storage.image_models_empty",
                    {
                        "config_keys": sorted(self._config.keys()),
                    },
                )

            self._state = await self._load_from_kv_locked()

            current_image_model = resolve_current_image_model(
                image_models,
                self._state.get(CURRENT_IMAGE_MODEL_KEY, ""),
            )
            self._state[CURRENT_IMAGE_MODEL_KEY] = current_image_model

            await self._sync_to_kv_locked()
            return self._snapshot()

    async def get_value(self, key: str, default: str = "") -> str:
        """读取缓存中的状态值。"""
        async with self._lock:
            value = self._state.get(key, default)
            if isinstance(value, str):
                return value
            return default

    async def set_value(self, key: str, value: str) -> PluginState:
        """写入状态值并同步到 KV。"""
        async with self._lock:
            normalized_value = value.strip()
            validator = self._validators.get(key)
            if validator is not None:
                normalized_value = validator(normalized_value)

            old_value = self._state.get(key)
            if isinstance(old_value, str) and old_value == normalized_value:
                return self._snapshot()

            self._state[key] = normalized_value
            await self._sync_to_kv_locked()
            return self._snapshot()

    async def sync_to_kv(self) -> None:
        """将内存状态全量同步到 KV。"""
        async with self._lock:
            await self._sync_to_kv_locked()

    async def _load_from_kv_locked(self) -> PluginState:
        raw_state = await self._kv_get(PLUGIN_STATE_KEY, {})
        return _normalize_plugin_state(raw_state)

    async def _sync_to_kv_locked(self) -> None:
        await self._kv_put(PLUGIN_STATE_KEY, self._snapshot())

    def _validate_current_image_model(self, value: str) -> str:
        image_models = self._config.get("image_models")
        if value and value not in image_models:
            raise ValueError(f"Unsupported image model: {value}")
        return value

    def _snapshot(self) -> PluginState:
        return dict(self._state)
