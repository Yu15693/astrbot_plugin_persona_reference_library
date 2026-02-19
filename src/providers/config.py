from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from .schema import ProviderAdapterConfig


def _require_mapping(raw_config: Any) -> Mapping[str, Any]:
    """确保原始配置是键值映射。"""
    if not isinstance(raw_config, Mapping):
        raise TypeError("Provider adapter config must be a mapping object.")
    return raw_config


def _require_keys(cfg: Mapping[str, Any], required: tuple[str, ...]) -> None:
    """仅校验必填字段是否存在。"""
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing required provider config keys: {', '.join(missing)}")


def read_provider_adapter_config(raw_config: Any) -> ProviderAdapterConfig:
    """读取并返回适配器配置；当前仅校验字段存在性。"""
    cfg = _require_mapping(raw_config)
    required = tuple(f.name for f in fields(ProviderAdapterConfig))
    _require_keys(cfg, required)
    payload = {key: cfg[key] for key in required}
    return ProviderAdapterConfig(**payload)
