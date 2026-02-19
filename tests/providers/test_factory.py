from __future__ import annotations

import pytest

from src.providers.factory import build_provider_adapter
from src.providers.openrouter import OpenRouterAdapter
from src.providers.schema import ProviderAdapterConfig


def _make_config(provider: str = "openrouter") -> ProviderAdapterConfig:
    return ProviderAdapterConfig(
        provider=provider,
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        timeout_sec=45,
        image_model="test-image-model",
        tool_model="test-tool-model",
    )


def test_build_provider_adapter_success() -> None:
    """验证：openrouter 配置可以成功创建适配器并正确映射字段。"""
    config = _make_config()

    adapter = build_provider_adapter(config)

    assert isinstance(adapter, OpenRouterAdapter)
    assert adapter.provider == "openrouter"
    assert adapter.base_url == config.base_url
    assert adapter.api_key == config.api_key
    assert adapter.timeout_sec == config.timeout_sec
    assert adapter.image_model == config.image_model
    assert adapter.tool_model == config.tool_model


def test_build_provider_adapter_unsupported_provider() -> None:
    """验证：未注册的 provider 会抛出 ValueError。"""
    config = _make_config(provider="unknown-provider")

    with pytest.raises(ValueError, match="Unsupported provider"):
        build_provider_adapter(config)
