from __future__ import annotations

import pytest
from src.providers.config import read_provider_adapter_config
from src.providers.schema import ProviderAdapterConfig


def _valid_raw_config() -> dict[str, object]:
    return {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "timeout_sec": 30,
        "image_model": "test-image-model",
        "tool_model": "test-tool-model",
        "save_image_format": "png",
    }


def test_read_provider_adapter_config_success() -> None:
    """验证：合法映射可以成功构造 ProviderAdapterConfig。"""
    config = read_provider_adapter_config(_valid_raw_config())

    assert isinstance(config, ProviderAdapterConfig)
    assert config.provider == "openrouter"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.api_key == "test-key"
    assert config.timeout_sec == 30
    assert config.image_model == "test-image-model"
    assert config.tool_model == "test-tool-model"
    assert config.save_image_format == "png"


def test_read_provider_adapter_config_success_ignores_extra_keys() -> None:
    """验证：原始配置包含额外字段时会被忽略，仅保留必需字段。"""
    raw_config = _valid_raw_config()
    raw_config["extra"] = "ignored"

    config = read_provider_adapter_config(raw_config)

    assert isinstance(config, ProviderAdapterConfig)
    assert not hasattr(config, "extra")


@pytest.mark.parametrize("missing_key", ["api_key", "save_image_format"])
def test_read_provider_adapter_config_missing_required_key(missing_key: str) -> None:
    """验证：缺少任一必填字段时抛出 KeyError。"""
    raw_config = _valid_raw_config()
    raw_config.pop(missing_key)

    with pytest.raises(KeyError, match="Missing required provider config keys"):
        read_provider_adapter_config(raw_config)
