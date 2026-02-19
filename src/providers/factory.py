from __future__ import annotations

from collections.abc import Callable

from .base import ProviderAdapter
from .config import ProviderAdapterConfig
from .openrouter import OpenRouterAdapter

AdapterBuilder = Callable[[ProviderAdapterConfig], ProviderAdapter]


def _build_openrouter(config: ProviderAdapterConfig) -> ProviderAdapter:
    return OpenRouterAdapter(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout_sec=config.timeout_sec,
        image_model=config.image_model,
        tool_model=config.tool_model,
    )


_ADAPTER_BUILDERS: dict[str, AdapterBuilder] = {
    "openrouter": _build_openrouter,
}


def build_provider_adapter(config: ProviderAdapterConfig) -> ProviderAdapter:
    provider = config.provider.strip().lower()
    builder = _ADAPTER_BUILDERS.get(provider)
    if builder is not None:
        return builder(config)
    raise ValueError(f"Unsupported provider: {config.provider}")
