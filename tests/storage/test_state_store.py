from __future__ import annotations

import pytest
from src.storage import PluginStateStore
from src.storage.keys import CURRENT_IMAGE_MODEL_KEY, PLUGIN_STATE_KEY
from src.storage.state_store import normalize_image_models


class _FakeKV:
    def __init__(self, initial: dict[str, object] | None = None) -> None:
        self.store = initial or {}
        self.put_calls: list[tuple[str, object]] = []

    async def get(self, key: str, default: object) -> object | None:
        value = self.store.get(key, default)
        return value

    async def put(self, key: str, value: object) -> None:
        self.store[key] = value
        self.put_calls.append((key, value))


def test_normalize_image_models() -> None:
    """验证：模型列表会被去空白并过滤空值。"""
    assert normalize_image_models([" model-a ", "", "  ", "model-b", 123]) == [
        "model-a",
        "model-b",
        "123",
    ]


@pytest.mark.asyncio
async def test_initialize_fallback_to_first_when_kv_invalid() -> None:
    """验证：KV 中模型不在列表时，回退到列表首项并持久化。"""
    config: dict[str, object] = {"image_models": ["model-a", "model-b"]}
    kv = _FakeKV(
        {
            PLUGIN_STATE_KEY: {
                CURRENT_IMAGE_MODEL_KEY: "invalid-model",
            }
        }
    )
    store = PluginStateStore(config=config, kv_get=kv.get, kv_put=kv.put)

    state = await store.initialize()

    assert state[CURRENT_IMAGE_MODEL_KEY] == "model-a"
    assert kv.store[PLUGIN_STATE_KEY] == {CURRENT_IMAGE_MODEL_KEY: "model-a"}
    assert kv.put_calls == [(PLUGIN_STATE_KEY, {CURRENT_IMAGE_MODEL_KEY: "model-a"})]


@pytest.mark.asyncio
async def test_initialize_keeps_existing_model_without_extra_write() -> None:
    """验证：KV 中模型合法时会沿用，并执行一次全量状态同步。"""
    config: dict[str, object] = {"image_models": ["model-a", "model-b"]}
    kv = _FakeKV(
        {
            PLUGIN_STATE_KEY: {
                CURRENT_IMAGE_MODEL_KEY: "model-b",
            }
        }
    )
    store = PluginStateStore(config=config, kv_get=kv.get, kv_put=kv.put)

    state = await store.initialize()

    assert state[CURRENT_IMAGE_MODEL_KEY] == "model-b"
    assert kv.put_calls == [(PLUGIN_STATE_KEY, {CURRENT_IMAGE_MODEL_KEY: "model-b"})]


@pytest.mark.asyncio
async def test_initialize_empty_models_resets_to_empty_string() -> None:
    """验证：当模型列表为空时，当前模型会被重置为空字符串。"""
    config: dict[str, object] = {"image_models": []}
    kv = _FakeKV(
        {
            PLUGIN_STATE_KEY: {
                CURRENT_IMAGE_MODEL_KEY: "model-a",
            }
        }
    )
    store = PluginStateStore(config=config, kv_get=kv.get, kv_put=kv.put)

    state = await store.initialize()

    assert state[CURRENT_IMAGE_MODEL_KEY] == ""
    assert kv.store[PLUGIN_STATE_KEY] == {CURRENT_IMAGE_MODEL_KEY: ""}
    assert kv.put_calls == [(PLUGIN_STATE_KEY, {CURRENT_IMAGE_MODEL_KEY: ""})]


@pytest.mark.asyncio
async def test_set_current_image_model_success_and_validation() -> None:
    """验证：通过通用 set_value 可切换模型；非法模型会抛错。"""
    config: dict[str, object] = {"image_models": ["model-a", "model-b"]}
    kv = _FakeKV()
    store = PluginStateStore(config=config, kv_get=kv.get, kv_put=kv.put)
    await store.initialize()

    state = await store.set_value(CURRENT_IMAGE_MODEL_KEY, " model-b ")
    assert state[CURRENT_IMAGE_MODEL_KEY] == "model-b"
    assert kv.store[PLUGIN_STATE_KEY] == {CURRENT_IMAGE_MODEL_KEY: "model-b"}

    with pytest.raises(ValueError, match="Unsupported image model"):
        await store.set_value(CURRENT_IMAGE_MODEL_KEY, "model-c")


@pytest.mark.asyncio
async def test_sync_to_kv_persists_full_state_snapshot() -> None:
    """验证：sync_to_kv 会把当前内存状态全量写入 KV。"""
    config: dict[str, object] = {"image_models": ["model-a", "model-b"]}
    kv = _FakeKV()
    store = PluginStateStore(config=config, kv_get=kv.get, kv_put=kv.put)
    await store.initialize()
    await store.set_value("custom_key", "custom_value")
    kv.put_calls.clear()

    await store.sync_to_kv()

    assert kv.put_calls == [
        (
            PLUGIN_STATE_KEY,
            {
                CURRENT_IMAGE_MODEL_KEY: "model-a",
                "custom_key": "custom_value",
            },
        )
    ]
