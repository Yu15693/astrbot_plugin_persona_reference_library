from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.providers.config import read_provider_adapter_config
from src.providers.factory import build_provider_adapter
from src.providers.openrouter import OpenRouterAdapter
from src.providers.schema import ImageGenerateInput, ImageGenerateOutput
from src.utils.id import generate_id
from src.utils.io import (
    decode_data_url,
    download_http_resource,
    save_file,
)
from src.utils.paths import PLUGIN_ROOT
from tests.utils.test_env import (
    is_env_enabled,
    required_env,
)

OPENROUTER_VALID_ASPECT_RATIOS = {
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "2:3",
    "3:2",
}
OPENROUTER_VALID_IMAGE_SIZES = {"1K", "2K", "4K"}


def _build_live_adapter() -> OpenRouterAdapter:
    config = read_provider_adapter_config(
        {
            "provider": "openrouter",
            "base_url": required_env("OPENROUTER_BASE_URL"),
            "api_key": required_env("OPENROUTER_API_KEY"),
            "timeout_sec": int(required_env("OPENROUTER_TIMEOUT_SEC")),
            "image_model": required_env("OPENROUTER_IMAGE_MODEL"),
            "tool_model": required_env("OPENROUTER_TOOL_MODEL"),
        }
    )
    adapter = build_provider_adapter(config)
    assert isinstance(adapter, OpenRouterAdapter)
    return adapter


def _assert_openrouter_image_config(aspect_ratio: str, image_size: str) -> None:
    assert (
        aspect_ratio in OPENROUTER_VALID_ASPECT_RATIOS
    ), f"Unsupported aspect_ratio for OpenRouter test: {aspect_ratio}"
    assert (
        image_size in OPENROUTER_VALID_IMAGE_SIZES
    ), f"Unsupported image_size for OpenRouter test: {image_size}"


async def _save_output_images(case_name: str, output: ImageGenerateOutput) -> Path:
    target_dir = PLUGIN_ROOT / "tmp"
    target_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "case_name": case_name,
        "items": items,
        "warnings": output.warnings,
    }

    for index, image in enumerate(output.images, start=1):
        item: dict[str, object] = {
            "index": index,
            "kind": image.kind,
        }
        if image.kind == "data_url":
            content, suffix = decode_data_url(image.value)
            output_path = target_dir / f"{generate_id()}{suffix}"
            save_file(output_path, content)
            item["filename"] = output_path.name
            item["status"] = "saved"
            item["source"] = "data_url"
            item["source_length"] = len(image.value)
            items.append(item)
            continue
        if image.kind == "http_url":
            item["source_url"] = image.value
            try:
                content, suffix = await download_http_resource(image.value, timeout_sec=30)
                output_path = target_dir / f"{generate_id()}{suffix}"
                save_file(output_path, content)
                item["filename"] = output_path.name
                item["status"] = "saved"
            except Exception as exc:
                item["status"] = "download_failed"
                item["error"] = str(exc)
            items.append(item)
            continue
        output_path = target_dir / f"{generate_id()}.txt"
        save_file(output_path, image.value)
        item["filename"] = output_path.name
        item["status"] = "saved_as_text_fallback"
        items.append(item)

    metadata_path = target_dir / f"{generate_id()}.metadata.json"
    save_file(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    print(f"[live] metadata saved ({case_name}): {metadata_path}")
    return target_dir


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_image_generate_live_smoke() -> None:
    """验证：可选的 OpenRouter 真实生图冒烟测试。"""
    if not is_env_enabled("OPENROUTER_RUN_LIVE_TEST"):
        pytest.skip("Live OpenRouter test is disabled. Set OPENROUTER_RUN_LIVE_TEST=1 to enable.")

    adapter = _build_live_adapter()
    aspect_ratio = required_env("OPENROUTER_TEST_ASPECT_RATIO")
    image_size = required_env("OPENROUTER_TEST_IMAGE_SIZE")
    _assert_openrouter_image_config(aspect_ratio, image_size)
    prompt = required_env("OPENROUTER_TEST_PROMPT")
    payload = ImageGenerateInput(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        count=1,
    )

    result = await adapter.image_generate(payload)
    if result.warnings:
        print(f"[live] result warnings: {json.dumps(result.warnings, ensure_ascii=False)}")
    assert result.images
    assert result.images[0].kind in {"http_url", "data_url"}
    await _save_output_images("smoke", result)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_image_generate_live_resolution_and_clarity() -> None:
    """验证：使用不同分辨率与清晰度要求时可成功生图。"""
    if not is_env_enabled("OPENROUTER_RUN_LIVE_TEST"):
        pytest.skip("Live OpenRouter test is disabled. Set OPENROUTER_RUN_LIVE_TEST=1 to enable.")

    adapter = _build_live_adapter()
    aspect_ratio = required_env("OPENROUTER_TEST_HD_ASPECT_RATIO")
    image_size = required_env("OPENROUTER_TEST_HD_IMAGE_SIZE")
    _assert_openrouter_image_config(aspect_ratio, image_size)
    prompt = required_env("OPENROUTER_TEST_HD_PROMPT")
    payload = ImageGenerateInput(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        count=1,
    )

    result = await adapter.image_generate(payload)
    if result.warnings:
        print(f"[live] result warnings: {json.dumps(result.warnings, ensure_ascii=False)}")
    assert result.images
    await _save_output_images("resolution_and_clarity", result)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_image_generate_live_prompt_two_images_and_n2() -> None:
    """验证：prompt 明确要求 2 图且 n=2 时请求参数与返回行为符合预期。"""
    if not is_env_enabled("OPENROUTER_RUN_LIVE_TEST"):
        pytest.skip("Live OpenRouter test is disabled. Set OPENROUTER_RUN_LIVE_TEST=1 to enable.")

    adapter = _build_live_adapter()
    aspect_ratio = required_env("OPENROUTER_TEST_ASPECT_RATIO")
    image_size = required_env("OPENROUTER_TEST_IMAGE_SIZE")
    _assert_openrouter_image_config(aspect_ratio, image_size)
    prompt = required_env("OPENROUTER_TEST_TWO_IMAGES_PROMPT")
    payload = ImageGenerateInput(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        count=2,
    )
    request_payload, _ = adapter._build_image_generate_payload(payload)
    assert request_payload.get("n") == 2

    result = await adapter.image_generate(payload)
    if result.warnings:
        print(f"[live] result warnings: {json.dumps(result.warnings, ensure_ascii=False)}")
    assert result.images
    if len(result.images) != 2:
        assert any("different from requested 2" in warning for warning in result.warnings)
    await _save_output_images("two_images_n2", result)
