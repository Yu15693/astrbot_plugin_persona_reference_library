from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from src.providers.config import read_provider_adapter_config
from src.providers.factory import build_provider_adapter
from src.providers.openrouter import OpenRouterAdapter
from src.providers.schema import ImageGenerateInput, ImageGenerateOutput
from src.utils.id import generate_id
from src.utils.io import save_file
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
SAVE_COMPRESS_IMAGE = True
SAVE_JPEG_QUALITY = 85


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


async def _save_output_images(
    case_name: str,
    output: ImageGenerateOutput,
) -> Path:
    base_dir = PLUGIN_ROOT / "tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_id()
    target_dir = base_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "run_id": run_id,
        "case_name": case_name,
        "compress_image": SAVE_COMPRESS_IMAGE,
        "jpeg_quality": SAVE_JPEG_QUALITY,
        "inference": asdict(output.metadata),
        "items": items,
        "warnings": output.warnings,
    }

    for index, image in enumerate(output.images, start=1):
        image_blob = await image.convert_to_image_blob(timeout_sec=30)
        output_blob = (
            image_blob.compress_to_jpg(quality=SAVE_JPEG_QUALITY)
            if SAVE_COMPRESS_IMAGE
            else image_blob
        )
        compressed = output_blob is not image_blob
        output_path = target_dir / f"{index}.{output_blob.extension}"
        output_blob.save(output_path)
        item = {
            "index": index,
            "kind": image.kind,
            "mime": image.mime,
            "filename": output_path.name,
            "compressed": compressed,
        }
        items.append(item)

    metadata_path = target_dir / "metadata.json"
    save_file(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    print(f"[live] artifacts saved ({case_name}): {target_dir}")
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
    assert result.images[0].kind in {"http_url", "data_url", "base64"}
    await _save_output_images(
        "smoke",
        result,
    )


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
    await _save_output_images(
        "resolution_and_clarity",
        result,
    )


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
    request_payload, _ = await adapter._build_image_generate_payload(
        payload,
        image_model=adapter.image_model,
    )
    assert request_payload.get("n") == 2

    result = await adapter.image_generate(payload)
    if result.warnings:
        print(f"[live] result warnings: {json.dumps(result.warnings, ensure_ascii=False)}")
    assert result.images
    if len(result.images) != 2:
        assert any("different from requested 2" in warning for warning in result.warnings)
    await _save_output_images(
        "two_images_n2",
        result,
    )
