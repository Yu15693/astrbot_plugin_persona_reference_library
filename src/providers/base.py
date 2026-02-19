from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import ImageGenerateInput, ImageGenerateOutput


class ProviderAdapter(ABC):
    provider: str
    image_model: str
    tool_model: str

    @abstractmethod
    async def image_generate(
        self, payload: ImageGenerateInput
    ) -> ImageGenerateOutput: ...
