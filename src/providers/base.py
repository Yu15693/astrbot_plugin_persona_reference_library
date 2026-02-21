from __future__ import annotations

from abc import ABC, abstractmethod

from astrbot.api import FunctionTool

from .schema import ImageGenerateInput, ImageGenerateOutput


class ProviderAdapter(ABC):
    provider: str
    image_model: str
    tool_model: str
    save_image_format: str
    image_generate_tool_name = "prl_image_generate"

    @abstractmethod
    async def image_generate(
        self, payload: ImageGenerateInput
    ) -> ImageGenerateOutput: ...

    @abstractmethod
    def get_image_generate_tool(
        self,
        *,
        show_image_generate_details: bool = True,
    ) -> FunctionTool: ...
