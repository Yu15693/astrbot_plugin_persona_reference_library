from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .src.providers import (
    ProviderAdapter,
    build_provider_adapter,
    read_provider_adapter_config,
)
from .src.providers.schema import ImageGenerateInput
from .src.providers.utils import build_image_generate_render_result
from .src.storage import PluginStateStore
from .src.storage.keys import (
    CONFIG_API_KEY_KEY,
    CONFIG_BASE_URL_KEY,
    CONFIG_IMAGE_MODELS_KEY,
    CONFIG_PROVIDER_KEY,
    CONFIG_SAVE_IMAGE_FORMAT_KEY,
    CONFIG_SHOW_IMAGE_GENERATE_DETAILS_KEY,
    CONFIG_TIMEOUT_SEC_KEY,
    CONFIG_TOOL_MODEL_KEY,
    CURRENT_IMAGE_MODEL_KEY,
)
from .src.tools.draw_args import parse_draw_args
from .src.tools.image import extract_images_from_event
from .src.utils.args import extract_command_args
from .src.utils.log import logger


class MyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config=config)
        self.state_store = PluginStateStore(
            config=config,
            kv_get=self.get_kv_data,
            kv_put=self.put_kv_data,
        )
        self.provider_adapter: ProviderAdapter

    async def _build_provider_adapter_from_store(self) -> ProviderAdapter:
        image_model = await self.state_store.get_value(
            CURRENT_IMAGE_MODEL_KEY,
            "",
        )

        adapter_config = read_provider_adapter_config(
            {
                CONFIG_PROVIDER_KEY: self.state_store.get_config_value(
                    CONFIG_PROVIDER_KEY
                ),
                CONFIG_BASE_URL_KEY: self.state_store.get_config_value(
                    CONFIG_BASE_URL_KEY
                ),
                CONFIG_API_KEY_KEY: self.state_store.get_config_value(
                    CONFIG_API_KEY_KEY
                ),
                CONFIG_TIMEOUT_SEC_KEY: self.state_store.get_config_value(
                    CONFIG_TIMEOUT_SEC_KEY
                ),
                "image_model": image_model,
                CONFIG_TOOL_MODEL_KEY: self.state_store.get_config_value(
                    CONFIG_TOOL_MODEL_KEY
                ),
                CONFIG_SAVE_IMAGE_FORMAT_KEY: self.state_store.get_config_value(
                    CONFIG_SAVE_IMAGE_FORMAT_KEY, "jpg"
                ),
            }
        )

        return build_provider_adapter(adapter_config)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        #  初始化状态
        await self.state_store.initialize()
        # 初始化供应商适配器
        self.provider_adapter = await self._build_provider_adapter_from_store()
        # 注册工具
        show_image_generate_details = bool(
            self.state_store.get_config_value(CONFIG_SHOW_IMAGE_GENERATE_DETAILS_KEY)
        )
        tool = self.provider_adapter.get_image_generate_tool(
            show_image_generate_details=show_image_generate_details
        )
        self.context.add_llm_tools(tool)
        # add_llm_tools 使用 tool.__module__ 推断归属；FunctionTool 实例默认模块不在插件下，
        # 这里显式修正到插件主模块，确保禁用/卸载时可被正确识别与移除。
        tool.handler_module_path = self.__class__.__module__

    @filter.command_group("prl")
    def prl(self) -> None:
        """插件指令组"""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @prl.command("models")
    async def prl_models(self, event: AstrMessageEvent):
        """列出当前生图模型和全部可用生图模型。"""
        current_model = await self.state_store.get_value(CURRENT_IMAGE_MODEL_KEY, "")
        image_models = self.state_store.get_config_value(CONFIG_IMAGE_MODELS_KEY, [])
        assert isinstance(image_models, list)

        lines = [
            "生图模型管理",
            f"当前模型：{current_model or '未设置'}",
            "可用模型列表：",
        ]
        if not image_models:
            lines.append("- （空）")
        else:
            for index, model_name in enumerate(image_models, start=1):
                lines.append(f"{index}. {model_name}")

        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @prl.command("use")
    async def prl_use(self, event: AstrMessageEvent, model_name: str = ""):
        """切换当前生图模型"""
        target_model = model_name.strip()
        if not target_model:
            yield event.plain_result("模型名不能为空。")
            return

        try:
            state = await self.state_store.set_value(
                CURRENT_IMAGE_MODEL_KEY, target_model
            )
        except ValueError:
            yield event.plain_result(
                "模型不在配置项 image_models 中，请先执行 /prl models 查看可用模型。"
            )
            return

        applied_model = state.get(CURRENT_IMAGE_MODEL_KEY, target_model)
        self.provider_adapter.image_model = applied_model

        yield event.plain_result(f"已切换生图模型为：{applied_model}")

    @prl.command("draw")
    async def prl_draw(self, event: AstrMessageEvent):
        """
        根据提示词生成图片，可选 `ratio=` 与 `size=` 参数，ratio 为图片比例，size 为图片分辨率。
        消息附带的图片会作为参考图，指令输入格式如
        /prl draw ratio=16:9 size=1K 一个可爱的动漫女孩 【同时上传多张参考图】
        """
        draw_args = extract_command_args(event.message_str, ("prl", "draw"))
        ratio, size, prompt = parse_draw_args(draw_args)
        if not prompt:
            yield event.plain_result(
                "提示词不能为空。用法：/prl draw [ratio=16:9] [size=1K] <prompt>"
            )
            return

        reference_images = await extract_images_from_event(event)
        yield event.plain_result(
            "\n".join(
                [
                    "生图任务开始",
                    f"ratio={ratio or '(默认)'}  size={size or '(默认)'}  参考图={len(reference_images)} 张",
                    f"prompt={prompt}",
                ]
            )
        )

        payload = ImageGenerateInput(
            prompt=prompt,
            aspect_ratio=ratio,
            image_size=size,
            reference_images=reference_images,
        )

        try:
            output = await self.provider_adapter.image_generate(payload)
        except Exception as exc:
            logger.exception("prl draw failed: %s", exc)
            yield event.plain_result("生图失败：上游请求失败。")
            return

        render_result = build_image_generate_render_result(
            event,
            output,
            requested_count=payload.count,
        )
        yield event.plain_result(render_result.detail_text)

        for result in render_result.send_results:
            yield result

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
        # 兜底移除动态注册的工具，避免插件卸载后残留。
        self.context.provider_manager.llm_tools.remove_func(
            self.provider_adapter.image_generate_tool_name
        )
