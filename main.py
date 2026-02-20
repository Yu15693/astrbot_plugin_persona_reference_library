from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .src.providers import (
    ProviderAdapter,
    build_provider_adapter,
    read_provider_adapter_config,
)
from .src.storage import PluginStateStore
from .src.storage.keys import (
    CONFIG_API_KEY_KEY,
    CONFIG_BASE_URL_KEY,
    CONFIG_IMAGE_MODELS_KEY,
    CONFIG_PROVIDER_KEY,
    CONFIG_TIMEOUT_SEC_KEY,
    CONFIG_TOOL_MODEL_KEY,
    CURRENT_IMAGE_MODEL_KEY,
)


class MyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config=config)
        self.state_store = PluginStateStore(
            config=config,
            kv_get=self.get_kv_data,
            kv_put=self.put_kv_data,
        )
        self.provider_adapter: ProviderAdapter | None = None

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
            }
        )

        return build_provider_adapter(adapter_config)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        #  初始化状态
        await self.state_store.initialize()
        # 初始化供应商适配器
        self.provider_adapter = await self._build_provider_adapter_from_store()

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

    # 注册指令的装饰器。指令名为 helloworld。注册成功后，发送 `/helloworld` 就会触发这个指令，并回复 `你好, {user_name}!`
    @filter.command("helloworld")
    async def helloworld(self, event: AstrMessageEvent):
        """这是一个 hello world 指令"""  # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
        user_name = event.get_sender_name()
        message_str = event.message_str  # 用户发的纯文本消息字符串
        message_chain = (
            event.get_messages()
        )  # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)
        yield event.plain_result(
            f"Hello, {user_name}, 你发了 {message_str}!"
        )  # 发送一条纯文本消息

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
