from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .src.providers import (
    ProviderAdapter,
    build_provider_adapter,
    read_provider_adapter_config,
)
from .src.storage import PluginStateStore
from .src.storage.keys import CURRENT_IMAGE_MODEL_KEY


class MyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config=config)
        self.plugin_config = config
        self.state_store = PluginStateStore(
            config=config,
            kv_get=self.get_kv_data,
            kv_put=self.put_kv_data,
        )
        self.provider_adapter: ProviderAdapter | None = None

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        #  初始化状态
        await self.state_store.initialize()

        # 初始化供应商适配器
        current_image_model = await self.state_store.get_value(
            CURRENT_IMAGE_MODEL_KEY,
            "",
        )
        adapter_config = read_provider_adapter_config(
            {
                "provider": self.plugin_config["provider"],
                "base_url": self.plugin_config["base_url"],
                "api_key": self.plugin_config["api_key"],
                "timeout_sec": self.plugin_config["timeout_sec"],
                "image_model": current_image_model,
                "tool_model": self.plugin_config["tool_model"],
            }
        )
        self.provider_adapter = build_provider_adapter(adapter_config)

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
