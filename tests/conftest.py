from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

bootstrap_plugin_root = Path(__file__).resolve().parents[1]
if str(bootstrap_plugin_root) not in sys.path:
    # 将插件根目录放到导入搜索路径最前面，确保测试能直接导入 src 下模块。
    sys.path.insert(0, str(bootstrap_plugin_root))

load_dotenv(bootstrap_plugin_root / ".env", override=False)


def pytest_configure(config) -> None:
    """测试时开启 debug 日志输出。"""
    config.option.log_cli = True
    config.option.log_cli_level = "DEBUG"
