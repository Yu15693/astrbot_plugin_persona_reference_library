from ..utils.url import is_data_url, is_http_url
from .schema import AdapterImage


def append_adapter_image(output: list[AdapterImage], raw_url: str) -> None:
    if is_data_url(raw_url):
        output.append(AdapterImage(kind="data_url", value=raw_url))
        return
    if is_http_url(raw_url):
        output.append(AdapterImage(kind="http_url", value=raw_url))
