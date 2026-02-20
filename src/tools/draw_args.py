from __future__ import annotations


def parse_draw_args(args_text: str) -> tuple[str, str, str]:
    """解析 draw 参数中的 `ratio=` 与 `size=` 键值对。

    当前仅识别两个键：
    - `ratio=<值>`
    - `size=<值>`

    其余 token（包括未知键值对）会保留到 prompt，
    以保证后续接入更多供应商时仍具备输入灵活性。
    """
    ratio = ""
    size = ""
    prompt_tokens: list[str] = []

    for token in args_text.split():
        if "=" not in token:
            prompt_tokens.append(token)
            continue
        key, value = token.split("=", 1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key == "ratio":
            ratio = normalized_value
            continue
        if normalized_key == "size":
            size = normalized_value
            continue
        prompt_tokens.append(token)

    return ratio, size, " ".join(prompt_tokens).strip()
