from __future__ import annotations

from collections.abc import Sequence


def extract_command_args(message_str: str, command_tokens: Sequence[str]) -> str:
    """提取命令后的原始参数文本。"""
    tokens = message_str.strip().split()
    normalized_message = " ".join(tokens)

    normalized_command = [
        token.strip().lower() for token in command_tokens if token.strip()
    ]
    if not normalized_command:
        return normalized_message

    if len(tokens) < len(normalized_command):
        return normalized_message

    matched = all(
        tokens[index].lower() == normalized_command[index]
        for index in range(len(normalized_command))
    )
    if not matched:
        return normalized_message

    return " ".join(tokens[len(normalized_command) :])
