"""Token counting helpers using a single shared encoding (cl100k_base)."""

from functools import lru_cache

import tiktoken

_TOKEN_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    """Return the shared tokenizer encoding."""
    return tiktoken.get_encoding(_TOKEN_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Count tokens with the shared encoding."""
    if not text:
        return 0
    return len(get_encoding().encode(text))


def encode_text(text: str) -> list[int]:
    return get_encoding().encode(text)


def decode_tokens(tokens: list[int]) -> str:
    return get_encoding().decode(tokens)
