"""GPT-2 tokenizer wrapper. We do not train a tokenizer."""

from __future__ import annotations

from functools import lru_cache

import tiktoken
import torch

GPT2_VOCAB_SIZE = 50257


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return get_encoding().encode_ordinary(text)


def decode(ids: list[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return get_encoding().decode(ids)


def encode_batch(texts: list[str]) -> list[list[int]]:
    enc = get_encoding()
    return [enc.encode_ordinary(text) for text in texts]
