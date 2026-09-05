"""Perplexity and generation helpers."""

from __future__ import annotations

import math

import torch

from llmarcheval.models.transformer import GPT
from llmarcheval.tokenizer import decode, encode


@torch.no_grad()
def sequence_nll(model: GPT, token_ids: torch.Tensor, loops: int | None = None) -> float:
    model.eval()
    idx = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    out = model(idx, targets, loops=loops)
    assert out.loss is not None
    return float(out.loss.item())


def perplexity(nll: float) -> float:
    return math.exp(min(nll, 20.0))


SIMPLE_COMPLETIONS = [
    "The cat sat on the",
    "Two plus two is",
    "Once upon a time",
    "The capital of France is",
    "1 2 3 4",
]


@torch.no_grad()
def completion_probe(
    model: GPT,
    device: torch.device,
    max_new_tokens: int = 8,
    loops: int | None = None,
    effort: str | None = None,
) -> list[dict]:
    model.eval()
    rows = []
    for prompt in SIMPLE_COMPLETIONS:
        ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        ids = ids % model.config.vocab_size
        if ids.size(1) >= model.config.block_size:
            ids = ids[:, -model.config.block_size + 1 :]
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0, loops=loops, effort=effort)
        rows.append({"prompt": prompt, "completion": decode(out[0])})
    return rows
