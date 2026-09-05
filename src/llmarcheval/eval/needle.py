"""Needle-in-a-haystack retrieval and DSA indexer drop-rate."""

from __future__ import annotations

import torch

from llmarcheval.models.transformer import GPT
from llmarcheval.tokenizer import decode, encode


NEEDLE = "The passcode is 93214."
QUERY = " What is the passcode?"


def _pad_or_trim(ids: list[int], length: int, pad_id: int = 220) -> list[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


@torch.no_grad()
def needle_eval(
    model: GPT,
    device: torch.device,
    haystack_len: int,
    loops: int | None = None,
) -> dict:
    """Place a needle near the middle of a haystack and ask for the passcode.

    Also reports whether the last-layer DSA indexer kept the needle tokens
    for the final query position (drop-rate probe).
    """
    model.eval()
    filler = encode("The river is quiet. " * 400)
    needle_ids = encode(NEEDLE)
    query_ids = encode(QUERY)
    prefix_len = max(haystack_len // 2, 1)
    prefix = filler[:prefix_len]
    rest_budget = haystack_len - len(prefix) - len(needle_ids) - len(query_ids)
    rest = filler[: max(rest_budget, 0)]
    sequence = prefix + needle_ids + rest + query_ids
    if len(sequence) > model.config.block_size:
        sequence = sequence[-model.config.block_size :]
    idx = torch.tensor([sequence], dtype=torch.long, device=device)
    out = model(idx, loops=loops)
    pred = out.logits[0, -1].argmax().item()
    generated = decode([pred])

    needle_start = len(prefix)
    needle_end = needle_start + len(needle_ids)
    # If we trimmed from the left, remap.
    trim = max(len(prefix) + len(needle_ids) + len(rest) + len(query_ids) - model.config.block_size, 0)
    needle_start = max(needle_start - trim, 0)
    needle_end = max(needle_end - trim, 0)

    drop_frac = None
    kept = None
    if "dsa_keep" in out.stats:
        keep = out.stats["dsa_keep"][0]  # [T, T]
        query_pos = keep.size(0) - 1
        if needle_end > needle_start:
            needle_slice = keep[query_pos, needle_start:needle_end]
            kept = bool(needle_slice.any().item())
            drop_frac = 1.0 - float(needle_slice.float().mean().item())
    return {
        "haystack_len": haystack_len,
        "seq_len": idx.size(1),
        "next_token": generated,
        "contains_digit": any(ch.isdigit() for ch in generated),
        "dsa_needle_kept": kept,
        "dsa_needle_drop_frac": drop_frac,
        "needle_span": [needle_start, needle_end],
    }
