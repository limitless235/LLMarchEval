"""Throughput and memory measurement on synthetic batches."""

from __future__ import annotations

import time

import torch

from llmarcheval.config import ModelConfig
from llmarcheval.models.transformer import GPT


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


@torch.no_grad()
def measure_forward(
    model: GPT,
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 1,
    steps: int = 5,
    warmup: int = 2,
    loops: int | None = None,
) -> dict:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
    for _ in range(warmup):
        model(x, y, loops=loops)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        model(x, y, loops=loops)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens = batch_size * config.block_size * steps
    return {
        "steps": steps,
        "batch_size": batch_size,
        "block_size": config.block_size,
        "elapsed_s": elapsed,
        "tok_s": tokens / max(elapsed, 1e-6),
        "peak_memory_bytes": peak_memory_bytes(device),
        "device": str(device),
    }
