"""Lightweight dynamic sparse attention indexer (GLM-inspired, not the production kernel)."""

from __future__ import annotations

import torch
import torch.nn as nn

from llmarcheval.config import ModelConfig


class DSAIndexer(nn.Module):
    """Score previous tokens and keep top-k plus a local causal window."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.local_window = config.dsa_local_window
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(config.n_embd, inner, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, inner, bias=config.bias)

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: [B, T, C]
            causal: [1, 1, T, T] boolean (True = allowed)
        Returns:
            keep: [B, T, T] boolean
            stats: indexer diagnostics including keep mask for probes
        """
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.mean(dim=1)  # [B, T, T]
        causal_bt = causal.squeeze(1).expand(batch, -1, -1)
        scores = scores.masked_fill(~causal_bt, torch.finfo(scores.dtype).min)

        k_keep = min(self.topk, seq_len)
        topk_idx = scores.topk(k_keep, dim=-1).indices
        keep = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=x.device)
        keep.scatter_(-1, topk_idx, True)

        query = torch.arange(seq_len, device=x.device).view(1, seq_len, 1)
        key = torch.arange(seq_len, device=x.device).view(1, 1, seq_len)
        local = (key <= query) & (key > query - self.local_window)
        keep = keep | local.expand(batch, -1, -1)
        keep = keep & causal_bt

        selected = keep.float().sum(dim=-1)
        stats = {
            "dsa_keep": keep.detach(),
            "dsa_selected_mean": selected.mean(),
            "dsa_selected_frac": selected.mean() / seq_len,
        }
        return keep, stats
