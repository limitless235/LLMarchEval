"""Multi-head attention and simplified multi-head latent attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmarcheval.config import ModelConfig
from llmarcheval.models.dsa import DSAIndexer
from llmarcheval.models.rope import RotaryEmbedding, apply_rotary


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).view(
        1, 1, seq_len, seq_len
    )


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.use_mla = config.use_mla
        self.use_dsa = config.use_dsa
        self.dropout_p = config.dropout

        if self.use_mla:
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.v_head_dim = config.v_head_dim
            q_dim = self.n_head * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            self.q_proj = nn.Linear(config.n_embd, q_dim, bias=config.bias)
            self.kv_down = nn.Linear(config.n_embd, config.kv_lora_rank, bias=config.bias)
            self.k_rope_proj = nn.Linear(
                config.n_embd, self.n_head * self.qk_rope_head_dim, bias=config.bias
            )
            self.kv_up_k = nn.Linear(
                config.kv_lora_rank, self.n_head * self.qk_nope_head_dim, bias=config.bias
            )
            self.kv_up_v = nn.Linear(
                config.kv_lora_rank, self.n_head * self.v_head_dim, bias=config.bias
            )
            self.c_proj = nn.Linear(self.n_head * self.v_head_dim, config.n_embd, bias=config.bias)
            self.rotary = RotaryEmbedding(self.qk_rope_head_dim, max_seq_len=config.block_size * 2)
            attn_scale_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size * 2)
            attn_scale_dim = self.head_dim

        self.attn_scale = attn_scale_dim**-0.5
        self.resid_dropout = nn.Dropout(config.dropout)
        self.indexer = DSAIndexer(config) if self.use_dsa else None

    def _mha_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary.get_cos_sin(seq_len, x.device, x.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        return q, k, v

    def _mla_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch, seq_len, self.n_head, self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)

        kv_latent = self.kv_down(x)
        k_nope = self.kv_up_k(kv_latent).view(batch, seq_len, self.n_head, self.qk_nope_head_dim)
        k_nope = k_nope.transpose(1, 2)
        v = self.kv_up_v(kv_latent).view(batch, seq_len, self.n_head, self.v_head_dim).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(batch, seq_len, self.n_head, self.qk_rope_head_dim)
        k_rope = k_rope.transpose(1, 2)

        cos, sin = self.rotary.get_cos_sin(seq_len, x.device, x.dtype)
        q_rope = apply_rotary(q_rope, cos, sin)
        k_rope = apply_rotary(k_rope, cos, sin)
        q = torch.cat((q_nope, q_rope), dim=-1)
        k = torch.cat((k_nope, k_rope), dim=-1)
        return q, k, v, kv_latent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, _ = x.shape
        stats: dict = {}
        kv_latent = None
        if self.use_mla:
            q, k, v, kv_latent = self._mla_qkv(x)
            stats["kv_latent"] = kv_latent.detach()
            stats["kv_latent_rank"] = kv_latent.size(-1)
        else:
            q, k, v = self._mha_qkv(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        causal = causal_mask(seq_len, x.device)
        if self.indexer is not None:
            keep, idx_stats = self.indexer(x, causal)
            stats.update(idx_stats)
            attn_scores = attn_scores.masked_fill(~keep.unsqueeze(1), torch.finfo(attn_scores.dtype).min)
        else:
            attn_scores = attn_scores.masked_fill(~causal, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y, stats
