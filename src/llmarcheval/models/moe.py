"""SwiGLU mixture-of-experts with a shared expert and switch-style aux loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmarcheval.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, intermediate: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate, bias=bias)
        self.w3 = nn.Linear(n_embd, intermediate, bias=bias)
        self.w2 = nn.Linear(intermediate, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MixtureOfExperts(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.moe_intermediate_size is None:
            raise ValueError("moe_intermediate_size must be set")
        self.n_experts = config.n_experts
        self.n_active = config.n_active
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.n_embd, config.moe_intermediate_size, config.bias, config.dropout)
                for _ in range(config.n_experts)
            ]
        )
        shared = []
        for _ in range(config.n_shared_experts):
            shared.append(
                SwiGLU(config.n_embd, config.moe_intermediate_size, config.bias, config.dropout)
            )
        self.shared = nn.ModuleList(shared)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        n_tokens = flat.size(0)

        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.n_active, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        me = probs.mean(dim=0)
        one_hot = F.one_hot(indices, num_classes=self.n_experts).to(dtype=probs.dtype)
        load = one_hot.sum(dim=1).sum(dim=0) / n_tokens
        aux = self.n_experts * torch.sum(me * load)

        routed = torch.zeros_like(flat)
        expert_tokens = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            assignment = indices == expert_id
            if not assignment.any():
                continue
            token_idx = assignment.any(dim=-1).nonzero(as_tuple=True)[0]
            expert_tokens[expert_id] = token_idx.numel()
            expert_weight = (assignment.to(weights.dtype) * weights).sum(dim=-1)[token_idx]
            routed = routed.index_add(
                0, token_idx, expert(flat[token_idx]) * expert_weight.unsqueeze(-1)
            )

        shared_out = torch.zeros_like(flat)
        for shared in self.shared:
            shared_out = shared_out + shared(flat)
        out = (shared_out + routed).view(batch, seq_len, hidden)

        entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1).mean()
        stats = {
            "expert_tokens": expert_tokens.detach(),
            "router_entropy": entropy.detach(),
            "router_load": load.detach(),
            "router_indices": indices.detach().view(batch, seq_len, self.n_active),
        }
        return out, aux, stats


def build_mlp(config: ModelConfig) -> nn.Module:
    if config.use_moe:
        return MixtureOfExperts(config)
    assert config.n_inner is not None
    return SwiGLU(config.n_embd, config.n_inner, config.bias, config.dropout)
