"""Shared GPT backbone with dense / MoE / MLA / DSA / recurrent flags."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmarcheval.config import ModelConfig
from llmarcheval.models.attention import Attention
from llmarcheval.models.moe import MixtureOfExperts, build_mlp


@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    stats: dict = field(default_factory=dict)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = build_mlp(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        attn_out, attn_stats = self.attn(self.ln_1(x))
        x = x + attn_out
        mlp_in = self.ln_2(x)
        aux = x.new_zeros(())
        stats = dict(attn_stats)
        if isinstance(self.mlp, MixtureOfExperts):
            mlp_out, aux, moe_stats = self.mlp(mlp_in)
            stats.update(moe_stats)
        else:
            mlp_out = self.mlp(mlp_in)
        x = x + mlp_out
        return x, aux, stats


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        if config.use_recurrent:
            self.prelude = nn.ModuleList(Block(config) for _ in range(config.n_prelude))
            self.recurrent = nn.ModuleList(Block(config) for _ in range(config.n_recurrent))
            self.coda = nn.ModuleList(Block(config) for _ in range(config.n_coda))
            self.h = None
        else:
            self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
            self.prelude = None
            self.recurrent = None
            self.coda = None
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_blocks(
        self, blocks: nn.ModuleList, x: torch.Tensor, aux_acc: list[torch.Tensor], stats_acc: list[dict]
    ) -> torch.Tensor:
        for block in blocks:
            x, aux, stats = block(x)
            aux_acc.append(aux)
            stats_acc.append(stats)
        return x

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loops: int | None = None,
        effort: str | None = None,
    ) -> GPTOutput:
        if idx.size(1) > self.config.block_size:
            raise ValueError(
                f"Sequence length {idx.size(1)} exceeds block_size {self.config.block_size}"
            )
        loop_count = self.config.loops_for_effort(
            effort, loops, training=self.training
        )

        x = self.drop(self.wte(idx))
        aux_acc: list[torch.Tensor] = []
        stats_acc: list[dict] = []

        if self.config.use_recurrent:
            assert self.prelude is not None and self.recurrent is not None and self.coda is not None
            x = self._run_blocks(self.prelude, x, aux_acc, stats_acc)
            for _ in range(loop_count):
                x = self._run_blocks(self.recurrent, x, aux_acc, stats_acc)
            x = self._run_blocks(self.coda, x, aux_acc, stats_acc)
        else:
            assert self.h is not None
            x = self._run_blocks(self.h, x, aux_acc, stats_acc)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        aux_loss = torch.stack(aux_acc).mean() if aux_acc else logits.new_zeros(())
        merged = _merge_stats(stats_acc)
        merged["loops"] = loop_count
        return GPTOutput(logits=logits, loss=loss, aux_loss=aux_loss, stats=merged)

    def blocks(self) -> list[Block]:
        if self.config.use_recurrent:
            assert self.prelude is not None and self.recurrent is not None and self.coda is not None
            return list(self.prelude) + list(self.recurrent) + list(self.coda)
        assert self.h is not None
        return list(self.h)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        loops: int | None = None,
        effort: str | None = None,
        max_think_tokens: int = 0,
    ) -> torch.Tensor:
        """Greedy-multinomial decode. `effort` maps to recurrent loops (Fable-style dial)."""
        loop_count = self.config.loops_for_effort(effort, loops, training=False)
        extra = max(max_think_tokens, 0)
        for _ in range(max_new_tokens + extra):
            idx_cond = idx[:, -self.config.block_size :]
            output = self(idx_cond, loops=loop_count)
            logits = output.logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        if extra:
            idx = idx[:, :-extra]
        return idx


def _merge_stats(stats_acc: list[dict]) -> dict:
    if not stats_acc:
        return {}
    merged: dict = {"n_blocks_executed": len(stats_acc)}
    entropies = [item["router_entropy"] for item in stats_acc if "router_entropy" in item]
    if entropies:
        merged["router_entropy"] = torch.stack(entropies).mean()
    loads = [item["expert_tokens"] for item in stats_acc if "expert_tokens" in item]
    if loads:
        merged["expert_tokens"] = torch.stack(loads).sum(dim=0)
    selected = [item["dsa_selected_mean"] for item in stats_acc if "dsa_selected_mean" in item]
    if selected:
        merged["dsa_selected_mean"] = torch.stack(selected).mean()
        merged["dsa_selected_frac"] = torch.stack(
            [item["dsa_selected_frac"] for item in stats_acc if "dsa_selected_frac" in item]
        ).mean()
    # Keep the last block's keep-mask for needle probes (most useful at the top).
    for item in reversed(stats_acc):
        if "dsa_keep" in item and "dsa_keep" not in merged:
            merged["dsa_keep"] = item["dsa_keep"]
        if "router_indices" in item and "router_indices" not in merged:
            merged["router_indices"] = item["router_indices"]
        if "kv_latent_rank" in item and "kv_latent_rank" not in merged:
            merged["kv_latent_rank"] = item["kv_latent_rank"]
    return merged


def build_model(config: ModelConfig) -> GPT:
    return GPT(config)
