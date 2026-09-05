"""Architecture-surface probes: routing entropy and loop-count consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmarcheval.models.transformer import GPT
from llmarcheval.tokenizer import encode

CLEAN = "The baker set pies on a shelf. Apple, berry, peach. Children chose carefully."
REPETITIVE = "aaaaaa " * 40
SHIFTED = "The baker set pies on a shelf. Apple, berry, peach. Children chose carefully."[::-1]


def _ids_for(model: GPT, text: str, device: torch.device) -> torch.Tensor:
    ids = torch.tensor([encode(text)], dtype=torch.long, device=device)
    ids = ids % model.config.vocab_size
    if ids.size(1) > model.config.block_size:
        ids = ids[:, : model.config.block_size]
    if ids.size(1) < 4:
        pad = torch.zeros(1, 8, dtype=torch.long, device=device)
        ids = torch.cat([ids, pad], dim=1)[:, :8]
        ids = ids % model.config.vocab_size
    return ids


@torch.no_grad()
def routing_probe(model: GPT, device: torch.device) -> dict:
    model.eval()
    if not model.config.use_moe:
        return {"skipped": True, "reason": "model is not MoE"}
    rows = {}
    for name, text in {"clean": CLEAN, "repetitive": REPETITIVE, "reversed": SHIFTED}.items():
        out = model(_ids_for(model, text, device), loops=None)
        rows[name] = {
            "router_entropy": float(out.stats["router_entropy"].item()) if "router_entropy" in out.stats else None,
            "expert_tokens": out.stats["expert_tokens"].tolist() if "expert_tokens" in out.stats else None,
        }
    return rows


@torch.no_grad()
def loop_consistency_probe(
    model: GPT, device: torch.device, loop_counts: tuple[int, ...] = (1, 2, 4, 8)
) -> dict:
    model.eval()
    if not model.config.use_recurrent:
        return {"skipped": True, "reason": "model is not recurrent"}
    prompt = _ids_for(model, "Once upon a time", device)
    logits = {}
    gens = {}
    for loops in loop_counts:
        out = model(prompt, loops=loops)
        logits[loops] = out.logits[0, -1]
        gens[loops] = int(out.logits[0, -1].argmax().item())
    pairs = []
    keys = list(loop_counts)
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        cos = F.cosine_similarity(logits[a].float(), logits[b].float(), dim=0).item()
        pairs.append({"loops_a": a, "loops_b": b, "cosine": cos, "same_argmax": gens[a] == gens[b]})
    return {"argmax": gens, "pairwise": pairs}


@torch.no_grad()
def effort_dial_probe(model: GPT, device: torch.device) -> dict:
    """Same weights, different test-time budget (Fable-style effort dial)."""
    model.eval()
    prompt = _ids_for(model, "Two plus two is", device)
    rows = {}
    for effort in ("low", "medium", "high"):
        loops = model.config.loops_for_effort(effort)
        out = model(prompt, loops=loops)
        rows[effort] = {
            "loops": loops,
            "argmax": int(out.logits[0, -1].argmax().item()),
        }
    return rows
