"""MoE routing probe across controlled text regimes.

Records per-expert token counts, utilization, routing entropy, and top-k
assignments. Results are architecture diagnostics — not security findings.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from llmarcheval.experiments.common import (
    base_record,
    build_model,
    load_probe_experiment,
    parameter_block,
    resolve_device,
    resolve_dtype,
    set_seed,
    write_result,
)
from llmarcheval.tokenizer import encode

TEXT_REGIMES: dict[str, str] = {
    "normal": (
        "The baker set pies on a shelf. Apple, berry, and peach cooled by the window. "
        "Children chose carefully while the street musicians played a slow waltz."
    ),
    "repetitive": ("apple pie " * 80).strip(),
    "structured": (
        "name: alice\nage: 30\ncity: paris\n"
        "name: bob\nage: 25\ncity: berlin\n"
        "name: cara\nage: 41\ncity: tokyo\n"
        "name: diego\nage: 19\ncity: lima\n"
        "name: elena\nage: 36\ncity: rome\n"
    ),
    "reversed": (
        "The baker set pies on a shelf. Apple, berry, and peach cooled by the window. "
        "Children chose carefully while the street musicians played a slow waltz."
    )[::-1],
}


def _ids_for(text: str, vocab_size: int, block_size: int, device: torch.device) -> torch.Tensor:
    ids = encode(text) or [1, 2, 3, 4]
    ids = [i % vocab_size for i in ids]
    if len(ids) > block_size:
        ids = ids[:block_size]
    if len(ids) < 4:
        ids = ids + [0] * (4 - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def _routing_metrics(stats: dict, n_experts: int) -> dict[str, Any]:
    expert_tokens = stats.get("expert_tokens")
    counts = [int(x) for x in expert_tokens.detach().cpu().tolist()] if expert_tokens is not None else [0] * n_experts
    total = max(sum(counts), 1)
    utilization_pct = [100.0 * c / total for c in counts]
    entropy = float(stats["router_entropy"].detach().cpu()) if "router_entropy" in stats else None
    topk_sample = None
    topk_hist = None
    if "router_indices" in stats:
        idx = stats["router_indices"].detach().cpu()
        flat = idx.reshape(-1, idx.size(-1))
        topk_sample = flat[: min(32, flat.size(0))].tolist()
        topk_hist = [int(x) for x in torch.bincount(idx.reshape(-1), minlength=n_experts).tolist()]
    return {
        "per_expert_token_counts": counts,
        "utilization_pct": utilization_pct,
        "routing_entropy": entropy,
        "topk_assignments_sample": topk_sample,
        "topk_assignment_counts": topk_hist,
        "n_tokens_routed": int(total),
    }


def _load_checkpoint(model: torch.nn.Module, ckpt: str | Path, device: torch.device) -> dict[str, Any]:
    """Load weights using the established trainer/eval checkpoint schema."""
    ckpt_path = Path(ckpt)
    blob = torch.load(ckpt_path, map_location=device)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"Checkpoint {ckpt_path} missing required 'model' state dict")
    model.load_state_dict(blob["model"])
    step = blob.get("step")
    return {
        "path": str(ckpt_path),
        "name": ckpt_path.name,
        "step": int(step) if step is not None else None,
    }


@torch.no_grad()
def run_moe_routing_experiment(
    *,
    variant: str = "v1_moe",
    scale: str = "smoke",
    seed: int = 1337,
    device_name: str = "cpu",
    train_config: str = "configs/smoke.yaml",
    out_dir: str | Path = "results/experiments",
    write: bool = True,
    ckpt: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    exp = load_probe_experiment(variant, train_config=train_config, scale=scale)
    if not exp.model.use_moe:
        raise ValueError(f"{variant} is not an MoE model")
    device = resolve_device(device_name)
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    checkpoint_meta: dict[str, Any] | None = None
    if ckpt is not None:
        checkpoint_meta = _load_checkpoint(model, ckpt, device)
    model.eval()
    params = parameter_block(model, exp.model)

    regime_metrics: dict[str, Any] = {}
    losses: dict[str, float] = {}
    tokens_processed = 0
    t0 = time.perf_counter()
    for name, text in TEXT_REGIMES.items():
        x = _ids_for(text, exp.model.vocab_size, exp.model.block_size, device)
        out = model(x, x)
        assert out.loss is not None
        losses[name] = float(out.loss.detach().cpu())
        regime_metrics[name] = _routing_metrics(out.stats, exp.model.n_experts)
        tokens_processed += int(x.numel())
    wall = time.perf_counter() - t0

    notes = [
        "Offline deterministic corpora; no network required.",
        "Expert counts accumulate across MoE layers in the forward merge.",
        "Do not interpret routing skew as a security finding.",
    ]
    if checkpoint_meta is None:
        notes.append("Fresh model initialization (no checkpoint).")
    else:
        notes.append(f"Weights loaded from checkpoint {checkpoint_meta['name']}.")

    record = base_record(
        experiment="moe_routing",
        variant=variant,
        seed=seed,
        exp=exp,
        device=device,
        dtype=dtype,
        params=params,
        context_length=exp.model.block_size,
        wall_clock_s=wall,
        tokens_processed=tokens_processed,
        training_steps=checkpoint_meta.get("step") if checkpoint_meta else None,
        loss_metrics={"per_regime_nll": losses},
        metrics={
            "regimes": regime_metrics,
            "checkpoint": checkpoint_meta,
            "interpretation": "Architecture routing diagnostic only; not a security evaluation.",
        },
        notes=notes,
    )
    if write:
        stem = f"moe_routing_{variant}_{seed}"
        if checkpoint_meta is not None:
            stem = f"{stem}_{Path(checkpoint_meta['name']).stem}"
        path = write_result(record, out_dir, stem=stem)
        record["output_path"] = str(path)
    return record
