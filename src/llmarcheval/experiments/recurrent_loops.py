"""Recurrent-depth loop sweep for V4.

Treats loop count as an experimental compute dial / hypothesis about
recurrent depth — not as evidence of proprietary Astra internals.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from llmarcheval.experiments.common import (
    base_record,
    build_model,
    load_probe_experiment,
    parameter_block,
    resolve_device,
    resolve_dtype,
    set_seed,
    synchronize,
    write_result,
)
from llmarcheval.models.accounting import estimate_from_config
from llmarcheval.tokenizer import encode

PROMPT = "Once upon a time in a quiet village"


def _prompt_ids(vocab_size: int, block_size: int, device: torch.device) -> torch.Tensor:
    ids = [t % vocab_size for t in encode(PROMPT)] or list(range(8))
    if len(ids) < 8:
        ids = ids + list(range(8 - len(ids)))
    if len(ids) > block_size:
        ids = ids[:block_size]
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def run_recurrent_loops_experiment(
    *,
    variant: str = "v4_recurrent",
    scale: str = "smoke",
    seed: int = 1337,
    device_name: str = "cpu",
    train_config: str = "configs/smoke.yaml",
    loops: Sequence[int] = (1, 2, 3, 4),
    out_dir: str | Path = "results/experiments",
    write: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    exp = load_probe_experiment(variant, train_config=train_config, scale=scale)
    if not exp.model.use_recurrent:
        raise ValueError(f"{variant} is not recurrent")
    device = resolve_device(device_name)
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    model.eval()
    params = parameter_block(model, exp.model)

    x = _prompt_ids(exp.model.vocab_size, exp.model.block_size, device)
    y = x.clone()

    rows: list[dict[str, Any]] = []
    prev_logits = None
    prev_loss = None
    prev_loops = None
    tokens_processed = 0
    t_all = time.perf_counter()
    for loop_count in loops:
        est = estimate_from_config(exp.model, loops=int(loop_count))
        synchronize(device)
        t0 = time.perf_counter()
        out = model(x, y, loops=int(loop_count))
        synchronize(device)
        wall = time.perf_counter() - t0
        assert out.loss is not None
        loss = float(out.loss.detach().cpu())
        logits = out.logits[0, -1].detach().float()
        agreement = None
        marginal_loss = None
        marginal_flops = None
        if prev_logits is not None and prev_loops is not None:
            agreement = {
                "cosine_with_prev": float(F.cosine_similarity(prev_logits, logits, dim=0).item()),
                "same_argmax_as_prev": bool(prev_logits.argmax().item() == logits.argmax().item()),
                "prev_loops": prev_loops,
            }
            marginal_loss = loss - float(prev_loss)
            prev_est = estimate_from_config(exp.model, loops=int(prev_loops))
            marginal_flops = float(est["flops_per_token"]) - float(prev_est["flops_per_token"])
        rows.append(
            {
                "loops": int(loop_count),
                "loss": loss,
                "loss_finite": math.isfinite(loss),
                "wall_clock_s": wall,
                "tokens_per_sec": x.numel() / max(wall, 1e-9),
                "flops_per_token_proxy": float(est["flops_per_token"]),
                "flops_per_token_is_proxy": True,
                "executed_layers": int(est["executed_layers"]),
                "n_blocks_executed": int(out.stats.get("n_blocks_executed", -1)),
                "logit_agreement_vs_prev": agreement,
                "marginal_loss_vs_prev": marginal_loss,
                "marginal_flops_proxy_vs_prev": marginal_flops,
            }
        )
        prev_logits = logits
        prev_loss = loss
        prev_loops = int(loop_count)
        tokens_processed += int(x.numel())
    wall_all = time.perf_counter() - t_all

    record = base_record(
        experiment="recurrent_loops",
        variant=variant,
        seed=seed,
        exp=exp,
        device=device,
        dtype=dtype,
        params=params,
        context_length=exp.model.block_size,
        wall_clock_s=wall_all,
        tokens_processed=tokens_processed,
        loss_metrics={"per_loop_nll": {str(r["loops"]): r["loss"] for r in rows}},
        metrics={
            "loops": list(loops),
            "rows": rows,
            "hypothesis": (
                "Recurrent depth is treated as an experimental compute dial, "
                "not as evidence of proprietary Astra internals."
            ),
        },
        notes=[
            "Unique parameters do not grow with loops; compute does.",
            "flops_per_token is an analytical proxy (2 * active_params * depth_scale).",
            "Offline prompt; no network required.",
        ],
    )
    if write:
        path = write_result(record, out_dir)
        record["output_path"] = str(path)
    return record
