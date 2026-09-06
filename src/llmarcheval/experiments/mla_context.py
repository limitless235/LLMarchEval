"""MLA vs dense-attention context scaling probe.

KV byte figures are theoretical layouts. This lab does not implement a production
decode KV cache; MLA training still materializes dense K/V.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import torch

from llmarcheval.experiments.common import (
    base_record,
    build_model,
    estimate_from_config,
    estimate_training_memory,
    load_probe_experiment,
    load_probe_model,
    parameter_block,
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    resolve_dtype,
    set_seed,
    synchronize,
    write_result,
)
from llmarcheval.train.checkpoint import load_weights_into_model, model_config_from_checkpoint

DEFAULT_CONTEXTS = (256, 512, 1024, 2048)


@torch.no_grad()
def _run_one(
    *,
    variant: str,
    scale: str,
    train_config: str,
    context_length: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int = 1,
    steps: int = 2,
    ckpt: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_meta = None
    if ckpt is not None:
        # Rebuild from checkpoint-authoritative architecture via shared helpers.
        # Only extend block_size for the context sweep (does not change weight shapes).
        model, exp, checkpoint_meta = load_probe_model(
            variant=variant,
            device=device,
            dtype=dtype,
            train_config=train_config,
            scale=scale,
            ckpt=ckpt,
        )
        cfg = deepcopy(exp.model)
        needed = int(context_length)
        if int(cfg.block_size) < needed:
            cfg.block_size = needed
            model = build_model(cfg, device, dtype)
            # Reload weights into the block_size-extended shell (same shapes).
            load_weights_into_model(model, ckpt, device)
    else:
        exp = load_probe_experiment(variant, train_config=train_config, scale=scale)
        cfg = deepcopy(exp.model)
        cfg.block_size = int(context_length)
        model = build_model(cfg, device, dtype)
    model.eval()
    params = parameter_block(model, cfg)
    est = estimate_from_config(cfg)
    mem_est = estimate_training_memory(cfg, batch_size=1)

    x = torch.randint(0, cfg.vocab_size, (1, context_length), device=device)
    y = x.clone()
    for _ in range(warmup):
        model(x, y)
    synchronize(device)
    reset_peak_memory(device)

    t0 = time.perf_counter()
    last_loss = None
    for _ in range(steps):
        out = model(x, y)
        assert out.loss is not None
        last_loss = float(out.loss.detach().cpu())
    synchronize(device)
    wall = time.perf_counter() - t0
    tokens = context_length * steps
    peak = peak_memory_bytes(device)
    row: dict[str, Any] = {
        "variant": variant,
        "use_mla": bool(cfg.use_mla),
        "context_length": int(context_length),
        "forward_time_s": wall,
        "tokens_per_sec": tokens / max(wall, 1e-9),
        "steps": steps,
        "loss": last_loss,
        "loss_finite": bool(last_loss is not None and math.isfinite(last_loss)),
        "peak_memory_bytes_measured": peak,
        "peak_memory_is_measured": peak is not None,
        "analytical_kv_bytes_fp32": int(est["kv_bytes_per_token_fp32"]) * int(context_length),
        "analytical_kv_bytes_are_theoretical": True,
        "analytical_attention_score_bytes": int(mem_est["estimated_attention_score_bytes"]),
        "parameter_counts": {
            "total": params["total_params"],
            "active": params["active_params"],
        },
        "model_config": {
            "n_embd": int(cfg.n_embd),
            "n_layer": int(cfg.n_layer),
            "n_head": int(cfg.n_head),
            "block_size": int(cfg.block_size),
            "use_moe": bool(cfg.use_moe),
            "use_mla": bool(cfg.use_mla),
        },
        "checkpoint": checkpoint_meta,
    }
    return row


def run_mla_context_experiment(
    *,
    variants: Sequence[str] = ("v1_moe", "v2_moe_mla"),
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    scale: str = "smoke",
    seed: int = 1337,
    device_name: str = "cpu",
    train_config: str = "configs/smoke.yaml",
    out_dir: str | Path = "results/experiments",
    write: bool = True,
    steps: int = 2,
    ckpts: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_name)
    dtype = resolve_dtype("float32", device)
    ckpt_map = dict(ckpts or {})

    rows: list[dict[str, Any]] = []
    tokens_processed = 0
    t0 = time.perf_counter()
    for variant in variants:
        for ctx in contexts:
            row = _run_one(
                variant=variant,
                scale=scale,
                train_config=train_config,
                context_length=int(ctx),
                device=device,
                dtype=dtype,
                steps=steps,
                ckpt=ckpt_map.get(variant),
            )
            rows.append(row)
            tokens_processed += int(ctx) * steps
    wall = time.perf_counter() - t0

    # Top-level architecture/provenance:
    # - single variant with ckpt → checkpoint-authoritative (matches other probes)
    # - otherwise keep YAML defaults for the lead variant; per-row records carry
    #   their own model_config/checkpoint so multi-variant results stay unambiguous
    single_variant = len(variants) == 1
    single_ckpt = ckpt_map.get(variants[0]) if single_variant else None
    top_checkpoint = None
    training_steps = None
    if single_ckpt is not None:
        _model, exp0, top_checkpoint = load_probe_model(
            variant=variants[0],
            device=device,
            dtype=dtype,
            train_config=train_config,
            scale=scale,
            ckpt=single_ckpt,
        )
        del _model
        params0 = parameter_block(build_model(exp0.model, device, dtype), exp0.model)
        training_steps = top_checkpoint.get("step") if top_checkpoint else None
        # Prefer the first row's checkpoint meta (same file) if present.
        if rows and rows[0].get("checkpoint") is not None:
            top_checkpoint = rows[0]["checkpoint"]
    else:
        exp0 = load_probe_experiment(variants[0], train_config=train_config, scale=scale)
        params0 = parameter_block(build_model(exp0.model, device, dtype), exp0.model)

    notes = [
        "Optional per-variant checkpoints may be supplied via ckpts mapping.",
        "Compares V1 (MHA+MoE) vs V2 (MLA+MoE).",
        "analytical_kv_bytes_fp32 is a theoretical decode-cache layout estimate.",
        "Current MLA training forward still materializes dense K/V; no production decode cache.",
        "peak_memory_bytes_measured is only populated on CUDA in this harness.",
    ]
    if top_checkpoint is not None:
        notes.insert(
            0,
            f"Weights loaded from checkpoint {top_checkpoint['name']}.",
        )
    elif any(r.get("checkpoint") is not None for r in rows):
        notes.insert(
            0,
            "Per-row checkpoints loaded; top-level config is the lead variant YAML "
            "(not a single checkpoint-authoritative aggregate).",
        )
    else:
        notes.insert(0, "Fresh model initialization (no checkpoint).")

    metrics: dict[str, Any] = {
        "contexts": list(contexts),
        "variants": list(variants),
        "rows": rows,
        "checkpoint": top_checkpoint,
    }

    record = base_record(
        experiment="mla_context",
        variant=",".join(variants),
        seed=seed,
        exp=exp0,
        device=device,
        dtype=dtype,
        params=params0,
        context_length=max(int(c) for c in contexts),
        wall_clock_s=wall,
        tokens_processed=tokens_processed,
        training_steps=training_steps,
        loss_metrics={
            "per_setting_nll": {
                f"{r['variant']}@{r['context_length']}": r["loss"] for r in rows
            }
        },
        metrics=metrics,
        notes=notes,
    )
    if write:
        path = write_result(record, out_dir, stem=f"mla_context_{seed}")
        record["output_path"] = str(path)
    return record
