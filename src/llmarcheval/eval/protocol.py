"""Fixed evaluation protocol for trained V0–V4 checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from llmarcheval.config import (
    CONFIGS_DIR,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    _apply_fields,
    load_experiment,
    variant_config_path,
)
from llmarcheval.eval.perplexity import perplexity
from llmarcheval.models.accounting import summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.checkpoint import load_checkpoint_blob, load_weights_into_model
from llmarcheval.train.data import TokenBatcher, load_tokens
from llmarcheval.train.trainer import auto_device, auto_dtype, estimate_loss

DEFAULT_EVAL_ITERS = 50
DEFAULT_EVAL_SEED = 1337


def evaluate_checkpoint(
    *,
    ckpt: str | Path,
    variant: str | None = None,
    train_config: str | Path | None = None,
    scale: str | None = None,
    device_name: str | None = None,
    eval_iters: int = DEFAULT_EVAL_ITERS,
    seed: int = DEFAULT_EVAL_SEED,
    out_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained checkpoint with a fixed validation protocol.

    Uses the same validation split construction as training (90/10 on the
    configured dataset), a fixed seed for the val batcher, and a fixed number
    of evaluation iterations. Reports NLL, perplexity (when finite), parameter
    counts, and checkpoint/training metadata.
    """
    ckpt_path = Path(ckpt)
    blob = load_checkpoint_blob(ckpt_path, map_location="cpu")
    cfg_blob = blob.get("config") if isinstance(blob.get("config"), dict) else {}
    chosen_variant = (
        variant
        or blob.get("variant")
        or (cfg_blob.get("model") or {}).get("variant")
    )
    if not chosen_variant:
        raise ValueError("Could not resolve variant; pass variant= explicitly")

    if train_config is not None:
        experiment = load_experiment(
            variant_config_path(chosen_variant), train_config, scale=scale
        )
        if isinstance(cfg_blob.get("model"), dict):
            experiment.model = _apply_fields(ModelConfig, cfg_blob["model"])
    elif isinstance(cfg_blob.get("model"), dict) and isinstance(cfg_blob.get("train"), dict):
        experiment = ExperimentConfig(
            model=_apply_fields(ModelConfig, cfg_blob["model"]),
            train=_apply_fields(TrainConfig, cfg_blob["train"]),
            model_path=str(cfg_blob.get("model_path", "")),
            train_path=str(cfg_blob.get("train_path", "")),
        )
        if scale is not None:
            experiment.train.scale = scale
    else:
        experiment = load_experiment(
            variant_config_path(chosen_variant),
            CONFIGS_DIR / "smoke.yaml",
            scale=scale,
        )

    device = auto_device(device_name or experiment.train.device)
    dtype = auto_dtype(experiment.train.dtype, device)
    torch.manual_seed(seed)

    model = GPT(experiment.model).to(device)
    meta = load_weights_into_model(model, ckpt_path, device)
    model.eval()

    tokens, source = load_tokens(experiment.train)
    n = tokens.size
    split = max(int(n * 0.9), experiment.model.block_size + 2)
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    if val_tokens.size <= experiment.model.block_size + 1:
        val_tokens = train_tokens
    val_batcher = TokenBatcher(
        val_tokens, experiment.model.block_size, experiment.train.batch_size, seed + 1
    )

    nll = estimate_loss(model, val_batcher, device, int(eval_iters), dtype)
    ppl = perplexity(nll) if math.isfinite(nll) else None
    accounting = summarize_model(
        model, experiment.model, batch_size=experiment.train.batch_size
    )

    step = meta.get("step")
    tokens_seen = meta.get("tokens_seen")
    if tokens_seen is None and step is not None:
        tokens_per_step = (
            experiment.train.batch_size
            * experiment.model.block_size
            * experiment.train.grad_accum
        )
        tokens_seen = int(step) * tokens_per_step

    record: dict[str, Any] = {
        "protocol": "fixed_val_nll_v1",
        "checkpoint": meta,
        "variant": experiment.model.variant,
        "dataset_source": source,
        "val_split": "tail_10pct_or_train_fallback",
        "eval_iters": int(eval_iters),
        "seed": int(seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "nll": float(nll),
        "perplexity": float(ppl) if ppl is not None else None,
        "perplexity_valid": ppl is not None and math.isfinite(ppl),
        "parameter_counts": {
            "total": int(
                accounting.get("measured_total_params", accounting.get("total_params", 0))
            ),
            "active": int(accounting["active_params"]),
        },
        "training_step": step,
        "training_tokens": tokens_seen,
        "steps_completed": meta.get("steps_completed"),
        "notes": [
            "Fixed validation protocol: same split rule as training, fixed seed, fixed eval_iters.",
            "Perplexity is exp(NLL) when NLL is finite.",
            "Does not claim frontier-model reproduction.",
        ],
    }

    if write:
        target = (
            Path(out_dir)
            if out_dir
            else Path(experiment.train.out_dir) / experiment.model.variant
        )
        target.mkdir(parents=True, exist_ok=True)
        out_path = target / f"eval_protocol_{ckpt_path.stem}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["output_path"] = str(out_path)
    return record


__all__ = ["DEFAULT_EVAL_ITERS", "DEFAULT_EVAL_SEED", "evaluate_checkpoint"]
