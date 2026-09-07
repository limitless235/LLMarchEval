"""Fixed evaluation protocol for trained V0–V4 checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from llmarcheval.config import ExperimentConfig, TrainConfig, _apply_fields, load_train_config
from llmarcheval.eval.perplexity import perplexity
from llmarcheval.models.accounting import summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.checkpoint import (
    load_checkpoint_blob,
    load_weights_into_model,
    model_config_from_checkpoint,
)
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

    Model architecture is always reconstructed from the checkpoint's stored
    ``config.model``. Train/dataset settings come from the checkpoint when
    present, or from an optional ``train_config`` YAML (never used to replace
    model dimensions with smoke/default architecture).
    """
    ckpt_path = Path(ckpt)
    blob = load_checkpoint_blob(ckpt_path, map_location="cpu")
    cfg_blob = blob.get("config") if isinstance(blob.get("config"), dict) else {}

    model_cfg = model_config_from_checkpoint(blob)
    if variant is not None:
        model_cfg.variant = variant
    chosen_variant = model_cfg.variant

    if train_config is not None:
        train_cfg = load_train_config(train_config)
    elif isinstance(cfg_blob.get("train"), dict):
        train_cfg = _apply_fields(TrainConfig, cfg_blob["train"])
    else:
        raise ValueError(
            "Checkpoint lacks config.train and no --train-config was provided; "
            "cannot run the fixed eval protocol without dataset/train settings. "
            "Model architecture was loaded from config.model."
        )
    if scale is not None:
        train_cfg.scale = scale

    experiment = ExperimentConfig(
        model=model_cfg,
        train=train_cfg,
        model_path=str(cfg_blob.get("model_path", "")),
        train_path=str(cfg_blob.get("train_path", "")),
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
        "variant": chosen_variant,
        "dataset_source": source,
        "val_split": "tail_10pct_or_train_fallback",
        "eval_iters": int(eval_iters),
        "seed": int(seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model_config": {
            "n_embd": experiment.model.n_embd,
            "n_layer": experiment.model.n_layer,
            "n_head": experiment.model.n_head,
            "block_size": experiment.model.block_size,
            "vocab_size": experiment.model.vocab_size,
            "scale_hint": experiment.train.scale,
        },
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
            "Model architecture reconstructed from checkpoint config.model (not smoke defaults).",
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
