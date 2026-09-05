"""Estimate single-GPU pretrain cost from a short synthetic pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llmarcheval.config import (
    CONFIGS_DIR,
    list_variants,
    load_experiment,
    variant_config_path,
)
from llmarcheval.eval.throughput import measure_forward
from llmarcheval.models.accounting import summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import auto_device, auto_dtype


TOKEN_BUDGETS = (2_000_000_000, 3_000_000_000, 5_000_000_000)


def _hours_for(tokens: int, tok_s: float) -> float:
    return tokens / max(tok_s, 1e-6) / 3600.0


def estimate_variant(variant: str, train_config: Path, steps: int, scale: str | None) -> dict:
    experiment = load_experiment(variant_config_path(variant), train_config, scale=scale)
    device = auto_device(experiment.train.device)
    dtype = auto_dtype(experiment.train.dtype, device)
    model = GPT(experiment.model).to(device)
    if dtype != torch.float32 and device.type == "cuda":
        model = model.to(dtype=dtype)
    accounting = summarize_model(model, experiment.model)
    try:
        throughput = measure_forward(
            model,
            experiment.model,
            device,
            batch_size=max(experiment.train.batch_size, 1),
            steps=steps,
        )
        error = None
    except torch.cuda.OutOfMemoryError:
        throughput = {"tok_s": None, "peak_memory_bytes": None, "error": "oom"}
        error = "oom"
    tok_s = throughput.get("tok_s") or 0.0
    hours = {
        f"{int(tokens / 1e9)}B": {
            "tokens": tokens,
            "device_hours": _hours_for(tokens, tok_s) if tok_s else None,
        }
        for tokens in TOKEN_BUDGETS
    }
    return {
        "variant": variant,
        "scale": experiment.train.scale if scale is None else scale,
        "device": str(device),
        "dtype": str(dtype),
        "accounting": accounting,
        "throughput": throughput,
        "projected": hours,
        "error": error,
        "epistemic": (
            "recurrent_depth_inferred_not_confirmed"
            if experiment.model.use_recurrent
            else None
        ),
        "caveat": "device_hours use measured tok/s on this machine. GPU will be much faster than CPU.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pilot throughput and cost estimate")
    parser.add_argument("--train-config", type=Path, default=CONFIGS_DIR / "pilot.yaml")
    parser.add_argument("--scale", type=str, default=None, help="Override scale (smoke|full)")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_cost.json"))
    args = parser.parse_args(argv)

    rows = [estimate_variant(variant, args.train_config, args.steps, args.scale) for variant in list_variants()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "These are short synthetic pilots of the configured shapes, not a "
            "multi-billion-token pretrain. device_hours use tok/s measured on "
            "this machine (CPU here unless CUDA is available)."
        ),
        "variants": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
