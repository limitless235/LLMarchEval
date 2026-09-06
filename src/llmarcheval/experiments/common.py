"""Shared experiment metadata, seeding, and JSON/JSONL writers."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llmarcheval.config import CONFIGS_DIR, ExperimentConfig, ModelConfig, TrainConfig, load_experiment, variant_config_path
from llmarcheval.models.accounting import estimate_from_config, estimate_training_memory, summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import auto_device, auto_dtype

SCHEMA_VERSION = "1.0"

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "experiment",
    "git_commit",
    "timestamp_utc",
    "seed",
    "variant",
    "config",
    "parameter_counts",
    "active_parameter_counts",
    "context_length",
    "batch_size",
    "grad_accumulation",
    "dtype",
    "device",
    "training_steps",
    "training_tokens",
    "wall_clock_s",
    "tokens_per_sec",
    "loss_metrics",
    "metrics",
    "accounting_proxy",
    "notes",
)

_ALLOW_FIELD = next(f.name for f in fields(TrainConfig) if f.name.startswith("allow_"))


def assert_result_schema(record: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in record]
    if missing:
        raise AssertionError(f"Missing schema keys: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise AssertionError(f"Unexpected schema_version {record['schema_version']!r}")
    if not isinstance(record["config"], dict) or "model" not in record["config"] or "train" not in record["config"]:
        raise AssertionError("config must include model and train")
    if not isinstance(record["parameter_counts"], dict):
        raise AssertionError("parameter_counts must be a dict")
    for key in ("total", "active"):
        if key not in record["parameter_counts"]:
            raise AssertionError(f"parameter_counts missing {key}")
    if not isinstance(record["loss_metrics"], dict):
        raise AssertionError("loss_metrics must be a dict")
    if not isinstance(record["metrics"], dict):
        raise AssertionError("metrics must be a dict")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str = "cpu") -> torch.device:
    return auto_device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    return auto_dtype(name, device)


def load_probe_experiment(
    variant: str,
    *,
    train_config: str | Path = "configs/smoke.yaml",
    scale: str | None = "smoke",
) -> ExperimentConfig:
    path = Path(train_config)
    if not path.exists():
        path = CONFIGS_DIR / Path(train_config).name
    return load_experiment(variant_config_path(variant), path, scale=scale)


def build_model(model_cfg: ModelConfig, device: torch.device, dtype: torch.dtype) -> GPT:
    model = GPT(deepcopy(model_cfg)).to(device)
    if dtype != torch.float32 and device.type == "cuda":
        model = model.to(dtype=dtype)
    return model


def load_probe_model(
    *,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
    train_config: str | Path = "configs/smoke.yaml",
    scale: str | None = "smoke",
    ckpt: str | Path | None = None,
) -> tuple[GPT, ExperimentConfig, dict[str, Any] | None]:
    """Build a probe model from YAML defaults, or from a trained checkpoint.

    When ``ckpt`` is set, architecture comes from the checkpoint's stored
    ``config.model`` (not smoke/default YAML). This is the generic path used by
    MoE/MLA/DSA/recurrent probes for trained-checkpoint loading.
    """
    if ckpt is None:
        exp = load_probe_experiment(variant, train_config=train_config, scale=scale)
        return build_model(exp.model, device, dtype), exp, None
    from llmarcheval.train.checkpoint import load_model_from_checkpoint

    model, exp, meta = load_model_from_checkpoint(
        ckpt,
        device=device,
        dtype=dtype,
        variant=variant,
    )
    return model, exp, meta


def parameter_block(model: GPT, config: ModelConfig) -> dict[str, Any]:
    accounting = summarize_model(model, config)
    est = estimate_from_config(config)
    return {
        "total_params": int(accounting.get("measured_total_params", est["total_params"])),
        "active_params": int(est["active_params"]),
        "components": {k: int(v) for k, v in est.get("components", {}).items()},
        "flops_per_token_proxy": float(est["flops_per_token"]),
        "kv_dims_per_token": int(est["kv_dims_per_token"]),
        "kv_bytes_per_token_fp32": int(est["kv_bytes_per_token_fp32"]),
    }


def base_record(
    *,
    experiment: str,
    variant: str,
    seed: int,
    exp: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    params: dict[str, Any],
    context_length: int,
    wall_clock_s: float | None = None,
    tokens_processed: int | None = None,
    training_steps: int | None = None,
    training_tokens: int | None = None,
    loss_metrics: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    tokens_per_sec = None
    if wall_clock_s is not None and tokens_processed is not None and wall_clock_s > 0:
        tokens_per_sec = tokens_processed / wall_clock_s
    train_cfg = {
        "scale": exp.train.scale,
        "dataset": exp.train.dataset,
        "batch_size": exp.train.batch_size,
        "grad_accum": exp.train.grad_accum,
        "dtype": exp.train.dtype,
        "device": exp.train.device,
        "max_iters": exp.train.max_iters,
        "seed": exp.train.seed,
        _ALLOW_FIELD: getattr(exp.train, _ALLOW_FIELD),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "git_commit": git_commit(),
        "timestamp_utc": utc_now(),
        "seed": int(seed),
        "variant": variant,
        "config": {
            "model": asdict(exp.model),
            "train": train_cfg,
            "model_path": exp.model_path,
            "train_path": exp.train_path,
        },
        "parameter_counts": {
            "total": params["total_params"],
            "active": params["active_params"],
            "components": params["components"],
        },
        "active_parameter_counts": params["active_params"],
        "context_length": int(context_length),
        "batch_size": int(exp.train.batch_size),
        "grad_accumulation": int(exp.train.grad_accum),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "training_steps": training_steps,
        "training_tokens": training_tokens,
        "wall_clock_s": wall_clock_s,
        "tokens_per_sec": tokens_per_sec,
        "tokens_processed": tokens_processed,
        "loss_metrics": loss_metrics or {},
        "metrics": metrics or {},
        "accounting_proxy": {
            "flops_per_token_proxy": params["flops_per_token_proxy"],
            "flops_per_token_is_proxy": True,
            "kv_dims_per_token": params["kv_dims_per_token"],
            "kv_bytes_per_token_fp32": params["kv_bytes_per_token_fp32"],
            "kv_bytes_are_theoretical": True,
        },
        "notes": notes or [],
    }


def write_result(record: dict[str, Any], out_dir: str | Path, stem: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = stem or f"{record['experiment']}_{record['variant'].replace(',', '_')}_{record['seed']}"
    json_path = out / f"{name}.json"
    jsonl_path = out / "results.jsonl"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + chr(10))
    return json_path


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_TOP_LEVEL_KEYS",
    "assert_result_schema",
    "git_commit",
    "utc_now",
    "set_seed",
    "resolve_device",
    "resolve_dtype",
    "load_probe_experiment",
    "build_model",
    "load_probe_model",
    "parameter_block",
    "base_record",
    "write_result",
    "peak_memory_bytes",
    "reset_peak_memory",
    "synchronize",
    "estimate_from_config",
    "estimate_training_memory",
]
