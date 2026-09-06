"""Checkpoint save/load and resume helpers.

Backward-compatible with existing trainer checkpoints that only store
``{"model", "config", "step"}``. New checkpoints add optimizer/RNG/metadata
fields without renaming the original keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from llmarcheval.config import ExperimentConfig, ModelConfig, load_experiment, variant_config_path
from llmarcheval.models.transformer import GPT

# Bump only when the on-disk schema becomes incompatible with prior loaders.
CHECKPOINT_VERSION = 2

REQUIRED_KEYS = ("model",)


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG states used by the training loop."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(
    *,
    model: nn.Module,
    config: ExperimentConfig | dict[str, Any],
    step: int,
    steps_completed: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    tokens_seen: int = 0,
    train_batcher_rng: Any | None = None,
    val_batcher_rng: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint dict.

    ``step`` keeps the legacy meaning used by existing Mac checkpoints:
    last finished loop index for mid-run saves, or ``max_iters`` for final.
    ``steps_completed`` is the unambiguous count of optimizer steps done
    (equal to the next ``for``-loop index to run).
    """
    cfg_payload = config.to_dict() if isinstance(config, ExperimentConfig) else dict(config)
    completed = steps_completed if steps_completed is not None else int(step)
    blob: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "config": cfg_payload,
        "step": int(step),
        "steps_completed": int(completed),
        "tokens_seen": int(tokens_seen),
        "rng": capture_rng_state(),
        "variant": None,
    }
    model_cfg = cfg_payload.get("model") if isinstance(cfg_payload, dict) else None
    if isinstance(model_cfg, dict):
        blob["variant"] = model_cfg.get("variant")
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        blob["scaler"] = scaler.state_dict()
    if train_batcher_rng is not None:
        blob["train_batcher_rng"] = train_batcher_rng
    if val_batcher_rng is not None:
        blob["val_batcher_rng"] = val_batcher_rng
    if extra:
        blob["extra"] = extra
    return blob


def save_checkpoint(path: str | Path, blob: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    return out


def load_checkpoint_blob(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(blob, dict):
        raise ValueError(f"Checkpoint {ckpt_path} is not a dict")
    for key in REQUIRED_KEYS:
        if key not in blob:
            raise ValueError(f"Checkpoint {ckpt_path} missing required key {key!r}")
    return blob


def resume_start_step(blob: dict[str, Any], max_iters: int) -> int:
    """Return the next optimizer step index to run."""
    if "steps_completed" in blob and blob["steps_completed"] is not None:
        return int(blob["steps_completed"])
    # Legacy: mid-run ``step`` is last finished index; final uses max_iters.
    step = int(blob.get("step", 0))
    if step >= max_iters:
        return max_iters
    return step + 1


def apply_model_state(model: nn.Module, blob: dict[str, Any], *, strict: bool = True) -> None:
    model.load_state_dict(blob["model"], strict=strict)


def apply_optimizer_state(optimizer: torch.optim.Optimizer, blob: dict[str, Any]) -> bool:
    """Restore optimizer if present. Returns True when restored."""
    if "optimizer" not in blob or blob["optimizer"] is None:
        return False
    optimizer.load_state_dict(blob["optimizer"])
    return True


def apply_scaler_state(scaler: Any | None, blob: dict[str, Any]) -> bool:
    if scaler is None or "scaler" not in blob or blob["scaler"] is None:
        return False
    scaler.load_state_dict(blob["scaler"])
    return True


def checkpoint_meta(blob: dict[str, Any], path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(path)
    step = blob.get("step")
    steps_completed = blob.get("steps_completed")
    return {
        "path": str(ckpt_path),
        "name": ckpt_path.name,
        "step": int(step) if step is not None else None,
        "steps_completed": int(steps_completed) if steps_completed is not None else None,
        "tokens_seen": int(blob["tokens_seen"]) if blob.get("tokens_seen") is not None else None,
        "variant": blob.get("variant"),
        "checkpoint_version": blob.get("checkpoint_version"),
        "has_optimizer": "optimizer" in blob and blob["optimizer"] is not None,
        "has_rng": "rng" in blob and blob["rng"] is not None,
    }


def experiment_from_checkpoint(
    blob: dict[str, Any],
    *,
    train_config: str | Path | None = None,
    scale: str | None = None,
    variant: str | None = None,
) -> ExperimentConfig:
    """Rebuild ExperimentConfig from checkpoint metadata, falling back to YAML."""
    cfg = blob.get("config")
    chosen_variant = variant or blob.get("variant")
    if isinstance(cfg, dict) and "model" in cfg and "train" in cfg:
        from llmarcheval.config import ModelConfig, TrainConfig, _apply_fields

        model = _apply_fields(ModelConfig, cfg["model"])
        train = _apply_fields(TrainConfig, cfg["train"])
        if train_config is not None:
            from llmarcheval.config import load_train_config

            train = load_train_config(train_config)
        if scale is not None:
            train.scale = scale
        if chosen_variant is not None:
            model.variant = chosen_variant
        return ExperimentConfig(
            model=model,
            train=train,
            model_path=str(cfg.get("model_path", "")),
            train_path=str(cfg.get("train_path", "")),
        )
    if chosen_variant is None:
        raise ValueError("Checkpoint lacks config/variant; pass variant= explicitly")
    from llmarcheval.config import CONFIGS_DIR, load_experiment

    train_path = Path(train_config) if train_config else CONFIGS_DIR / "smoke.yaml"
    return load_experiment(variant_config_path(chosen_variant), train_path, scale=scale)


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    variant: str | None = None,
    train_config: str | Path | None = None,
    scale: str | None = None,
    model_cfg: ModelConfig | None = None,
    strict: bool = True,
) -> tuple[GPT, ExperimentConfig, dict[str, Any]]:
    """Generic V0–V4 checkpoint load via the existing GPT builder path.

    Prefer an explicit ``model_cfg`` (research harness) so architecture YAML wins;
    otherwise reconstruct from checkpoint config.
    """
    blob = load_checkpoint_blob(path, map_location=device)
    if model_cfg is not None:
        exp = ExperimentConfig(model=model_cfg)
        if isinstance(blob.get("config"), dict) and "train" in blob["config"]:
            try:
                exp = experiment_from_checkpoint(
                    blob, train_config=train_config, scale=scale, variant=variant or model_cfg.variant
                )
                exp.model = model_cfg
            except Exception:
                exp = ExperimentConfig(model=model_cfg)
        else:
            exp = ExperimentConfig(model=model_cfg)
    else:
        exp = experiment_from_checkpoint(
            blob, train_config=train_config, scale=scale, variant=variant
        )
    model = GPT(exp.model).to(device)
    if dtype is not None and dtype != torch.float32 and device.type == "cuda":
        model = model.to(dtype=dtype)
    apply_model_state(model, blob, strict=strict)
    meta = checkpoint_meta(blob, path)
    return model, exp, meta


def load_weights_into_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load only weights into an already-built model (experiment harness path)."""
    blob = load_checkpoint_blob(path, map_location=device)
    apply_model_state(model, blob, strict=strict)
    return checkpoint_meta(blob, path)


__all__ = [
    "CHECKPOINT_VERSION",
    "apply_model_state",
    "apply_optimizer_state",
    "apply_scaler_state",
    "build_checkpoint",
    "capture_rng_state",
    "checkpoint_meta",
    "experiment_from_checkpoint",
    "load_checkpoint_blob",
    "load_model_from_checkpoint",
    "load_weights_into_model",
    "restore_rng_state",
    "resume_start_step",
    "save_checkpoint",
]
