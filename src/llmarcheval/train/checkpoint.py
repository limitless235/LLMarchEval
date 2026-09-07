"""Checkpoint save/load and resume helpers.

Backward-compatible with existing trainer checkpoints that only store
``{"model", "config", "step"}``. New checkpoints add optimizer/RNG/metadata
fields without renaming the original keys.

Trained-checkpoint loading reconstructs architecture from ``config.model``
(authoritative). Callers must not inject unrelated default/smoke dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from llmarcheval.config import ExperimentConfig, ModelConfig, TrainConfig, _apply_fields
from llmarcheval.models.transformer import GPT

# Bump only when the on-disk schema becomes incompatible with prior loaders.
CHECKPOINT_VERSION = 2

REQUIRED_KEYS = ("model",)

# Core fields required to rebuild the exact trained architecture.
_CORE_ARCH_KEYS = (
    "variant",
    "vocab_size",
    "n_embd",
    "n_head",
    "block_size",
    "use_moe",
    "use_mla",
    "use_dsa",
    "use_recurrent",
)

_MOE_ARCH_KEYS = ("n_experts", "n_active", "n_shared_experts")
_MLA_ARCH_KEYS = ("kv_lora_rank", "qk_rope_head_dim")
_DSA_ARCH_KEYS = ("index_n_heads", "index_head_dim", "index_topk", "dsa_local_window")
_RECURRENT_ARCH_KEYS = ("n_prelude", "n_recurrent", "n_coda")


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
    """Atomically write a checkpoint (temp file + os.replace).

    Avoids leaving a truncated ``.pt`` at the destination if serialization fails
    mid-write (previously observed as zip central-directory corruption).
    """
    import os
    import tempfile

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        torch.save(blob, tmp_path)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, out)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return out


def prune_mid_checkpoints(out_dir: str | Path, keep_last: int) -> list[Path]:
    """Delete older mid-run ``ckpt_{step}.pt`` files; never touches ``ckpt_final.pt``.

    Returns paths that were removed. ``keep_last <= 0`` deletes all mid checkpoints.
    """
    directory = Path(out_dir)
    if not directory.is_dir():
        return []
    mids: list[tuple[int, Path]] = []
    for path in directory.glob("ckpt_*.pt"):
        if path.name == "ckpt_final.pt":
            continue
        stem = path.stem  # ckpt_1000
        if not stem.startswith("ckpt_"):
            continue
        suffix = stem[len("ckpt_") :]
        if not suffix.isdigit():
            continue
        mids.append((int(suffix), path))
    mids.sort(key=lambda item: item[0])
    keep = max(int(keep_last), 0)
    to_remove = mids if keep == 0 else mids[:-keep] if len(mids) > keep else []
    removed: list[Path] = []
    for _step, path in to_remove:
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


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


def _missing_architecture_keys(model_raw: dict[str, Any]) -> list[str]:
    """Return architecture keys absent from a checkpoint ``config.model`` dict."""
    missing = [key for key in _CORE_ARCH_KEYS if key not in model_raw or model_raw[key] is None]
    if model_raw.get("use_recurrent"):
        missing.extend(key for key in _RECURRENT_ARCH_KEYS if key not in model_raw or model_raw[key] is None)
    elif "n_layer" not in model_raw or model_raw["n_layer"] is None:
        missing.append("n_layer")
    if model_raw.get("use_moe"):
        missing.extend(key for key in _MOE_ARCH_KEYS if key not in model_raw or model_raw[key] is None)
    if model_raw.get("use_mla"):
        missing.extend(key for key in _MLA_ARCH_KEYS if key not in model_raw or model_raw[key] is None)
    if model_raw.get("use_dsa"):
        missing.extend(key for key in _DSA_ARCH_KEYS if key not in model_raw or model_raw[key] is None)
    # Preserve order while de-duplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for key in missing:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def model_config_from_checkpoint(blob: dict[str, Any]) -> ModelConfig:
    """Reconstruct ``ModelConfig`` from checkpoint-stored architecture metadata.

    The checkpoint's ``config.model`` is the source of truth. Missing required
    fields raise rather than falling back to smoke/default dimensions.
    """
    cfg = blob.get("config")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict):
        raise ValueError(
            "Checkpoint lacks authoritative architecture metadata under config.model; "
            "cannot reconstruct the trained model. Legacy checkpoints must include "
            "{model, config, step} with a complete config.model matching the trained "
            "architecture (hidden size, layers, heads, MoE/MLA/DSA/recurrent fields, "
            "vocab_size, block_size)."
        )
    model_raw = cfg["model"]
    missing = _missing_architecture_keys(model_raw)
    if missing:
        raise ValueError(
            "Checkpoint config.model is insufficient to reconstruct the exact "
            f"architecture; missing required fields: {missing}. Refusing to build a "
            "different (e.g. smoke/default) model."
        )
    return _apply_fields(ModelConfig, model_raw)


def experiment_from_checkpoint(
    blob: dict[str, Any],
    *,
    train_config: str | Path | None = None,
    scale: str | None = None,
    variant: str | None = None,
) -> ExperimentConfig:
    """Rebuild ExperimentConfig using checkpoint architecture as source of truth.

    Model dimensions always come from ``config.model``. Train settings come from
    the checkpoint when present, otherwise from an optional ``train_config`` YAML
    (dataset/device overrides only — never used to replace model architecture).
    """
    model = model_config_from_checkpoint(blob)
    if variant is not None:
        model.variant = variant

    cfg = blob.get("config") if isinstance(blob.get("config"), dict) else {}
    train: TrainConfig
    if train_config is not None:
        from llmarcheval.config import load_train_config

        train = load_train_config(train_config)
    elif isinstance(cfg.get("train"), dict):
        train = _apply_fields(TrainConfig, cfg["train"])
    else:
        raise ValueError(
            "Checkpoint lacks config.train and no train_config= was provided; "
            "cannot rebuild ExperimentConfig. Architecture was found in config.model "
            "but training hyperparameters are missing."
        )
    if scale is not None:
        train.scale = scale
    return ExperimentConfig(
        model=model,
        train=train,
        model_path=str(cfg.get("model_path", "")),
        train_path=str(cfg.get("train_path", "")),
    )


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

    Architecture is always reconstructed from the checkpoint's stored
    ``config.model``. The optional ``model_cfg`` argument is accepted for API
    compatibility but is ignored when checkpoint metadata is present — callers
    must not override trained dimensions with smoke/default YAML.
    """
    del model_cfg  # Checkpoint architecture is authoritative; do not use caller dims.
    blob = load_checkpoint_blob(path, map_location=device)
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
    """Load only weights into an already-built model (experiment harness path).

    Prefer ``load_model_from_checkpoint`` when the model was not already built
    from the checkpoint's stored architecture — loading into a differently sized
    model will fail under ``strict=True``.
    """
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
    "model_config_from_checkpoint",
    "prune_mid_checkpoints",
    "restore_rng_state",
    "resume_start_step",
    "save_checkpoint",
]
