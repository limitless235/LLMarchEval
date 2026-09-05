"""Experiment and model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_DIR = CONFIGS_DIR / "models"

EFFORT_TO_LOOPS = {
    "low": 1,
    "medium": 2,
    "high": 4,
    "xhigh": 8,
    "max": 8,
}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _apply_fields(cls: type, data: dict[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class ModelConfig:
    variant: str = "v0_dense"
    vocab_size: int = 50257
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    n_inner: int | None = None

    use_moe: bool = False
    n_experts: int = 8
    n_active: int = 2
    n_shared_experts: int = 1
    moe_intermediate_size: int | None = None
    router_aux_loss_coef: float = 0.01

    use_mla: bool = False
    kv_lora_rank: int = 32
    qk_rope_head_dim: int = 16
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None

    use_dsa: bool = False
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 32
    dsa_local_window: int = 16

    use_recurrent: bool = False
    n_prelude: int = 1
    n_recurrent: int = 2
    n_coda: int = 1
    train_loops: int = 2
    eval_loops: int = 2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        head_dim = self.head_dim
        if self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = max(head_dim - self.qk_rope_head_dim, 8)
        if self.v_head_dim is None:
            self.v_head_dim = head_dim
        if self.use_moe and self.moe_intermediate_size is None:
            # Match dense SwiGLU active FLOPs: shared + top-k routed ≈ one dense FFN.
            active_ffns = self.n_shared_experts + self.n_active
            self.moe_intermediate_size = max(self.n_inner // max(active_ffns, 1), 32)
        if self.use_recurrent:
            unique = self.n_prelude + self.n_recurrent + self.n_coda
            if unique <= 0:
                raise ValueError("recurrent model needs at least one unique layer")
            self.n_layer = unique

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def unique_layers(self) -> int:
        if self.use_recurrent:
            return self.n_prelude + self.n_recurrent + self.n_coda
        return self.n_layer

    def loops_for_effort(
        self,
        effort: str | None = None,
        loops: int | None = None,
        *,
        training: bool = False,
    ) -> int:
        """Resolve recurrent loop count.

        Priority: explicit ``loops`` > ``effort`` dial > train/eval default.
        Non-recurrent models always return 1.
        """
        if loops is not None:
            return max(int(loops), 1)
        if not self.use_recurrent:
            return 1
        if effort is not None:
            key = effort.lower()
            if key not in EFFORT_TO_LOOPS:
                raise ValueError(f"Unknown effort {effort!r}; expected one of {sorted(EFFORT_TO_LOOPS)}")
            return EFFORT_TO_LOOPS[key]
        return self.train_loops if training else self.eval_loops


@dataclass
class TrainConfig:
    scale: str = "smoke"
    dataset: str = "tinystories"
    max_docs: int | None = 2000
    max_iters: int | None = 200
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3.0e-4
    warmup_iters: int = 20
    min_lr: float = 3.0e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 20
    eval_interval: int = 50
    eval_iters: int = 10
    ckpt_interval: int = 100
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1337
    out_dir: str = "results/smoke"
    tokens_budget: int | None = None
    num_workers: int = 0
    compile: bool = False
    # If False (default), TinyStories/FineWeb load failures raise — no silent smoke corpus.
    # Smoke configs may set True for offline development only.
    allow_dataset_fallback: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_path: str = ""
    train_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "train": asdict(self.train),
            "model_path": self.model_path,
            "train_path": self.train_path,
        }


def load_model_config(path: str | Path, scale: str = "smoke") -> ModelConfig:
    raw = _load_yaml(path)
    merged: dict[str, Any] = dict(raw)
    scales = merged.pop("scales", {}) or {}
    if scale not in scales:
        raise KeyError(f"Scale {scale!r} not in {path}; have {sorted(scales)}")
    merged.update(scales[scale])
    merged.pop("notes", None)
    return _apply_fields(ModelConfig, merged)


def load_train_config(path: str | Path) -> TrainConfig:
    raw = _load_yaml(path)
    return _apply_fields(TrainConfig, raw)


def load_experiment(
    model_path: str | Path,
    train_path: str | Path,
    scale: str | None = None,
) -> ExperimentConfig:
    train = load_train_config(train_path)
    chosen_scale = scale or train.scale
    model = load_model_config(model_path, scale=chosen_scale)
    return ExperimentConfig(
        model=model,
        train=train,
        model_path=str(model_path),
        train_path=str(train_path),
    )


def variant_config_path(variant: str) -> Path:
    path = MODELS_DIR / f"{variant}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config for variant {variant!r} at {path}")
    return path


def list_variants() -> list[str]:
    return sorted(path.stem for path in MODELS_DIR.glob("*.yaml"))
