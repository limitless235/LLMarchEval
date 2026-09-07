"""Tests for robust checkpoint save/load/resume and generic V0–V4 loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llmarcheval.config import ExperimentConfig, ModelConfig, TrainConfig, list_variants
from llmarcheval.eval.protocol import evaluate_checkpoint
from llmarcheval.experiments.common import build_model, load_probe_experiment, resolve_device, resolve_dtype, set_seed
from llmarcheval.train.checkpoint import (
    build_checkpoint,
    load_checkpoint_blob,
    load_model_from_checkpoint,
    load_weights_into_model,
    resume_start_step,
    save_checkpoint,
)
from llmarcheval.train.trainer import train


def _tiny_exp(tmp_path: Path, *, max_iters: int = 4, seed: int = 0) -> ExperimentConfig:
    model = ModelConfig(
        variant="v0_dense",
        vocab_size=50257,
        n_layer=1,
        n_embd=32,
        n_head=4,
        block_size=16,
    )
    return ExperimentConfig(
        model=model,
        train=TrainConfig(
            scale="smoke",
            dataset="corpus",
            max_iters=max_iters,
            batch_size=2,
            warmup_iters=0,
            log_interval=100,
            eval_interval=0,
            ckpt_interval=2,
            out_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            seed=seed,
            allow_dataset_fallback=True,
        ),
    )


def test_checkpoint_roundtrip_restores_model_and_optimizer(tmp_path: Path):
    exp = _tiny_exp(tmp_path, max_iters=3, seed=1)
    train(exp)
    ckpt_path = Path(exp.train.out_dir) / exp.model.variant / "ckpt_final.pt"
    assert ckpt_path.exists()
    blob = load_checkpoint_blob(ckpt_path)
    assert "model" in blob
    assert "optimizer" in blob
    assert "rng" in blob
    assert blob["step"] == exp.train.max_iters
    assert blob["steps_completed"] == exp.train.max_iters
    assert blob.get("tokens_seen", 0) > 0

    device = resolve_device("cpu")
    model = build_model(exp.model, device, resolve_dtype("float32", device))
    meta = load_weights_into_model(model, ckpt_path, device)
    assert meta["step"] == exp.train.max_iters
    assert meta["has_optimizer"] is True

    for key, tensor in blob["model"].items():
        assert torch.allclose(model.state_dict()[key].cpu(), tensor.cpu())

    opt = torch.optim.AdamW(model.parameters(), lr=exp.train.lr)
    opt.load_state_dict(blob["optimizer"])
    assert opt.state_dict()["state"]


def test_resume_preserves_step(tmp_path: Path):
    exp = _tiny_exp(tmp_path / "a", max_iters=4, seed=2)
    train(exp)
    mid = Path(exp.train.out_dir) / exp.model.variant / "ckpt_2.pt"
    assert mid.exists()
    blob = load_checkpoint_blob(mid)
    assert resume_start_step(blob, max_iters=4) == 3

    exp2 = _tiny_exp(tmp_path / "b", max_iters=4, seed=2)
    summary = train(exp2, resume_from=mid)
    assert summary["start_step"] == 3
    assert summary["resumed_from"] == str(mid)
    assert all(row["step"] >= 3 for row in summary["history"])


def test_legacy_checkpoint_still_loads(tmp_path: Path):
    set_seed(0)
    exp = load_probe_experiment("v0_dense", train_config="configs/smoke.yaml", scale="smoke")
    device = resolve_device("cpu")
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "config": exp.to_dict(), "step": 50}, path)
    loaded, loaded_exp, meta = load_model_from_checkpoint(path, device=device, variant="v0_dense")
    assert meta["step"] == 50
    assert meta["has_optimizer"] is False
    assert resume_start_step(load_checkpoint_blob(path), max_iters=200) == 51


@pytest.mark.parametrize("variant", list_variants())
def test_all_variants_construct_and_generic_ckpt_load(variant: str, tmp_path: Path):
    set_seed(0)
    exp = load_probe_experiment(variant, train_config="configs/smoke.yaml", scale="smoke")
    device = resolve_device("cpu")
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    path = tmp_path / f"{variant}.pt"
    blob = build_checkpoint(model=model, config=exp, step=10, steps_completed=10, tokens_seen=100)
    save_checkpoint(path, blob)
    loaded, _, meta = load_model_from_checkpoint(
        path, device=device, model_cfg=exp.model, variant=variant
    )
    assert loaded is not None
    assert meta["step"] == 10
    again = build_model(exp.model, device, dtype)
    meta2 = load_weights_into_model(again, path, device)
    assert meta2["step"] == 10


def test_eval_protocol_smoke(tmp_path: Path):
    exp = _tiny_exp(tmp_path, max_iters=2, seed=3)
    train(exp)
    ckpt = Path(exp.train.out_dir) / exp.model.variant / "ckpt_final.pt"
    record = evaluate_checkpoint(
        ckpt=ckpt,
        variant=exp.model.variant,
        train_config="configs/smoke.yaml",
        device_name="cpu",
        eval_iters=2,
        seed=123,
        out_dir=tmp_path / "eval",
        write=True,
    )
    assert record["protocol"] == "fixed_val_nll_v1"
    assert "nll" in record
    assert "perplexity" in record
    assert record["parameter_counts"]["total"] > 0
    assert record["parameter_counts"]["active"] > 0
    assert record["training_step"] == exp.train.max_iters
    assert Path(record["output_path"]).exists()
