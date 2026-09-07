"""Tests for packed FineWeb corpus loading and atomic checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from llmarcheval.config import TrainConfig, load_train_config
from llmarcheval.train.checkpoint import prune_mid_checkpoints, save_checkpoint
from llmarcheval.train.data import load_packed_tokens, load_tokens


def test_load_packed_tokens_roundtrip(tmp_path: Path):
    path = tmp_path / "tokens.uint16.npy"
    arr = np.arange(2000, dtype=np.uint16)
    np.save(path, arr)
    loaded = load_packed_tokens(path)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (2000,)
    assert int(loaded[10]) == 10


def test_load_tokens_packed_dataset_key(tmp_path: Path):
    path = tmp_path / "tokens.uint16.npy"
    np.save(path, np.arange(3000, dtype=np.uint16))
    cfg = TrainConfig(dataset="packed", packed_tokens_path=str(path), allow_dataset_fallback=False)
    tokens, source = load_tokens(cfg)
    assert source.startswith("packed:")
    assert tokens.size == 3000


def test_fineweb_configs_resolve_and_token_budgets():
    cfg1 = load_train_config("configs/fineweb_edu_1m.yaml")
    cfg20 = load_train_config("configs/fineweb_edu_20m.yaml")
    assert cfg1.scale == "local_16gb_w768"
    assert cfg20.scale == "local_16gb_w768"
    assert cfg1.dataset == "packed"
    assert cfg20.dataset == "packed"
    assert cfg1.packed_tokens_path.endswith("tokens.uint16.npy")
    assert cfg1.batch_size == 1 and cfg1.grad_accum == 8
    assert cfg1.dtype == "float32" and cfg1.device == "mps"
    assert cfg1.seed == 1337 and cfg20.seed == 1337
    assert cfg1.ckpt_interval == 1000 and cfg20.ckpt_interval == 1000
    assert cfg1.ckpt_keep_last == 2 and cfg20.ckpt_keep_last == 2
    assert cfg1.warmup_iters == 20 and cfg20.warmup_iters == 20
    assert cfg1.allow_dataset_fallback is False
    # Campaign accounting (also encoded as defaults in YAML).
    assert cfg1.max_iters == 245
    assert cfg20.max_iters == 4883
    assert cfg1.tokens_budget == 1_000_000
    assert cfg20.tokens_budget == 20_000_000


def test_campaign_dry_run_fineweb_step_counts():
    from llmarcheval.cli.campaign import _tokens_to_iters
    from llmarcheval.config import load_experiment, variant_config_path

    exp = load_experiment(variant_config_path("v0_dense"), "configs/fineweb_edu_20m.yaml")
    assert exp.model.n_embd == 768
    assert exp.model.block_size == 512
    steps = _tokens_to_iters(
        20_000_000, exp.train.batch_size, exp.model.block_size, exp.train.grad_accum
    )
    assert steps == 4883
    assert steps * exp.train.batch_size * exp.model.block_size * exp.train.grad_accum == 20_000_768
    steps1 = _tokens_to_iters(
        1_000_000, exp.train.batch_size, exp.model.block_size, exp.train.grad_accum
    )
    assert steps1 == 245
    assert steps1 * 4096 == 1_003_520


def test_atomic_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "ckpt_1000.pt"
    blob = {"model": {"w": torch.ones(3)}, "step": 1000, "checkpoint_version": 2}
    save_checkpoint(path, blob)
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["step"] == 1000
    assert torch.equal(loaded["model"]["w"], torch.ones(3))


def test_prune_mid_checkpoints_keeps_final_and_last_n(tmp_path: Path):
    for step in (1000, 2000, 3000, 4000):
        (tmp_path / f"ckpt_{step}.pt").write_bytes(b"x")
    (tmp_path / "ckpt_final.pt").write_bytes(b"final")
    removed = prune_mid_checkpoints(tmp_path, keep_last=2)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "ckpt_final.pt" in names
    assert "ckpt_3000.pt" in names
    assert "ckpt_4000.pt" in names
    assert "ckpt_1000.pt" not in names
    assert "ckpt_2000.pt" not in names
    assert len(removed) == 2


def test_pack_fineweb_dry_run_no_download():
    import importlib.util
    from pathlib import Path as P

    path = P("scripts/pack_fineweb_edu.py")
    spec = importlib.util.spec_from_file_location("pack_fineweb_edu", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.pack_fineweb_edu(
        target_tokens=100_000_000,
        out_dir=P("data/fineweb_edu_100m"),
        seed=1337,
        revision="deadbeef",
        dry_run=True,
    )
    assert plan["dry_run"] is True
    assert plan["target_packed_tokens"] == 100_000_000
    assert not P("data/fineweb_edu_100m/tokens.uint16.npy").exists()
