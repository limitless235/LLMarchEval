"""Tests for the local_16gb research profile."""

from __future__ import annotations

import torch

from llmarcheval.config import (
    list_variants,
    load_experiment,
    load_model_config,
    load_train_config,
    variant_config_path,
)
from llmarcheval.models.accounting import estimate_from_config, summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import auto_device, auto_dtype, runtime_diagnostics


def test_local_16gb_train_config_loads():
    cfg = load_train_config("configs/local_16gb.yaml")
    assert cfg.scale == "local_16gb"
    assert cfg.batch_size == 1
    assert cfg.grad_accum >= 2
    assert cfg.allow_dataset_fallback is False
    assert cfg.dtype in {"float32", "fp32"}
    assert cfg.device == "mps"


def test_local_16gb_param_range_and_components():
    actives = []
    for variant in list_variants():
        model = load_model_config(variant_config_path(variant), "local_16gb")
        assert model.block_size == 512
        assert model.n_embd == 384
        assert model.n_head == 6
        est = estimate_from_config(model)
        assert 30_000_000 <= est["active_params"] <= 50_000_000, (variant, est["active_params"])
        actives.append(est["active_params"])
        assert "components" in est
        assert est["components"]["embedding"] > 0
    lo, hi = min(actives), max(actives)
    assert (hi - lo) / lo <= 0.03, (lo, hi)


def test_local_16gb_all_variants_instantiate():
    for variant in list_variants():
        cfg = load_model_config(variant_config_path(variant), "local_16gb")
        model = GPT(cfg)
        measured = sum(p.numel() for p in model.parameters())
        est = estimate_from_config(cfg)
        assert measured == est["total_params"]


def test_local_16gb_train_loops_wired():
    cfg = load_model_config(variant_config_path("v4_recurrent"), "local_16gb")
    assert cfg.use_recurrent
    assert cfg.train_loops >= 2
    assert cfg.unique_layers == 8
    assert cfg.n_prelude + cfg.n_recurrent + cfg.n_coda == 8
    model = GPT(cfg)
    model.train()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(idx, loops=cfg.train_loops)
    assert out.stats["loops"] == cfg.train_loops
    exp = load_experiment(variant_config_path("v4_recurrent"), "configs/local_16gb.yaml")
    assert exp.model.train_loops == cfg.train_loops


def test_mps_device_falls_back_to_cpu_when_unavailable(monkeypatch):
    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(torch.backends, "mps", _Mps())
    device = auto_device("mps")
    assert device.type == "cpu"
    dtype = auto_dtype("float32", device)
    assert dtype == torch.float32


def test_runtime_diagnostics_fields():
    exp = load_experiment(variant_config_path("v0_dense"), "configs/local_16gb.yaml")
    model = GPT(exp.model)
    accounting = summarize_model(model, exp.model, batch_size=exp.train.batch_size)
    diag = runtime_diagnostics(exp, torch.device("cpu"), torch.float32, accounting)
    assert diag["dtype"] == "float32"
    assert diag["batch_size"] == 1
    assert diag["grad_accum"] == exp.train.grad_accum
    assert diag["context_length"] == 512
    assert diag["estimated_tokens_per_step"] == (
        exp.train.batch_size * 512 * exp.train.grad_accum
    )
    assert diag["configured_training_steps"] == exp.train.max_iters
    assert diag["allow_dataset_fallback"] is False
    assert diag["memory_figures_are_estimates"] is True
    assert diag["total_params"] == accounting["measured_total_params"]
