"""Tests for the opt-in local_16gb_w768 (~100M active) scale profile.

Frozen local_16gb (~37M) must remain unchanged; this profile is isolated.
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmarcheval.config import (
    list_variants,
    load_experiment,
    load_model_config,
    load_train_config,
    variant_config_path,
)
from llmarcheval.models.accounting import (
    estimate_from_config,
    estimate_training_memory,
    summarize_model,
)
from llmarcheval.models.transformer import GPT
from llmarcheval.train.checkpoint import (
    build_checkpoint,
    load_model_from_checkpoint,
    model_config_from_checkpoint,
    save_checkpoint,
)

SCALE = "local_16gb_w768"
TRAIN_CFG = "configs/local_16gb_w768.yaml"

# Exact repository-derived counts from the read-only ~100M scale-up audit.
EXPECTED = {
    "v0_dense": {
        "total": 110_582_016,
        "active": 110_582_016,
        "unique_layers": 8,
        "executed_layers": 8,
        "depth_scale": 1.0,
        "loops": 1,
        "n_inner": 2880,
    },
    "v1_moe": {
        "total": 216_799_488,
        "active": 110_631_168,
        "unique_layers": 8,
        "executed_layers": 8,
        "depth_scale": 1.0,
        "loops": 1,
        "n_inner": 2880,
    },
    "v2_moe_mla": {
        "total": 226_666_752,
        "active": 108_775_680,
        "unique_layers": 8,
        "executed_layers": 8,
        "depth_scale": 1.0,
        "loops": 1,
        "n_inner": 3200,
    },
    "v3_moe_mla_dsa": {
        "total": 228_239_616,
        "active": 110_348_544,
        "unique_layers": 8,
        "executed_layers": 8,
        "depth_scale": 1.0,
        "loops": 1,
        "n_inner": 3200,
    },
    "v4_recurrent": {
        "total": 228_239_616,
        "active": 110_348_544,
        "unique_layers": 8,
        "executed_layers": 14,
        "depth_scale": 1.75,
        "loops": 2,
        "n_inner": 3200,
    },
}

# Analytical FP32 memory sketch (B=1, T=512) — not measured peak memory.
EXPECTED_MEMORY = {
    "v0_dense": {
        "estimated_model_bytes": 442_328_064,
        "estimated_grad_bytes": 442_328_064,
        "estimated_adamw_state_bytes": 884_656_128,
        "estimated_model_grad_optimizer_bytes": 1_769_312_256,
        "estimated_attention_score_bytes": 12_582_912,
    },
    "v1_moe": {
        "estimated_model_bytes": 867_197_952,
        "estimated_grad_bytes": 867_197_952,
        "estimated_adamw_state_bytes": 1_734_395_904,
        "estimated_model_grad_optimizer_bytes": 3_468_791_808,
        "estimated_attention_score_bytes": 12_582_912,
    },
    "v2_moe_mla": {
        "estimated_model_bytes": 906_667_008,
        "estimated_grad_bytes": 906_667_008,
        "estimated_adamw_state_bytes": 1_813_334_016,
        "estimated_model_grad_optimizer_bytes": 3_626_668_032,
        "estimated_attention_score_bytes": 12_582_912,
    },
    "v3_moe_mla_dsa": {
        "estimated_model_bytes": 912_958_464,
        "estimated_grad_bytes": 912_958_464,
        "estimated_adamw_state_bytes": 1_825_916_928,
        "estimated_model_grad_optimizer_bytes": 3_651_833_856,
        "estimated_attention_score_bytes": 12_582_912,
    },
    "v4_recurrent": {
        "estimated_model_bytes": 912_958_464,
        "estimated_grad_bytes": 912_958_464,
        "estimated_adamw_state_bytes": 1_825_916_928,
        "estimated_model_grad_optimizer_bytes": 3_651_833_856,
        "estimated_attention_score_bytes": 12_582_912,
    },
}


def test_local_16gb_w768_train_config_is_opt_in():
    cfg = load_train_config(TRAIN_CFG)
    assert cfg.scale == SCALE
    assert cfg.batch_size == 1
    assert cfg.grad_accum == 8
    assert cfg.dtype in {"float32", "fp32"}
    assert cfg.device == "mps"
    assert cfg.allow_dataset_fallback is False
    assert cfg.out_dir == "results/local_16gb_w768"
    # Frozen 37M profile stays the campaign default path when callers use it.
    frozen = load_train_config("configs/local_16gb.yaml")
    assert frozen.scale == "local_16gb"
    assert frozen.out_dir == "results/local_16gb"


def test_local_16gb_w768_profile_construction():
    for variant in list_variants():
        model = load_model_config(variant_config_path(variant), SCALE)
        assert model.n_embd == 768
        assert model.n_layer == 8 or (
            model.use_recurrent and model.unique_layers == 8
        )
        assert model.unique_layers == 8
        assert model.n_head == 12
        assert model.head_dim == 64
        assert model.block_size == 512
        assert model.n_inner == EXPECTED[variant]["n_inner"]
        if variant == "v4_recurrent":
            assert model.n_prelude == 1
            assert model.n_recurrent == 6
            assert model.n_coda == 1
            assert model.train_loops == 2
            assert model.eval_loops == 2
        gpt = GPT(model)
        measured = sum(p.numel() for p in gpt.parameters())
        est = estimate_from_config(model)
        assert measured == est["total_params"]


def test_local_16gb_w768_parameter_accounting_exact():
    actives = []
    for variant, exp in EXPECTED.items():
        model = load_model_config(variant_config_path(variant), SCALE)
        loops = model.train_loops if model.use_recurrent else 1
        est = estimate_from_config(model, loops=loops)
        assert est["total_params"] == exp["total"], variant
        assert est["active_params"] == exp["active"], variant
        assert est["unique_layers"] == exp["unique_layers"], variant
        assert est["executed_layers"] == exp["executed_layers"], variant
        assert est["depth_scale"] == exp["depth_scale"], variant
        assert est["loops"] == exp["loops"], variant
        measured = sum(p.numel() for p in GPT(model).parameters())
        assert measured == exp["total"], variant
        actives.append(est["active_params"])
    lo, hi = min(actives), max(actives)
    spread = (hi - lo) / lo
    assert spread <= 0.02, (lo, hi, spread)


def test_local_16gb_w768_compute_accounting():
    for variant, exp in EXPECTED.items():
        model = load_model_config(variant_config_path(variant), SCALE)
        loops = model.train_loops if model.use_recurrent else 1
        est = estimate_from_config(model, loops=loops)
        assert est["depth_scale"] == exp["depth_scale"]
        assert est["executed_layers"] == exp["executed_layers"]
        assert est["flops_per_token_is_proxy"] is True
        assert est["flops_per_token"] == 2.0 * est["active_params"] * est["depth_scale"]
    v4 = load_model_config(variant_config_path("v4_recurrent"), SCALE)
    est4 = estimate_from_config(v4, loops=v4.train_loops)
    assert est4["executed_layers"] == 14
    assert est4["depth_scale"] == 1.75


def test_local_16gb_w768_analytical_memory_accounting():
    for variant, mem_exp in EXPECTED_MEMORY.items():
        model = load_model_config(variant_config_path(variant), SCALE)
        mem = estimate_training_memory(model, batch_size=1, bytes_per_param=4)
        assert mem["memory_figures_are_estimates"] is True
        for key, value in mem_exp.items():
            assert mem[key] == value, (variant, key, mem[key], value)
        # Keep analytical memory distinct from any measured MPS peak claim.
        assert "peak" not in mem
        assert "measured" not in {k for k in mem if "estimate" not in k and k != "note"}


def test_local_16gb_w768_checkpoint_roundtrip(tmp_path: Path):
    """Tiny config-preserving checkpoint; not a training artifact."""
    variant = "v3_moe_mla_dsa"
    exp = load_experiment(variant_config_path(variant), TRAIN_CFG)
    assert exp.train.scale == SCALE
    assert exp.model.n_embd == 768
    model = GPT(exp.model)
    path = tmp_path / "tiny_w768_ckpt.pt"
    blob = build_checkpoint(
        model=model,
        config=exp,
        step=1,
        steps_completed=1,
        tokens_seen=exp.train.batch_size * exp.model.block_size * exp.train.grad_accum,
    )
    save_checkpoint(path, blob)

    cfg_from_ckpt = model_config_from_checkpoint(blob)
    assert cfg_from_ckpt.n_embd == 768
    assert cfg_from_ckpt.n_head == 12
    assert cfg_from_ckpt.n_layer == 8
    assert cfg_from_ckpt.block_size == 512
    assert cfg_from_ckpt.n_inner == 3200
    assert cfg_from_ckpt.use_moe and cfg_from_ckpt.use_mla and cfg_from_ckpt.use_dsa

    loaded, loaded_exp, meta = load_model_from_checkpoint(
        path, device=torch.device("cpu"), variant=variant
    )
    assert loaded_exp.model.n_embd == 768
    assert loaded_exp.model.n_inner == 3200
    assert loaded_exp.train.scale == SCALE
    assert sum(p.numel() for p in loaded.parameters()) == EXPECTED[variant]["total"]
    assert meta is not None


def test_local_16gb_w768_summarize_matches_estimate():
    exp = load_experiment(variant_config_path("v0_dense"), TRAIN_CFG)
    model = GPT(exp.model)
    summary = summarize_model(model, exp.model, batch_size=exp.train.batch_size)
    assert summary["measured_total_params"] == EXPECTED["v0_dense"]["total"]
    assert summary["memory_estimate"]["memory_figures_are_estimates"] is True
