"""Regression: probes must rebuild from checkpoint architecture, not smoke dims.

Physical Mac failure mode: local_16gb (n_embd=384) checkpoints loaded into
smoke (n_embd=256) probe models → load_state_dict size mismatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llmarcheval.config import load_experiment, variant_config_path
from llmarcheval.eval.protocol import evaluate_checkpoint
from llmarcheval.experiments.common import (
    build_model,
    load_probe_model,
    resolve_device,
    resolve_dtype,
    set_seed,
)
from llmarcheval.experiments.dsa_needle import run_dsa_needle_experiment
from llmarcheval.experiments.mla_context import run_mla_context_experiment
from llmarcheval.experiments.recurrent_loops import run_recurrent_loops_experiment
from llmarcheval.train.checkpoint import (
    build_checkpoint,
    load_checkpoint_blob,
    load_model_from_checkpoint,
    model_config_from_checkpoint,
    save_checkpoint,
)


def _local_16gb_ckpt(variant: str, tmp_path: Path, *, step: int = 5) -> tuple[Path, object]:
    """Create a local_16gb-scale checkpoint (384-d) like the Mac smoke trains."""
    set_seed(0)
    exp = load_experiment(variant_config_path(variant), "configs/local_16gb.yaml")
    assert exp.model.n_embd == 384
    assert exp.train.scale == "local_16gb"
    device = resolve_device("cpu")
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    path = tmp_path / f"{variant}_local16gb_ckpt.pt"
    blob = build_checkpoint(
        model=model,
        config=exp,
        step=step,
        steps_completed=step,
        tokens_seen=step
        * exp.train.batch_size
        * exp.model.block_size
        * exp.train.grad_accum,
    )
    save_checkpoint(path, blob)
    return path, exp.model


@pytest.mark.parametrize("variant", ["v2_moe_mla", "v3_moe_mla_dsa", "v4_recurrent"])
def test_local_16gb_ckpt_reconstructs_384_not_smoke_256(variant: str, tmp_path: Path):
    ckpt, trained = _local_16gb_ckpt(variant, tmp_path)
    device = resolve_device("cpu")
    dtype = resolve_dtype("float32", device)
    # Deliberately pass smoke YAML/scale: with ckpt set, architecture must come from ckpt.
    model, exp, meta = load_probe_model(
        variant=variant,
        device=device,
        dtype=dtype,
        train_config="configs/smoke.yaml",
        scale="smoke",
        ckpt=ckpt,
    )
    assert exp.model.n_embd == 384
    assert exp.model.n_embd == trained.n_embd
    assert exp.model.n_layer == trained.n_layer
    assert exp.model.n_head == trained.n_head
    assert exp.model.vocab_size == trained.vocab_size
    assert exp.model.block_size == trained.block_size
    assert model.wte.weight.shape == (trained.vocab_size, 384)
    assert meta is not None
    assert meta["step"] == 5


def test_v3_dsa_probe_loads_local_16gb_ckpt(tmp_path: Path):
    ckpt, trained = _local_16gb_ckpt("v3_moe_mla_dsa", tmp_path)
    record = run_dsa_needle_experiment(
        write=False,
        device_name="cpu",
        haystack_lens=(64,),
        answer_tokens=2,
        train_config="configs/smoke.yaml",
        scale="smoke",
        ckpt=ckpt,
    )
    assert record["metrics"]["checkpoint"]["step"] == 5
    assert record["config"]["model"]["n_embd"] == 384
    assert record["config"]["model"]["n_embd"] == trained.n_embd
    assert record["config"]["model"]["n_head"] == trained.n_head
    assert record["config"]["model"]["use_dsa"] is True
    assert record["config"]["model"]["use_mla"] is True
    assert record["config"]["model"]["index_topk"] == trained.index_topk
    assert record["training_steps"] == 5


def test_v4_recurrent_probe_loads_local_16gb_ckpt(tmp_path: Path):
    ckpt, trained = _local_16gb_ckpt("v4_recurrent", tmp_path)
    record = run_recurrent_loops_experiment(
        write=False,
        device_name="cpu",
        loops=(1, 2),
        train_config="configs/smoke.yaml",
        scale="smoke",
        ckpt=ckpt,
    )
    assert record["metrics"]["checkpoint"]["step"] == 5
    assert record["config"]["model"]["n_embd"] == 384
    assert record["config"]["model"]["n_prelude"] == trained.n_prelude
    assert record["config"]["model"]["n_recurrent"] == trained.n_recurrent
    assert record["config"]["model"]["n_coda"] == trained.n_coda


def test_v2_mla_probe_loads_local_16gb_ckpt(tmp_path: Path):
    ckpt, trained = _local_16gb_ckpt("v2_moe_mla", tmp_path)
    record = run_mla_context_experiment(
        variants=["v2_moe_mla"],
        contexts=[64],
        steps=1,
        write=False,
        device_name="cpu",
        train_config="configs/smoke.yaml",
        scale="smoke",
        ckpts={"v2_moe_mla": ckpt},
    )
    # Top-level record must be checkpoint-authoritative (not smoke 256-d).
    model_cfg = record["config"]["model"]
    assert model_cfg["n_embd"] == 384
    assert model_cfg["n_layer"] == 8
    assert model_cfg["n_head"] == 6
    assert model_cfg["use_moe"] is True
    assert model_cfg["use_mla"] is True
    assert model_cfg["n_embd"] == trained.n_embd
    assert model_cfg["n_layer"] == trained.n_layer
    assert model_cfg["n_head"] == trained.n_head

    assert record["parameter_counts"]["total"] == 64_728_192
    assert record["parameter_counts"]["active"] == 36_416_640
    # Regression: must not claim smoke dims while reporting local_16gb counts.
    assert record["config"]["model"]["n_embd"] != 256

    ckpt_meta = record["metrics"]["checkpoint"]
    assert ckpt_meta is not None
    assert ckpt_meta["path"] == str(ckpt)
    assert ckpt_meta["name"] == ckpt.name
    assert ckpt_meta["step"] == 5
    assert ckpt_meta["tokens_seen"] is not None
    assert ckpt_meta["checkpoint_version"] is not None
    assert "has_optimizer" in ckpt_meta
    assert "has_rng" in ckpt_meta
    assert record["training_steps"] == 5

    rows = record["metrics"]["rows"]
    assert len(rows) == 1
    assert rows[0]["parameter_counts"]["total"] == 64_728_192
    assert rows[0]["parameter_counts"]["active"] == 36_416_640
    assert rows[0]["model_config"]["n_embd"] == 384
    assert rows[0]["checkpoint"]["step"] == 5
    assert rows[0]["loss_finite"] is True
    assert rows[0]["analytical_kv_bytes_are_theoretical"] is True

    loaded, exp, _ = load_model_from_checkpoint(
        ckpt, device=resolve_device("cpu"), variant="v2_moe_mla"
    )
    assert exp.model.n_embd == 384
    assert exp.model.kv_lora_rank == trained.kv_lora_rank
    assert exp.model.qk_rope_head_dim == trained.qk_rope_head_dim
    assert loaded.wte.weight.shape[-1] == 384


def test_mla_context_no_checkpoint_still_uses_smoke(tmp_path: Path):
    record = run_mla_context_experiment(
        variants=["v2_moe_mla"],
        contexts=[32],
        steps=1,
        write=False,
        device_name="cpu",
        train_config="configs/smoke.yaml",
        scale="smoke",
    )
    assert record["config"]["model"]["n_embd"] == 256
    assert record["metrics"]["checkpoint"] is None
    assert record["metrics"]["rows"][0]["checkpoint"] is None
    assert record["parameter_counts"]["total"] < 40_000_000


def test_mla_context_multi_variant_with_one_ckpt_keeps_per_row_provenance(tmp_path: Path):
    ckpt, _trained = _local_16gb_ckpt("v2_moe_mla", tmp_path)
    record = run_mla_context_experiment(
        variants=["v1_moe", "v2_moe_mla"],
        contexts=[32],
        steps=1,
        write=False,
        device_name="cpu",
        train_config="configs/smoke.yaml",
        scale="smoke",
        ckpts={"v2_moe_mla": ckpt},
    )
    # Multi-variant: do not invent a single checkpoint-authoritative aggregate.
    assert record["metrics"]["checkpoint"] is None
    by_variant = {r["variant"]: r for r in record["metrics"]["rows"]}
    assert by_variant["v1_moe"]["checkpoint"] is None
    assert by_variant["v1_moe"]["model_config"]["n_embd"] == 256
    assert by_variant["v2_moe_mla"]["checkpoint"]["step"] == 5
    assert by_variant["v2_moe_mla"]["model_config"]["n_embd"] == 384
    assert by_variant["v2_moe_mla"]["parameter_counts"]["total"] == 64_728_192


def test_eval_protocol_uses_checkpoint_384_not_smoke(tmp_path: Path):
    ckpt, trained = _local_16gb_ckpt("v3_moe_mla_dsa", tmp_path)
    record = evaluate_checkpoint(
        ckpt=ckpt,
        train_config="configs/smoke.yaml",
        scale="smoke",
        device_name="cpu",
        eval_iters=1,
        seed=1,
        out_dir=tmp_path / "eval",
        write=False,
    )
    assert record["model_config"]["n_embd"] == 384
    assert record["model_config"]["n_layer"] == trained.n_layer
    assert record["model_config"]["n_head"] == trained.n_head
    assert record["model_config"]["vocab_size"] == trained.vocab_size


def test_missing_architecture_metadata_fails_clearly(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"model": {"wte.weight": torch.zeros(2, 2)}, "step": 1}, path)
    with pytest.raises(ValueError, match="authoritative architecture metadata"):
        model_config_from_checkpoint(load_checkpoint_blob(path))

    path2 = tmp_path / "partial.pt"
    torch.save(
        {
            "model": {"wte.weight": torch.zeros(2, 2)},
            "config": {"model": {"variant": "v0_dense", "n_embd": 384}, "train": {}},
            "step": 1,
        },
        path2,
    )
    with pytest.raises(ValueError, match="insufficient"):
        model_config_from_checkpoint(load_checkpoint_blob(path2))


def test_legacy_smoke_ckpt_still_loads(tmp_path: Path):
    """Backward compat: legacy-shaped smoke checkpoints keep working."""
    set_seed(0)
    exp = load_experiment(variant_config_path("v0_dense"), "configs/smoke.yaml", scale="smoke")
    device = resolve_device("cpu")
    model = build_model(exp.model, device, resolve_dtype("float32", device))
    path = tmp_path / "legacy_smoke.pt"
    torch.save({"model": model.state_dict(), "config": exp.to_dict(), "step": 7}, path)
    loaded, loaded_exp, meta = load_model_from_checkpoint(
        path, device=device, variant="v0_dense"
    )
    assert meta["step"] == 7
    assert loaded_exp.model.n_embd == 256
    assert loaded.wte.weight.shape[-1] == 256
