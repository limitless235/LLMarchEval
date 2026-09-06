"""Focused tests for optional MoE-routing checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from llmarcheval.cli.experiment import main
from llmarcheval.config import ModelConfig
from llmarcheval.experiments.common import build_model, load_probe_experiment, resolve_device, resolve_dtype, set_seed
from llmarcheval.experiments.moe_routing import run_moe_routing_experiment
from llmarcheval.models.transformer import GPT


def _smoke_ckpt(tmp_path: Path, *, fill: float | None = 0.123, step: int = 200) -> Path:
    set_seed(0)
    exp = load_probe_experiment("v1_moe", train_config="configs/smoke.yaml", scale="smoke")
    device = resolve_device("cpu")
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    if fill is not None:
        with torch.no_grad():
            next(model.parameters()).fill_(fill)
    path = tmp_path / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "config": {"model": {}}, "step": step}, path)
    return path


def test_cli_accepts_ckpt_argument():
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "metrics": {}}

    with patch("llmarcheval.cli.experiment.run_moe_routing_experiment", side_effect=_fake):
        with patch("builtins.print"):
            main(
                [
                    "moe-routing",
                    "--ckpt",
                    "results/local_16gb/v1_moe/ckpt_final.pt",
                    "--no-write",
                    "--device",
                    "cpu",
                ]
            )
    assert captured["ckpt"] == "results/local_16gb/v1_moe/ckpt_final.pt"


def test_checkpoint_loading_succeeds_and_replaces_init(tmp_path: Path):
    ckpt = _smoke_ckpt(tmp_path, fill=0.123, step=200)
    # Different seed => different fresh init than the checkpoint source seed.
    record = run_moe_routing_experiment(
        variant="v1_moe",
        scale="smoke",
        seed=1337,
        device_name="cpu",
        train_config="configs/smoke.yaml",
        write=False,
        ckpt=ckpt,
    )
    assert record["metrics"]["checkpoint"] is not None
    assert record["metrics"]["checkpoint"]["name"] == "ckpt_final.pt"
    assert record["metrics"]["checkpoint"]["step"] == 200
    assert record["training_steps"] == 200

    # Reload and confirm weights match the sentinel fill, not a fresh seed-1337 init.
    set_seed(1337)
    exp = load_probe_experiment("v1_moe", train_config="configs/smoke.yaml", scale="smoke")
    device = resolve_device("cpu")
    dtype = resolve_dtype("float32", device)
    fresh = build_model(exp.model, device, dtype)
    loaded = build_model(exp.model, device, dtype)
    blob = torch.load(ckpt, map_location=device)
    loaded.load_state_dict(blob["model"])
    fresh_p = next(fresh.parameters()).detach()
    loaded_p = next(loaded.parameters()).detach()
    assert not torch.allclose(fresh_p, loaded_p)
    assert torch.allclose(loaded_p, torch.full_like(loaded_p, 0.123))


def test_no_checkpoint_behavior_unchanged():
    record = run_moe_routing_experiment(
        variant="v1_moe",
        scale="smoke",
        seed=1337,
        device_name="cpu",
        write=False,
    )
    assert record["metrics"]["checkpoint"] is None
    assert record["training_steps"] is None
    assert any("no checkpoint" in n.lower() or "fresh" in n.lower() for n in record["notes"])
    assert set(record["metrics"]["regimes"]) == {"normal", "repetitive", "structured", "reversed"}


def test_checkpoint_dimension_mismatch_fails_clearly(tmp_path: Path):
    tiny = ModelConfig(
        variant="v1_moe",
        vocab_size=128,
        n_layer=1,
        n_embd=32,
        n_head=4,
        block_size=16,
        use_moe=True,
        n_experts=4,
        n_active=2,
        n_shared_experts=1,
    )
    model = GPT(tiny)
    path = tmp_path / "bad_shape.pt"
    torch.save({"model": model.state_dict(), "config": {}, "step": 1}, path)

    with pytest.raises((RuntimeError, ValueError)):
        run_moe_routing_experiment(
            variant="v1_moe",
            scale="smoke",
            seed=1337,
            device_name="cpu",
            write=False,
            ckpt=path,
        )
