"""Trainer evaluation integrity: persist val metrics, final val, fail-closed."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from llmarcheval.config import ExperimentConfig, ModelConfig, TrainConfig
from llmarcheval.models.transformer import GPT, GPTOutput
from llmarcheval.train.trainer import NonFiniteLossError, train


def _tiny_exp(
    tmp_path: Path,
    *,
    max_iters: int = 6,
    eval_interval: int = 2,
    log_interval: int = 100,
    ckpt_interval: int = 0,
    eval_iters: int = 2,
    seed: int = 0,
) -> ExperimentConfig:
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
            log_interval=log_interval,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            ckpt_interval=ckpt_interval,
            out_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            seed=seed,
            allow_dataset_fallback=True,
        ),
    )


def test_scheduled_validation_is_persisted(tmp_path: Path):
    """A: scheduled validation writes val_loss/val_ppl into metrics/history."""
    exp = _tiny_exp(tmp_path, max_iters=5, eval_interval=2, log_interval=100)
    summary = train(exp)
    val_rows = [r for r in summary["history"] if "val_loss" in r]
    assert val_rows, "expected persisted validation rows in history"
    assert all("val_ppl" in r for r in val_rows)
    assert all(r["step"] > 0 and r["step"] % 2 == 0 for r in val_rows)

    metrics_path = Path(exp.train.out_dir) / exp.model.variant / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    metric_val = [r for r in lines if "val_loss" in r]
    assert metric_val
    for row in metric_val:
        assert math.isfinite(row["val_loss"])
        assert row["val_ppl"] == pytest.approx(math.exp(row["val_loss"]))


def test_final_validation_even_when_not_eval_interval(tmp_path: Path):
    """B/C/D: final val always runs; summary has final_val_loss/ppl = exp(loss)."""
    # max_iters=5 → steps 0..4; eval_interval=2 → scheduled at 2,4.
    # Final step index 4 IS a scheduled point; use max_iters=5 with interval 3
    # so scheduled at step 3 only, final after step 4 (not a scheduled point).
    exp = _tiny_exp(tmp_path, max_iters=5, eval_interval=3, log_interval=100)
    summary = train(exp)

    scheduled_steps = {r["step"] for r in summary["history"] if "val_loss" in r}
    assert 3 in scheduled_steps
    assert 4 not in scheduled_steps  # final step is not a scheduled eval point

    assert "final_val_loss" in summary
    assert "final_val_ppl" in summary
    assert summary["final_val_loss"] is not None
    assert math.isfinite(summary["final_val_loss"])
    assert summary["final_val_ppl"] == pytest.approx(math.exp(summary["final_val_loss"]))

    disk = json.loads(
        (Path(exp.train.out_dir) / exp.model.variant / "summary.json").read_text(encoding="utf-8")
    )
    assert disk["final_val_loss"] == pytest.approx(summary["final_val_loss"])
    assert disk["final_val_ppl"] == pytest.approx(math.exp(disk["final_val_loss"]))


def test_final_validation_not_faked_from_scheduled(tmp_path: Path):
    """Final validation is a distinct pass (not copied from last scheduled val)."""
    exp = _tiny_exp(tmp_path, max_iters=5, eval_interval=3, eval_iters=2)
    calls: list[str] = []
    real_estimate = None

    import llmarcheval.train.trainer as trainer_mod

    real_estimate = trainer_mod.estimate_loss

    def tracking_estimate(*args, **kwargs):
        calls.append("estimate")
        return real_estimate(*args, **kwargs)

    with patch.object(trainer_mod, "estimate_loss", side_effect=tracking_estimate):
        summary = train(exp)

    # One scheduled (step 3) + one final after last step.
    assert len(calls) == 2
    assert summary["final_val_loss"] is not None


def test_nonfinite_train_loss_aborts(tmp_path: Path):
    """E: non-finite training loss raises and does not report success."""
    exp = _tiny_exp(tmp_path, max_iters=2, eval_interval=0, log_interval=1)
    real_forward = GPT.forward

    def nan_forward(self, idx, targets=None, loops=None):
        out = real_forward(self, idx, targets, loops=loops)
        if out.loss is not None:
            # Preserve autograd while forcing a non-finite training loss.
            return GPTOutput(
                logits=out.logits,
                loss=out.loss * float("nan"),
                aux_loss=out.aux_loss,
            )
        return out

    with patch.object(GPT, "forward", nan_forward):
        with pytest.raises(NonFiniteLossError, match="training"):
            train(exp)

    summary_path = Path(exp.train.out_dir) / exp.model.variant / "summary.json"
    assert not summary_path.exists()


def test_nonfinite_validation_loss_aborts(tmp_path: Path):
    """F: non-finite validation loss aborts before a successful summary."""
    exp = _tiny_exp(tmp_path, max_iters=2, eval_interval=0, log_interval=100)

    import llmarcheval.train.trainer as trainer_mod

    with patch.object(trainer_mod, "estimate_loss", return_value=float("nan")):
        with pytest.raises(NonFiniteLossError, match="validation"):
            train(exp)

    summary_path = Path(exp.train.out_dir) / exp.model.variant / "summary.json"
    assert not summary_path.exists()


def test_resume_still_gets_final_validation(tmp_path: Path):
    """G-adjacent: resume that reaches the final step still runs final validation."""
    exp = _tiny_exp(
        tmp_path / "a",
        max_iters=4,
        eval_interval=0,
        ckpt_interval=2,
        log_interval=100,
    )
    train(exp)
    mid = Path(exp.train.out_dir) / exp.model.variant / "ckpt_2.pt"
    assert mid.exists()

    exp2 = _tiny_exp(
        tmp_path / "b",
        max_iters=4,
        eval_interval=0,
        ckpt_interval=0,
        log_interval=100,
        seed=0,
    )
    summary = train(exp2, resume_from=mid)
    assert summary["start_step"] == 3
    assert summary["final_val_loss"] is not None
    assert summary["final_val_ppl"] == pytest.approx(math.exp(summary["final_val_loss"]))
