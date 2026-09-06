"""Campaign CLI smoke tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from llmarcheval.cli.campaign import main


def test_campaign_dry_run_resolves_config():
    with patch("builtins.print") as printer:
        main(
            [
                "--variant",
                "v0_dense",
                "--train-config",
                "configs/smoke.yaml",
                "--max-iters",
                "3",
                "--seed",
                "7",
                "--out-dir",
                "results/campaign_test",
                "--dry-run",
            ]
        )
    printed = " ".join(str(c) for c in printer.call_args_list)
    assert "v0_dense" in printed
    assert "max_iters" in printed


def test_campaign_passes_resume(tmp_path):
    captured = {}

    def _fake(experiment, resume_from=None):
        captured["variant"] = experiment.model.variant
        captured["resume_from"] = resume_from
        captured["max_iters"] = experiment.train.max_iters
        captured["seed"] = experiment.train.seed
        return {"first_loss": 1.0, "last_loss": 0.9, "tokens_seen": 10, "start_step": 0}

    with patch("llmarcheval.cli.campaign.train", side_effect=_fake):
        with patch("builtins.print"):
            main(
                [
                    "--variant",
                    "v1_moe",
                    "--train-config",
                    "configs/smoke.yaml",
                    "--max-iters",
                    "5",
                    "--seed",
                    "11",
                    "--resume",
                    str(tmp_path / "ckpt_final.pt"),
                ]
            )
    assert captured["variant"] == "v1_moe"
    assert captured["max_iters"] == 5
    assert captured["seed"] == 11
    assert str(captured["resume_from"]).endswith("ckpt_final.pt")


def test_campaign_main_end_to_end_persists_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """campaign.main → real train() must persist scheduled + final validation artifacts."""
    out_dir = tmp_path / "campaign_out"
    train_yaml = tmp_path / "tiny_train.yaml"
    train_yaml.write_text(
        "\n".join(
            [
                "scale: smoke",
                "dataset: corpus",
                "max_iters: 5",
                "batch_size: 2",
                "grad_accum: 1",
                "lr: 3.0e-4",
                "warmup_iters: 0",
                "min_lr: 3.0e-5",
                "weight_decay: 0.1",
                "grad_clip: 1.0",
                "log_interval: 100",
                "eval_interval: 3",
                "eval_iters: 2",
                "ckpt_interval: 0",
                "device: cpu",
                "dtype: float32",
                "seed: 0",
                f"out_dir: {out_dir}",
                "allow_dataset_fallback: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    # Tiny model so this E2E path stays CPU-cheap; still exercises real train().
    model_yaml = tmp_path / "v0_dense.yaml"
    model_yaml.write_text(
        "\n".join(
            [
                "variant: v0_dense",
                "use_moe: false",
                "use_mla: false",
                "use_dsa: false",
                "use_recurrent: false",
                "tie_weights: true",
                "dropout: 0.0",
                "bias: false",
                "scales:",
                "  smoke:",
                "    n_layer: 1",
                "    n_embd: 32",
                "    n_head: 4",
                "    block_size: 16",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "llmarcheval.cli.campaign.variant_config_path",
        lambda _variant: model_yaml,
    )

    main(
        [
            "--variant",
            "v0_dense",
            "--train-config",
            str(train_yaml),
            "--max-iters",
            "5",
            "--seed",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )

    metrics_path = out_dir / "v0_dense" / "metrics.jsonl"
    summary_path = out_dir / "v0_dense" / "summary.json"
    assert metrics_path.is_file()
    assert summary_path.is_file()

    metrics_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [row for row in metrics_rows if "val_loss" in row]
    assert val_rows, "expected at least one persisted validation row in metrics.jsonl"
    assert {row["step"] for row in val_rows} == {3}
    for row in val_rows:
        assert math.isfinite(row["val_loss"])
        assert math.isfinite(row["val_ppl"])
        assert row["val_ppl"] == pytest.approx(math.exp(row["val_loss"]))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # campaign.main must not discard trainer-written summary fields
    assert "first_loss" in summary and summary["first_loss"] is not None
    assert "last_loss" in summary and summary["last_loss"] is not None
    assert "tokens_seen" in summary and summary["tokens_seen"] > 0
    assert "final_val_loss" in summary
    assert "final_val_ppl" in summary
    assert summary["final_val_loss"] is not None
    assert summary["final_val_ppl"] is not None
    assert math.isfinite(summary["final_val_loss"])
    assert math.isfinite(summary["final_val_ppl"])
    assert summary["final_val_ppl"] == pytest.approx(math.exp(summary["final_val_loss"]))
    # Final validation is separate from scheduled step-3 validation (not a metrics row at step 4/5).
    assert 4 not in {row["step"] for row in val_rows}
    assert 5 not in {row["step"] for row in val_rows}
    assert summary.get("max_iters") == 5
