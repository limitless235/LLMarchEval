"""Campaign CLI smoke tests."""

from __future__ import annotations

from unittest.mock import patch

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
