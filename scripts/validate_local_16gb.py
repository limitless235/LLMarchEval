#!/usr/bin/env python3
"""Short local_16gb validation — not a training job.

Checks init / forward / backward / optimizer / finite loss trajectory, and that
train_loops affects the recurrent variant. Uses the bundled corpus so the check
runs offline; research runs should keep TinyStories with fallback disabled.
"""

from __future__ import annotations

import argparse
import math

import torch

from llmarcheval.config import CONFIGS_DIR, list_variants, load_experiment, variant_config_path
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import train


def _check_train_loops(device: torch.device) -> None:
    exp = load_experiment(variant_config_path("v4_recurrent"), CONFIGS_DIR / "local_16gb.yaml")
    model = GPT(exp.model).to(device)
    model.train()
    idx = torch.randint(0, exp.model.vocab_size, (1, min(32, exp.model.block_size)), device=device)
    out1 = model(idx, loops=1)
    out2 = model(idx, loops=exp.model.train_loops)
    assert out1.stats["loops"] == 1
    assert out2.stats["loops"] == exp.model.train_loops
    assert out2.stats["n_blocks_executed"] > out1.stats["n_blocks_executed"]
    assert exp.model.train_loops >= 2
    print(
        f"train_loops check ok: loops 1 -> {out1.stats['n_blocks_executed']} blocks, "
        f"loops {exp.model.train_loops} -> {out2.stats['n_blocks_executed']} blocks",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate local_16gb profile (short run)")
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--variant", type=str, default="v0_dense")
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args(argv)

    variants = list_variants() if args.all_variants else [args.variant]

    results = []
    for variant in variants:
        experiment = load_experiment(
            variant_config_path(variant),
            CONFIGS_DIR / "local_16gb.yaml",
        )
        experiment.train.max_iters = args.max_iters
        experiment.train.dataset = "corpus"
        experiment.train.log_interval = max(args.max_iters - 1, 1)
        experiment.train.eval_interval = 0
        experiment.train.ckpt_interval = 0
        experiment.train.out_dir = "results/local_16gb_validate"
        assert experiment.train.allow_dataset_fallback is False
        assert experiment.train.dtype in {"float32", "fp32"}
        assert experiment.model.block_size == 512
        summary = train(experiment)
        assert summary["first_loss"] is not None and math.isfinite(summary["first_loss"])
        assert summary["last_loss"] is not None and math.isfinite(summary["last_loss"])
        assert "runtime_diagnostics" in summary
        results.append(summary)
        print(
            f"validate {variant}: first={summary['first_loss']:.4f} "
            f"last={summary['last_loss']:.4f} device={summary['device']}",
            flush=True,
        )

    device = torch.device(results[-1]["device"])
    _check_train_loops(device)
    print("local_16gb validation passed", flush=True)


if __name__ == "__main__":
    main()
