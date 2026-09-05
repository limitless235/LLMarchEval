"""CLI: train one or all variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from llmarcheval.config import CONFIGS_DIR, list_variants, load_experiment, variant_config_path
from llmarcheval.train.trainer import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Frontier-SLM variant")
    parser.add_argument("--train-config", type=Path, default=CONFIGS_DIR / "smoke.yaml")
    parser.add_argument("--variant", type=str, default="v0_dense")
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--scale", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args(argv)

    variants = list_variants() if args.all_variants else [args.variant]
    for variant in variants:
        experiment = load_experiment(variant_config_path(variant), args.train_config, scale=args.scale)
        if args.max_iters is not None:
            experiment.train.max_iters = args.max_iters
        summary = train(experiment)
        dropped = summary["loss_dropped"]
        print(
            f"done {variant}: first={summary['first_loss']:.4f} "
            f"last={summary['last_loss']:.4f} dropped={dropped}"
        )


if __name__ == "__main__":
    main()
