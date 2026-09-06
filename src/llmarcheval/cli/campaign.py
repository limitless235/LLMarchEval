"""CLI: controlled single-variant training campaign entrypoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llmarcheval.config import CONFIGS_DIR, list_variants, load_experiment, variant_config_path
from llmarcheval.train.trainer import train


def _tokens_to_iters(tokens_budget: int, batch_size: int, block_size: int, grad_accum: int) -> int:
    tokens_per_step = batch_size * block_size * grad_accum
    if tokens_per_step <= 0:
        raise ValueError("Invalid tokens_per_step")
    return max(int(math.ceil(tokens_budget / tokens_per_step)), 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled Frontier-SLM training campaign entrypoint "
            "(variant, seed, budget, out-dir, resume)."
        )
    )
    parser.add_argument("--train-config", type=Path, default=CONFIGS_DIR / "local_16gb.yaml")
    parser.add_argument("--variant", type=str, required=True, choices=list_variants())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None, help="Optimizer step budget")
    parser.add_argument(
        "--tokens-budget",
        type=int,
        default=None,
        help="Approximate token budget; converted to max_iters using batch*block*grad_accum",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--scale", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint (model/optimizer/RNG when present)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved experiment config and exit without training",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment(
        variant_config_path(args.variant), args.train_config, scale=args.scale
    )
    if args.seed is not None:
        experiment.train.seed = args.seed
    if args.out_dir is not None:
        experiment.train.out_dir = args.out_dir
    if args.tokens_budget is not None:
        experiment.train.tokens_budget = args.tokens_budget
        experiment.train.max_iters = _tokens_to_iters(
            args.tokens_budget,
            experiment.train.batch_size,
            experiment.model.block_size,
            experiment.train.grad_accum,
        )
    if args.max_iters is not None:
        experiment.train.max_iters = args.max_iters

    payload = {
        "variant": experiment.model.variant,
        "seed": experiment.train.seed,
        "max_iters": experiment.train.max_iters,
        "tokens_budget": experiment.train.tokens_budget,
        "out_dir": experiment.train.out_dir,
        "train_config": str(args.train_config),
        "resume": str(args.resume) if args.resume else None,
        "device": experiment.train.device,
        "dtype": experiment.train.dtype,
        "batch_size": experiment.train.batch_size,
        "grad_accum": experiment.train.grad_accum,
        "block_size": experiment.model.block_size,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run:
        return

    summary = train(experiment, resume_from=args.resume)
    print(
        f"done {args.variant}: first={summary.get('first_loss')} "
        f"last={summary.get('last_loss')} tokens={summary.get('tokens_seen')} "
        f"start_step={summary.get('start_step')}"
    )


if __name__ == "__main__":
    main()
