"""CLI: fixed validation protocol for trained checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmarcheval.eval.protocol import DEFAULT_EVAL_ITERS, DEFAULT_EVAL_SEED, evaluate_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fixed checkpoint evaluation protocol")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--scale", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_EVAL_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    record = evaluate_checkpoint(
        ckpt=args.ckpt,
        variant=args.variant,
        train_config=args.train_config,
        scale=args.scale,
        device_name=args.device,
        eval_iters=args.eval_iters,
        seed=args.seed,
        out_dir=args.out_dir,
        write=not args.no_write,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
